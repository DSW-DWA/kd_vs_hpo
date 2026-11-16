"""Restore completed trials from atomic CSV snapshots after a server restart."""

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna

from kd_vs_hpo.hpo.config import HPOExperimentConfig


def validate_resume(experiment: HPOExperimentConfig) -> None:
    # Imported/distributed studies have a different journal and recovery layout.
    if experiment.imported_trials or experiment.optuna.pruners != ("none",):
        raise ValueError("Resume currently supports experiments without imports or pruning")
    path = experiment.output_dir / "run_config.json"
    saved = json.loads(path.read_text())["experiment"]
    current = json.loads(json.dumps(asdict(experiment), default=str))
    saved.pop("resume", None)
    current.pop("resume", None)
    if saved != current:
        raise ValueError("Resume config differs from saved run_config.json")
    if not (experiment.output_dir / "recovery").is_dir():
        raise FileNotFoundError("No recovery directory; this experiment may already be complete")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    integers = {
        "trial_id", "trial_seed", "batch_size", "completed_epochs", "best_epoch",
        "train_flops", "validation_flops", "total_flops", "epoch",
        "cumulative_trial_flops",
    }
    floats = {
        "lr", "weight_decay", "momentum", "best_val_acc1", "trial_seconds",
        "train_loss", "val_acc1", "learning_rate",
    }
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if value != "" and key in integers:
                row[key] = int(value)
            elif value != "" and key in floats:
                row[key] = float(value)
    return rows


def restore_study(
    study: optuna.Study,
    experiment: HPOExperimentConfig,
    directory: Path,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    rows = _read_csv(directory / "trials.csv")
    epochs = _read_csv(directory / "epoch_metrics.csv")
    search = experiment.search_space
    distributions = {
        "lr": optuna.distributions.FloatDistribution(*search.lr, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(*search.weight_decay, log=True),
        "batch_size": optuna.distributions.IntDistribution(
            *search.batch_size, step=search.batch_size_step
        ),
        "momentum": optuna.distributions.FloatDistribution(*search.momentum),
    }
    records = []
    retry = None
    for row in sorted(rows, key=lambda row: row["trial_id"]):
        if row["trial_id"] != len(records):
            raise ValueError("Recovery trial IDs must be contiguous, starting from zero")
        parameters = {key: row[key] for key in distributions}
        if row["state"] == "INTERRUPTED":
            if row is not rows[-1]:
                raise ValueError("Only the last trial may be interrupted")
            retry = parameters
            break
        if row["state"] not in {"COMPLETE", "FAIL"}:
            raise ValueError(f"Unsupported recovery state: {row['state']}")
        if row["state"] == "COMPLETE":
            checkpoint = (experiment.output_dir / "checkpoints" / metadata["study_name"]
                          / f"trial_{row['trial_id']:03d}.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            row["checkpoint_path"] = str(checkpoint)
        trial_epochs = [e for e in epochs if e["trial_id"] == row["trial_id"]]
        if row["state"] == "COMPLETE" and len(trial_epochs) != row["completed_epochs"]:
            raise ValueError("Recovery epoch snapshot is incomplete; restore the backup")
        state = optuna.trial.TrialState[row["state"]]
        study.add_trial(optuna.trial.create_trial(
            state=state,
            value=row["best_val_acc1"] if state == optuna.trial.TrialState.COMPLETE else None,
            params=parameters,
            distributions=distributions,
            intermediate_values={e["epoch"]: e["val_acc1"] for e in trial_epochs},
        ))
        records.append({**metadata, **row})
    kept_epochs = [e for e in epochs if e["trial_id"] < len(records)]
    logging.info("%s: restored %d trials; %d remaining", metadata["study_name"],
                 len(records), max(0, experiment.optuna.n_trials - len(records)))
    return records, kept_epochs, retry
