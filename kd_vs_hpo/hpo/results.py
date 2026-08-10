from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from kd_vs_hpo.common.flops import CounterMode, FlopsBudgetTracker, count_flops_params
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.hpo.config import HPOExperimentConfig
from kd_vs_hpo.hpo.persistence import TRIAL_METADATA_FIELDS
from kd_vs_hpo.hpo.training import build_lightning_module, validate_lightning_module


@dataclass(frozen=True)
class HPOExperimentResult:
    epochs: pd.DataFrame
    trials: pd.DataFrame
    studies: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path


def build_experiment_result(
    *,
    trial_records: list[dict[str, Any]],
    epoch_records: list[dict[str, Any]],
    costs: dict[int, dict[str, Any]],
    n_test: int,
    test_loader: DataLoader,
    evaluation_device: torch.device,
    output_dir: Path,
    experiment: HPOExperimentConfig,
) -> HPOExperimentResult:
    raw_trials = pd.DataFrame(trial_records).sort_values(
        ["study_name", "trial_id"],
        ignore_index=True,
    )
    epochs = pd.DataFrame(epoch_records).sort_values(
        ["study_name", "trial_id", "epoch"],
        ignore_index=True,
    )
    studies = build_study_summary(raw_trials)

    summary = build_architecture_summary(
        raw_trials,
        test_loader,
        evaluation_device,
        experiment,
        output_dir,
    )
    if not summary.empty:
        summary["test_flops"] = summary["arch_row"].map(
            lambda row: int(costs[int(row)]["forward_flops_per_sample"]) * n_test
        )
        summary["total_experiment_flops"] = (
            summary["total_hpo_flops"] + summary["test_flops"]
        )

    return HPOExperimentResult(
        epochs=epochs,
        trials=raw_trials.drop(columns=list(TRIAL_METADATA_FIELDS)),
        studies=studies,
        summary=summary,
        output_dir=output_dir,
    )


def build_study_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study_name, group in trials.groupby("study_name", sort=False):
        complete = group.loc[group["state"] == "COMPLETE"]
        best = complete.sort_values("best_val_acc1").tail(1)
        base = group.iloc[0]
        rows.append(
            {
                "study_name": study_name,
                "sampler": base["sampler"],
                "pruner": base["pruner"],
                "arch_row": base["arch_row"],
                "arch_index": base["arch_index"],
                "complete_trials": int((group["state"] == "COMPLETE").sum()),
                "pruned_trials": int((group["state"] == "PRUNED").sum()),
                "total_train_flops": int(group["train_flops"].sum()),
                "total_validation_flops": int(group["validation_flops"].sum()),
                "total_study_flops": int(group["total_flops"].sum()),
                "best_val_acc1": (
                    float(best.iloc[0]["best_val_acc1"]) if not best.empty else None
                ),
                "best_trial_id": int(best.iloc[0]["trial_id"])
                if not best.empty
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["arch_row", "sampler", "pruner"],
        ignore_index=True,
    )


def build_architecture_summary(
    trials: pd.DataFrame,
    test_loader: DataLoader,
    device: torch.device,
    experiment: HPOExperimentConfig,
    output_dir: Path,
) -> pd.DataFrame:
    complete = trials.loc[trials["state"] == "COMPLETE"]
    if complete.empty:
        return pd.DataFrame()
    best = (
        complete.sort_values(
            [
                "arch_row",
                "best_val_acc1",
                "total_flops",
                "study_name",
                "trial_id",
            ],
            ascending=[True, False, True, True, True],
        )
        .groupby("arch_row", as_index=False, sort=False)
        .head(1)
        .sort_values(
            ["best_val_acc1", "arch_row"],
            ascending=[False, True],
        )
        .copy()
    )
    architecture_flops = trials.groupby("arch_row")["total_flops"].sum()
    best = best.join(architecture_flops.rename("total_hpo_flops"), on="arch_row")
    best["test_acc1"] = [
        _evaluate_checkpoint(path, test_loader, device, experiment, output_dir)
        for path in best["checkpoint_path"]
    ]
    return best


def _evaluate_checkpoint(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    experiment: HPOExperimentConfig,
    output_dir: Path,
) -> float:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = state["arch_record"]
    model = create_nats_model(architecture)
    model.load_state_dict(state["model"])
    forward_flops_per_sample, _ = count_flops_params(model)
    tracker = FlopsBudgetTracker(
        budget=forward_flops_per_sample * len(test_loader.dataset),
        mode=CounterMode.SILENT,
    )
    lightning_module = build_lightning_module(
        model=model,
        architecture=architecture,
        train_config=experiment.train,
        lr=experiment.search_space.initial_lr,
        weight_decay=experiment.search_space.initial_weight_decay,
        max_epochs=1,
        forward_flops_per_sample=forward_flops_per_sample,
        flops_tracker=tracker,
    )
    metrics = validate_lightning_module(
        lightning_module=lightning_module,
        val_loader=test_loader,
        run_name=f"final_validation_arch_{architecture['arch_index']}",
        checkpoint_dir=output_dir / "validation_checkpoints",
        log_dir=output_dir / "validation_logs",
        deterministic=experiment.train.deterministic,
        amp=experiment.train.amp,
        grad_clip_norm=experiment.train.grad_clip_norm,
    )
    return 100.0 * metrics["val_acc"]
