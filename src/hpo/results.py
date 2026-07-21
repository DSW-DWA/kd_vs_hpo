from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.nats import create_nats_model
from src.hpo.persistence import TRIAL_METADATA_FIELDS
from src.hpo.training import evaluate


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
        _evaluate_checkpoint(path, test_loader, device)
        for path in best["checkpoint_path"]
    ]
    return best


def _evaluate_checkpoint(
    checkpoint_path: str, test_loader: DataLoader, device: torch.device
) -> float:
    state = torch.load(checkpoint_path, map_location=device)
    model = create_nats_model(state["arch_record"]).to(device)
    model.load_state_dict(state["model"])
    return evaluate(model, test_loader, device)
