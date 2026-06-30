from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.nats import create_nats_model
from src.hpo.infrastructure.training import evaluate


@dataclass(frozen=True)
class HPOExperimentResult:
    stages: pd.DataFrame
    summary: pd.DataFrame
    stages_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]


def build_summary(
    stages: pd.DataFrame,
    test_loader: DataLoader,
    n_test: int,
    device: torch.device,
) -> pd.DataFrame:
    if stages.empty:
        return pd.DataFrame()
    summary = (
        stages.sort_values(["arch_row", "val_acc1"])
        .groupby("arch_row", as_index=False)
        .tail(1)
        .sort_values("val_acc1", ascending=False)
        .copy()
    )
    spent = stages.groupby("arch_row").agg(
        spent_flops=("stage_flops", "sum"),
        spent_train_flops=("train_flops", "sum"),
        spent_validation_flops=("validation_flops", "sum"),
        completed_stages=("status", "size"),
    )
    summary = summary.join(spent, on="arch_row")
    summary["spent_budget_ratio"] = summary["spent_flops"] / summary["budget_flops"]
    summary["test_acc1"] = [
        _evaluate_checkpoint(row.checkpoint_path, test_loader, device)
        for row in summary.itertuples()
    ]
    summary["test_flops"] = summary["forward_flops_per_sample"] * n_test
    return summary


def _evaluate_checkpoint(
    checkpoint_path: str, test_loader: DataLoader, device: torch.device
) -> float:
    state = torch.load(checkpoint_path, map_location=device)
    model = create_nats_model(state["arch_record"]).to(device)
    model.load_state_dict(state["model"])
    return evaluate(model, test_loader, device)
