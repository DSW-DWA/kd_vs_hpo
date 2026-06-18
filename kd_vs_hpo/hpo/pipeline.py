import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.common.optim import create_optimizer_and_scheduler
from kd_vs_hpo.common.utils import (
    accuracy_top1_from_logits,
    extract_logits,
    set_seed,
)
from kd_vs_hpo.hpo.asha import ASHAPlan, TrialConfig, make_plan, sample_trials
from kd_vs_hpo.hpo.config import HPOExperimentConfig
from kd_vs_hpo.hpo.plotting import save_hpo_plots

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HPOExperimentResult:
    stages: pd.DataFrame
    summary: pd.DataFrame
    stages_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]


def _load_architectures(path: Path, rows: tuple[int, ...] | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    selected_rows = set(rows) if rows is not None else None
    selected: list[dict[str, Any]] = []
    for row, source in enumerate(records):
        if selected_rows is not None and row not in selected_rows:
            continue
        record = dict(source)
        record["arch_row"] = row
        record["arch_index"] = int(record["arch_index"])
        selected.append(record)
    if not selected:
        raise ValueError("No architectures selected")
    return selected


def _load_costs(path: Path) -> dict[int, dict[str, Any]]:
    costs = pd.read_csv(path)
    required = {"arch_row", "arch_index", "forward_flops_per_sample"}
    missing = required - set(costs.columns)
    if missing:
        raise ValueError(f"Costs CSV is missing columns: {sorted(missing)}")
    return {
        int(row): values
        for row, values in costs.set_index("arch_row").to_dict(orient="index").items()
    }


def _checkpoint_path(
    checkpoint_dir: Path,
    arch_index: int,
    trial_id: int,
    target_epochs: int | None = None,
) -> Path:
    suffix = "" if target_epochs is None else f"_epoch_{target_epochs:04d}"
    return checkpoint_dir / f"arch_{arch_index:05d}_trial_{trial_id:02d}{suffix}.pt"


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    scaler: GradScaler,
    grad_clip_norm: float | None,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    use_amp = scaler.is_enabled() and device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = extract_logits(model(images))
            loss = criterion(logits, targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.inference_mode()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = extract_logits(model(images))
        total_correct += accuracy_top1_from_logits(logits, targets)
        total_examples += int(targets.size(0))
    return 100.0 * total_correct / max(1, total_examples)


def _run_training_stage(
    *,
    trial: TrialConfig,
    target_epochs: int,
    plan: ASHAPlan,
    seed: int,
    arch_record: dict[str, Any],
    experiment: HPOExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed, deterministic=experiment.train.deterministic)
    model = create_nats_model(arch_record).to(device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=trial.lr,
        weight_decay=trial.weight_decay,
        schedule_max_epochs=plan.max_epochs,
        momentum=experiment.train.momentum,
    )
    scaler = GradScaler(enabled=experiment.train.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = experiment.output_dir / "checkpoints"
    checkpoint = _checkpoint_path(
        checkpoint_dir,
        arch_record["arch_index"],
        trial.trial_id,
    )
    stage_checkpoint = _checkpoint_path(
        checkpoint_dir,
        arch_record["arch_index"],
        trial.trial_id,
        target_epochs,
    )

    if stage_checkpoint.exists():
        state = torch.load(stage_checkpoint, map_location=device)
        return {
            "val_acc1": float(state["val_acc1"]),
            "checkpoint_path": str(stage_checkpoint),
        }

    start_epoch = 0
    last_val_acc1 = float("nan")
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        checkpoint_epoch = int(state.get("epoch", 0))
        if (
            state.get("schedule_max_epochs") == plan.max_epochs
            and checkpoint_epoch < target_epochs
        ):
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = checkpoint_epoch
            last_val_acc1 = float(state.get("val_acc1", float("nan")))

    if start_epoch >= target_epochs and not math.isnan(last_val_acc1):
        existing = stage_checkpoint if stage_checkpoint.exists() else checkpoint
        return {"val_acc1": last_val_acc1, "checkpoint_path": str(existing)}

    started_at = time.time()
    for epoch in range(start_epoch, target_epochs):
        train_loss = _train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            scaler,
            experiment.train.grad_clip_norm,
            device,
        )
        scheduler.step()
        logger.info(
            "arch=%s trial=%s epoch=%s/%s loss=%.4f lr=%.3e",
            arch_record["arch_index"],
            trial.trial_id,
            epoch + 1,
            target_epochs,
            train_loss,
            optimizer.param_groups[0]["lr"],
        )

    val_acc1 = _evaluate(model, val_loader, device)
    state = {
        "epoch": target_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "schedule_max_epochs": plan.max_epochs,
        "val_acc1": val_acc1,
        "trial": asdict(trial),
        "arch_record": arch_record,
    }
    torch.save(state, checkpoint)
    torch.save(state, stage_checkpoint)
    logger.info(
        "arch=%s trial=%s target_epochs=%s val_acc1=%.2f elapsed_min=%.1f",
        arch_record["arch_index"],
        trial.trial_id,
        target_epochs,
        val_acc1,
        (time.time() - started_at) / 60,
    )
    return {"val_acc1": val_acc1, "checkpoint_path": str(stage_checkpoint)}


def _run_architecture(
    *,
    arch_record: dict[str, Any],
    cost: dict[str, Any],
    experiment: HPOExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train: int,
    n_val: int,
    device: torch.device,
) -> pd.DataFrame:
    forward_flops = int(float(cost["forward_flops_per_sample"]))
    epoch_flops = int(experiment.train.train_step_multiplier * forward_flops * n_train)
    validation_flops = forward_flops * n_val
    plan = make_plan(experiment.asha, epoch_flops, validation_flops)
    logger.info(
        "Starting architecture row=%s index=%s: initial_configs=%s rungs=%s "
        "planned_flops=%s budget_flops=%s",
        arch_record["arch_row"],
        arch_record["arch_index"],
        plan.num_initial_configs,
        plan.rungs,
        plan.planned_flops,
        experiment.asha.budget_flops_per_arch,
    )
    seed = experiment.train.seed + arch_record["arch_row"]
    trials = sample_trials(plan.num_initial_configs, experiment.search_space, seed)
    completed_epochs = {trial.trial_id: 0 for trial in trials}
    alive = list(trials)
    spent_flops = 0
    spent_train_flops = 0
    spent_validation_flops = 0
    records: list[dict[str, Any]] = []

    for rung, target_epochs in enumerate(plan.rungs):
        logger.info(
            "arch=%s rung=%s target_epochs=%s active_trials=%s budget_used=%.2f%%",
            arch_record["arch_index"],
            rung,
            target_epochs,
            len(alive),
            100 * spent_flops / experiment.asha.budget_flops_per_arch,
        )
        rung_records: list[dict[str, Any]] = []
        for trial in alive:
            incremental_epochs = target_epochs - completed_epochs[trial.trial_id]
            train_flops = incremental_epochs * epoch_flops
            stage_flops = train_flops + validation_flops
            if spent_flops + stage_flops > experiment.asha.budget_flops_per_arch:
                continue
            stage = _run_training_stage(
                trial=trial,
                target_epochs=target_epochs,
                plan=plan,
                seed=seed + trial.trial_id,
                arch_record=arch_record,
                experiment=experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
            spent_flops += stage_flops
            spent_train_flops += train_flops
            spent_validation_flops += validation_flops
            completed_epochs[trial.trial_id] = target_epochs
            record = {
                "arch_row": arch_record["arch_row"],
                "arch_index": arch_record["arch_index"],
                "arch_str": arch_record["arch_str"],
                "dataset": arch_record.get("dataset"),
                **asdict(trial),
                "rung": rung,
                "target_epochs": target_epochs,
                "incremental_epochs": incremental_epochs,
                "train_flops": train_flops,
                "validation_flops": validation_flops,
                "stage_flops": stage_flops,
                "cumulative_flops": spent_flops,
                "cumulative_train_flops": spent_train_flops,
                "cumulative_validation_flops": spent_validation_flops,
                "val_acc1": float(stage["val_acc1"]),
                "checkpoint_path": stage["checkpoint_path"],
                "status": "completed",
                "asha_min_epochs": plan.min_epochs,
                "asha_max_epochs": plan.max_epochs,
                "asha_reduction_factor": plan.reduction_factor,
                "asha_num_initial_configs": plan.num_initial_configs,
                "asha_planned_flops": plan.planned_flops,
                "forward_flops_per_sample": forward_flops,
                "epoch_flops": epoch_flops,
                "budget_flops": experiment.asha.budget_flops_per_arch,
                "params": cost.get("params", np.nan),
                "latency": cost.get("latency", np.nan),
            }
            records.append(record)
            rung_records.append(record)
        if not rung_records:
            break
        ranked = sorted(rung_records, key=lambda row: row["val_acc1"], reverse=True)
        promoted_count = max(1, math.ceil(len(ranked) / plan.reduction_factor))
        promoted_ids = {row["trial_id"] for row in ranked[:promoted_count]}
        logger.info(
            "arch=%s rung=%s completed=%s promoted_trials=%s",
            arch_record["arch_index"],
            rung,
            len(rung_records),
            sorted(promoted_ids),
        )
        alive = [trial for trial in alive if trial.trial_id in promoted_ids]

    result = pd.DataFrame(records)
    if not result.empty:
        result["best_val_acc1_so_far"] = result["val_acc1"].cummax()
    return result


def _build_summary(
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

    test_acc1 = []
    for row in summary.itertuples():
        state = torch.load(row.checkpoint_path, map_location=device)
        model = create_nats_model(state["arch_record"]).to(device)
        model.load_state_dict(state["model"])
        test_acc1.append(_evaluate(model, test_loader, device))
    summary["test_acc1"] = test_acc1
    summary["test_flops"] = summary["forward_flops_per_sample"] * n_test
    return summary


def run_hpo_experiment(
    experiment: HPOExperimentConfig,
    device: torch.device,
) -> HPOExperimentResult:
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    (experiment.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    architectures = _load_architectures(experiment.architectures_path, experiment.arch_rows)
    costs = _load_costs(experiment.costs_path)
    logger.info(
        "Loaded %s architectures from %s",
        len(architectures),
        experiment.architectures_path,
    )
    logger.info("Preparing CIFAR-10 dataloaders")
    train_loader, val_loader, test_loader, n_train, n_val, n_test = (
        build_cifar10_dataloaders(experiment.train, device)
    )
    logger.info(
        "Dataset sizes: train=%s validation=%s test=%s",
        n_train,
        n_val,
        n_test,
    )

    results = []
    for arch_record in architectures:
        arch_row = arch_record["arch_row"]
        if arch_row not in costs:
            raise KeyError(f"No FLOPs record for architecture row {arch_row}")
        results.append(
            _run_architecture(
                arch_record=arch_record,
                cost=costs[arch_row],
                experiment=experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                n_train=n_train,
                n_val=n_val,
                device=device,
            )
        )

    stages = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    summary = _build_summary(stages, test_loader, n_test, device)
    stages_path = experiment.output_dir / "hpo_results.csv"
    summary_path = experiment.output_dir / "hpo_summary.csv"
    stages.to_csv(stages_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_paths = (
        save_hpo_plots(stages, summary, experiment.output_dir)
        if experiment.generate_plots
        else ()
    )
    return HPOExperimentResult(
        stages=stages,
        summary=summary,
        stages_path=stages_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
    )
