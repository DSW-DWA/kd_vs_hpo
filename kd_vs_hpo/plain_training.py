"""Fixed-hyperparameter training compatible with the current shared project API."""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.flops import count_flops_params
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.common.optim import create_optimizer_and_scheduler
from kd_vs_hpo.common.utils import accuracy_top1_from_logits, extract_logits, set_seed


INITIAL_LR = 5.0e-2
INITIAL_WEIGHT_DECAY = 5.0e-4
TRIAL_EPOCHS: tuple[int, ...] = (100, 200)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlainExperimentResult:
    runs: pd.DataFrame
    epochs: pd.DataFrame
    output_dir: Path


def load_architectures_by_rows(
    path: Path,
    rows: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        source_records = json.load(file)
    if not isinstance(source_records, list):
        raise ValueError("Architecture file must contain a JSON list")

    requested_rows = None if rows is None else set(rows)
    if requested_rows is not None and any(row < 0 for row in requested_rows):
        raise ValueError("Architecture rows cannot be negative")

    architectures: list[dict[str, Any]] = []
    for arch_row, source in enumerate(source_records):
        if requested_rows is not None and arch_row not in requested_rows:
            continue
        if not isinstance(source, dict):
            raise ValueError(f"Architecture row {arch_row} must be a JSON object")
        architecture = dict(source)
        if "arch_index" not in architecture or "arch_str" not in architecture:
            raise ValueError(
                f"Architecture row {arch_row} is missing arch_index or arch_str"
            )
        architecture.update(
            arch_row=arch_row,
            arch_index=int(architecture["arch_index"]),
        )
        architectures.append(architecture)

    if not architectures:
        raise ValueError("No architectures selected")
    if requested_rows is not None:
        missing = sorted(requested_rows - {item["arch_row"] for item in architectures})
        if missing:
            raise ValueError(f"Architecture rows are out of range: {missing}")
    return architectures


def plain_train_config_payload(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


def run_plain_experiment(
    *,
    architectures: list[dict[str, Any]],
    initial_lr: float,
    weight_decay: float,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    trial_epochs: tuple[int, ...] = TRIAL_EPOCHS,
    verbose: bool = False,
) -> PlainExperimentResult:
    if not architectures:
        raise ValueError("At least one architecture is required")
    if initial_lr <= 0:
        raise ValueError("initial_lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not trial_epochs or any(epoch < 1 for epoch in trial_epochs):
        raise ValueError("trial_epochs must contain positive epoch counts")
    if len(set(trial_epochs)) != len(trial_epochs):
        raise ValueError("trial_epochs must be unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_config.json",
        {
            "architectures": architectures,
            "trial_epochs": trial_epochs,
            "initial_lr": initial_lr,
            "weight_decay": weight_decay,
            "device": str(device),
            "train": plain_train_config_payload(train_config),
            "independent_runs": True,
            "uses_distillation": False,
            "uses_optuna": False,
            "uses_pruning": False,
        },
    )

    run_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    for architecture in architectures:
        for trial_id, target_epochs in enumerate(trial_epochs):
            run, epochs = _run_plain_trial(
                architecture=architecture,
                trial_id=trial_id,
                target_epochs=target_epochs,
                initial_lr=initial_lr,
                weight_decay=weight_decay,
                train_config=train_config,
                device=device,
                verbose=verbose,
            )
            run_records.append(run)
            epoch_records.extend(epochs)
            _save_tables(output_dir, run_records, epoch_records)

    runs = pd.DataFrame(run_records).sort_values(
        ["arch_row", "target_epochs"], ignore_index=True
    )
    epochs = pd.DataFrame(epoch_records).sort_values(
        ["arch_row", "target_epochs", "epoch"], ignore_index=True
    )
    return PlainExperimentResult(runs=runs, epochs=epochs, output_dir=output_dir)


def _run_plain_trial(
    *,
    architecture: dict[str, Any],
    trial_id: int,
    target_epochs: int,
    initial_lr: float,
    weight_decay: float,
    train_config: TrainConfig,
    device: torch.device,
    verbose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial_seed = train_config.seed + trial_id
    set_seed(trial_seed, deterministic=train_config.deterministic)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = (
        build_cifar10_dataloaders(
            train_config.checkpoint_dir,
            train_config.log_dir,
            train_config.data_root,
            trial_seed,
            train_config.batch_size,
            train_config.num_workers,
            train_config.validation_fraction,
            device,
        )
    )

    model = create_nats_model(architecture)
    forward_flops_per_sample, params = count_flops_params(model)
    model = model.to(device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=initial_lr,
        weight_decay=weight_decay,
        schedule_max_epochs=target_epochs,
        momentum=train_config.momentum,
    )
    scaler = GradScaler(enabled=train_config.amp and device.type == "cuda")
    checkpoint_path = (
        train_config.checkpoint_dir
        / f"arch_{architecture['arch_index']}_trial_{trial_id:02d}_"
        f"epoch_{target_epochs:04d}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    best_val_acc1 = float("-inf")
    best_epoch = 0
    epoch_records: list[dict[str, Any]] = []
    train_flops_per_epoch = int(
        train_config.train_step_multiplier * forward_flops_per_sample * n_train
    )
    validation_flops_per_epoch = int(forward_flops_per_sample * n_val)

    try:
        for epoch in range(1, target_epochs + 1):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = _train_one_epoch(
                model,
                optimizer,
                train_loader,
                scaler,
                train_config.grad_clip_norm,
                device,
            )
            val_acc1 = _evaluate(model, val_loader, device)
            scheduler.step()
            if val_acc1 > best_val_acc1:
                best_val_acc1 = val_acc1
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch_record": architecture,
                        "trial_id": trial_id,
                        "trial_seed": trial_seed,
                        "target_epochs": target_epochs,
                        "best_epoch": best_epoch,
                        "best_val_acc1": best_val_acc1,
                        "initial_lr": initial_lr,
                        "weight_decay": weight_decay,
                    },
                    checkpoint_path,
                )
            epoch_records.append(
                {
                    "arch_row": architecture["arch_row"],
                    "arch_index": architecture["arch_index"],
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    "target_epochs": target_epochs,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_acc1": val_acc1,
                    "best_val_acc1": best_val_acc1,
                    "learning_rate": current_lr,
                    "cumulative_flops": epoch
                    * (train_flops_per_epoch + validation_flops_per_epoch),
                }
            )
            if verbose:
                logger.info(
                    "arch=%s trial=%s epoch=%s/%s loss=%.4f val_acc1=%.2f",
                    architecture["arch_index"],
                    trial_id,
                    epoch,
                    target_epochs,
                    train_loss,
                    val_acc1,
                )

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_acc1 = _evaluate(model, test_loader, device)
        total_train_flops = train_flops_per_epoch * target_epochs
        total_validation_flops = validation_flops_per_epoch * target_epochs
        test_flops = int(forward_flops_per_sample * n_test)
        run_record = {
            "arch_row": architecture["arch_row"],
            "arch_index": architecture["arch_index"],
            "arch_str": architecture["arch_str"],
            "trial_id": trial_id,
            "trial_seed": trial_seed,
            "target_epochs": target_epochs,
            "completed_epochs": target_epochs,
            "initial_lr": initial_lr,
            "weight_decay": weight_decay,
            "best_epoch": best_epoch,
            "best_val_acc1": best_val_acc1,
            "test_acc1": test_acc1,
            "params": params,
            "forward_flops_per_sample": forward_flops_per_sample,
            "train_flops": total_train_flops,
            "validation_flops": total_validation_flops,
            "test_flops": test_flops,
            "total_flops": total_train_flops + total_validation_flops + test_flops,
            "checkpoint_path": str(checkpoint_path),
            "run_seconds": time.perf_counter() - started,
        }
        return run_record, epoch_records
    finally:
        del train_loader, val_loader, test_loader, scaler, scheduler, optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


def _save_tables(
    output_dir: Path,
    run_records: list[dict[str, Any]],
    epoch_records: list[dict[str, Any]],
) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(tables_dir / "runs.csv", pd.DataFrame(run_records))
    _write_csv(tables_dir / "epoch_metrics.csv", pd.DataFrame(epoch_records))


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temporary_path = path.with_suffix(".tmp")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    scaler: GradScaler,
    grad_clip_norm: float | None,
    device: torch.device,
) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            enabled=scaler.is_enabled() and device.type == "cuda",
        ):
            loss = criterion(extract_logits(model(images)), targets)
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
    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        correct += accuracy_top1_from_logits(extract_logits(model(images)), targets)
        total += int(targets.size(0))
    return 100.0 * correct / max(total, 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
    temporary_path.replace(path)
