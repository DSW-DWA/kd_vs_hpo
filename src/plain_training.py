from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.common.config import TrainConfig
from src.common.dataloader import build_cifar10_datasets, build_dataloader
from src.common.nats import create_nats_model
from src.common.optim import create_optimizer_and_scheduler
from src.common.utils import accuracy_top1_from_logits, extract_logits, set_seed


INITIAL_LR = 5.0e-2
INITIAL_WEIGHT_DECAY = 5.0e-4
TRIAL_EPOCHS = (100, 200)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlainTrainingResult:
    runs: pd.DataFrame
    epochs: pd.DataFrame


def plain_train_config_payload(config: TrainConfig) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "validation_fraction": config.validation_fraction,
        "momentum": config.momentum,
        "grad_clip_norm": config.grad_clip_norm,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "amp": config.amp,
        "data_root": config.data_root,
        "checkpoint_dir": config.checkpoint_dir,
        "log_dir": config.log_dir,
    }


def load_architectures_by_index(
    path: Path,
    arch_indices: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    if not arch_indices:
        raise ValueError("At least one architecture index is required")
    if len(set(arch_indices)) != len(arch_indices):
        raise ValueError("Architecture indices must be unique")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    by_index: dict[int, dict[str, Any]] = {}
    for row, record in enumerate(records):
        arch_index = int(record["arch_index"])
        if arch_index in by_index:
            raise ValueError(f"Architecture index {arch_index} is not unique in {path}")
        by_index[arch_index] = {
            **record,
            "arch_row": row,
            "arch_index": arch_index,
        }

    missing = [index for index in arch_indices if index not in by_index]
    if missing:
        raise ValueError(f"Architecture indices were not found in {path}: {missing}")
    return tuple(by_index[index] for index in arch_indices)


def run_plain_experiment(
    *,
    architectures: tuple[dict[str, Any], ...],
    initial_lr: float,
    weight_decay: float,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    verbose: bool,
) -> PlainTrainingResult:
    _validate_inputs(
        architectures=architectures,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tables_dir = output_dir / "tables"
    checkpoints_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "runs"
    tables_dir.mkdir()
    checkpoints_dir.mkdir()
    tensorboard_dir.mkdir()

    train_dataset, val_dataset, test_dataset = build_cifar10_datasets(train_config)
    run_specs = [
        {
            "trial_id": trial_id,
            "target_epochs": target_epochs,
            "trial_seed": train_config.seed + trial_id,
        }
        for trial_id, target_epochs in enumerate(TRIAL_EPOCHS)
    ]
    _write_json(
        output_dir / "run_config.json",
        {
            "experiment": "plain_training",
            "device": str(device),
            "architectures": list(architectures),
            "trials": run_specs,
            "train": plain_train_config_payload(train_config),
            "optimizer": {
                "name": "SGD",
                "initial_lr": initial_lr,
                "weight_decay": weight_decay,
                "momentum": train_config.momentum,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": "target_epochs of each independent trial",
                "eta_min": 0.0,
            },
            "loss": "CrossEntropyLoss",
            "uses_distillation": False,
            "uses_optuna": False,
            "uses_pruning": False,
        },
    )

    run_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    for architecture in architectures:
        for spec in run_specs:
            run_record, new_epoch_records = _run_trial(
                architecture=architecture,
                trial_id=int(spec["trial_id"]),
                trial_seed=int(spec["trial_seed"]),
                target_epochs=int(spec["target_epochs"]),
                initial_lr=initial_lr,
                weight_decay=weight_decay,
                train_config=train_config,
                device=device,
                output_dir=output_dir,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                prior_epoch_records=epoch_records,
                verbose=verbose,
            )
            epoch_records.extend(new_epoch_records)
            run_records.append(run_record)
            _write_table(tables_dir / "runs.csv", run_records)

    return PlainTrainingResult(
        runs=pd.DataFrame.from_records(run_records),
        epochs=pd.DataFrame.from_records(epoch_records),
    )


def _run_trial(
    *,
    architecture: dict[str, Any],
    trial_id: int,
    trial_seed: int,
    target_epochs: int,
    initial_lr: float,
    weight_decay: float,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    test_dataset: Dataset[Any],
    prior_epoch_records: list[dict[str, Any]],
    verbose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arch_index = int(architecture["arch_index"])
    run_name = (
        f"plain_arch_{arch_index}__trial_{trial_id:02d}__epochs_{target_epochs:04d}"
    )
    trial_checkpoint_dir = output_dir / "checkpoints" / run_name
    trial_tensorboard_dir = output_dir / "runs" / run_name
    trial_checkpoint_dir.mkdir()
    trial_tensorboard_dir.mkdir()
    best_checkpoint = trial_checkpoint_dir / "best.pt"
    final_checkpoint = trial_checkpoint_dir / "final.pt"

    set_seed(trial_seed, deterministic=train_config.deterministic)
    train_loader = build_dataloader(
        train_dataset,
        train_config,
        device,
        shuffle=True,
        seed=trial_seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        train_config,
        device,
        shuffle=False,
        seed=trial_seed + 1,
    )
    test_loader = build_dataloader(
        test_dataset,
        train_config,
        device,
        shuffle=False,
        seed=trial_seed + 2,
    )
    model = create_nats_model(architecture).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=initial_lr,
        weight_decay=weight_decay,
        schedule_max_epochs=target_epochs,
        momentum=train_config.momentum,
    )
    scaler = GradScaler(enabled=train_config.amp and device.type == "cuda")

    best_val_acc1 = float("-inf")
    best_epoch = 0
    final_val_acc1 = float("nan")
    new_epoch_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    writer = SummaryWriter(log_dir=str(trial_tensorboard_dir))
    writer.add_text("run/arch_record", str(architecture))
    writer.add_text(
        "run/config",
        str(
            {
                "trial_id": trial_id,
                "trial_seed": trial_seed,
                "target_epochs": target_epochs,
                "batch_size": train_config.batch_size,
                "num_workers": train_config.num_workers,
                "validation_fraction": train_config.validation_fraction,
                "momentum": train_config.momentum,
                "grad_clip_norm": train_config.grad_clip_norm,
                "deterministic": train_config.deterministic,
                "amp": train_config.amp,
                "initial_lr": initial_lr,
                "weight_decay": weight_decay,
            }
        ),
    )

    LOGGER.info(
        "Starting %s: seed=%d lr=%g weight_decay=%g device=%s",
        run_name,
        trial_seed,
        initial_lr,
        weight_decay,
        device,
    )
    with writer:
        iterator = tqdm(
            range(1, target_epochs + 1),
            desc=run_name,
            unit="epoch",
            disable=not verbose,
        )
        for epoch in iterator:
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_acc1 = _train_one_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                grad_clip_norm=train_config.grad_clip_norm,
                scaler=scaler,
                device=device,
            )
            val_loss, final_val_acc1 = _evaluate(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
            )
            scheduler.step()
            improved = final_val_acc1 > best_val_acc1
            if improved:
                best_val_acc1 = final_val_acc1
                best_epoch = epoch
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    architecture=architecture,
                    run_name=run_name,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    target_epochs=target_epochs,
                    epoch=epoch,
                    initial_lr=initial_lr,
                    weight_decay=weight_decay,
                    val_acc1=final_val_acc1,
                )

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc1", train_acc1, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc1", final_val_acc1, epoch)
            writer.add_scalar("optim/lr", current_lr, epoch)
            new_epoch_records.append(
                {
                    "run_name": run_name,
                    "arch_row": int(architecture["arch_row"]),
                    "arch_index": arch_index,
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    "target_epochs": target_epochs,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc1": train_acc1,
                    "val_loss": val_loss,
                    "val_acc1": final_val_acc1,
                    "best_val_acc1": best_val_acc1,
                    "learning_rate": current_lr,
                }
            )
            _write_table(
                output_dir / "tables" / "epoch_metrics.csv",
                [*prior_epoch_records, *new_epoch_records],
            )
            LOGGER.info(
                "%s | epoch=%d/%d | lr=%.6g | train_loss=%.4f | "
                "train_acc1=%.2f | val_loss=%.4f | val_acc1=%.2f | best=%.2f@%d",
                run_name,
                epoch,
                target_epochs,
                current_lr,
                train_loss,
                train_acc1,
                val_loss,
                final_val_acc1,
                best_val_acc1,
                best_epoch,
            )

        test_loss, test_acc1 = _evaluate(
            model=model,
            criterion=criterion,
            loader=test_loader,
            device=device,
        )
        writer.add_scalar("test/loss", test_loss, target_epochs)
        writer.add_scalar("test/acc1", test_acc1, target_epochs)

    elapsed_seconds = time.perf_counter() - started
    _save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        architecture=architecture,
        run_name=run_name,
        trial_id=trial_id,
        trial_seed=trial_seed,
        target_epochs=target_epochs,
        epoch=target_epochs,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        val_acc1=final_val_acc1,
        test_acc1=test_acc1,
    )
    run_record = {
        "run_name": run_name,
        "arch_row": int(architecture["arch_row"]),
        "arch_index": arch_index,
        "trial_id": trial_id,
        "trial_seed": trial_seed,
        "target_epochs": target_epochs,
        "completed_epochs": target_epochs,
        "initial_lr": initial_lr,
        "weight_decay": weight_decay,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": target_epochs,
        "best_epoch": best_epoch,
        "best_val_acc1": best_val_acc1,
        "final_val_acc1": final_val_acc1,
        "tested_checkpoint": "final",
        "test_loss": test_loss,
        "test_acc1": test_acc1,
        "elapsed_seconds": elapsed_seconds,
        "best_checkpoint_path": str(best_checkpoint),
        "final_checkpoint_path": str(final_checkpoint),
        "tensorboard_dir": str(trial_tensorboard_dir),
    }
    LOGGER.info(
        "Finished %s: best_val_acc1=%.2f@%d final_val_acc1=%.2f test_acc1=%.2f",
        run_name,
        best_val_acc1,
        best_epoch,
        final_val_acc1,
        test_acc1,
    )

    del train_loader, val_loader, test_loader
    del scaler, scheduler, optimizer, criterion, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return run_record, new_epoch_records


def _train_one_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader[Any],
    grad_clip_norm: float | None,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
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
        total_correct += accuracy_top1_from_logits(logits.detach(), targets)
        total_examples += batch_size

    return (
        total_loss / max(total_examples, 1),
        100.0 * total_correct / max(total_examples, 1),
    )


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = extract_logits(model(images))
        loss = criterion(logits, targets)
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_top1_from_logits(logits, targets)
        total_examples += batch_size
    return (
        total_loss / max(total_examples, 1),
        100.0 * total_correct / max(total_examples, 1),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    architecture: dict[str, Any],
    run_name: str,
    trial_id: int,
    trial_seed: int,
    target_epochs: int,
    epoch: int,
    initial_lr: float,
    weight_decay: float,
    val_acc1: float,
    test_acc1: float | None = None,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "architecture": architecture,
            "run_name": run_name,
            "trial_id": trial_id,
            "trial_seed": trial_seed,
            "target_epochs": target_epochs,
            "epoch": epoch,
            "initial_lr": initial_lr,
            "weight_decay": weight_decay,
            "val_acc1": val_acc1,
            "test_acc1": test_acc1,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _validate_inputs(
    *,
    architectures: tuple[dict[str, Any], ...],
    initial_lr: float,
    weight_decay: float,
    output_dir: Path,
) -> None:
    if not architectures:
        raise ValueError("At least one architecture is required")
    arch_indices = [int(architecture["arch_index"]) for architecture in architectures]
    if len(set(arch_indices)) != len(arch_indices):
        raise ValueError("Architecture indices must be unique")
    if initial_lr <= 0:
        raise ValueError("Initial learning rate must be positive")
    if weight_decay < 0:
        raise ValueError("Weight decay cannot be negative")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use a new directory for a clean experiment."
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
    temporary_path.replace(path)


def _write_table(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(".tmp")
    pd.DataFrame.from_records(records).to_csv(temporary_path, index=False)
    temporary_path.replace(path)
