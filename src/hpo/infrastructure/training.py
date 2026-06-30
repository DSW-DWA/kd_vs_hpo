import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.common.nats import create_nats_model
from src.common.optim import create_optimizer_and_scheduler
from src.common.utils import accuracy_top1_from_logits, extract_logits, set_seed
from src.hpo.domain.asha import ASHAPlan, TrialConfig
from src.hpo.domain.config import HPOExperimentConfig

logger = logging.getLogger(__name__)


def checkpoint_path(
    checkpoint_dir: Path,
    arch_index: int,
    trial_id: int,
    target_epochs: int | None = None,
) -> Path:
    suffix = "" if target_epochs is None else f"_epoch_{target_epochs:04d}"
    return checkpoint_dir / f"arch_{arch_index:05d}_trial_{trial_id:02d}{suffix}.pt"


def train_one_epoch(
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
    return total_loss / max(1, total_examples)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        total_correct += accuracy_top1_from_logits(extract_logits(model(images)), targets)
        total_examples += int(targets.size(0))
    return 100.0 * total_correct / max(1, total_examples)


def run_training_stage(
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
    checkpoint = checkpoint_path(checkpoint_dir, arch_record["arch_index"], trial.trial_id)
    stage_checkpoint = checkpoint_path(
        checkpoint_dir, arch_record["arch_index"], trial.trial_id, target_epochs
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
        if state.get("schedule_max_epochs") == plan.max_epochs and checkpoint_epoch < target_epochs:
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
        train_loss = train_one_epoch(
            model, optimizer, criterion, train_loader, scaler,
            experiment.train.grad_clip_norm, device,
        )
        scheduler.step()
        logger.info(
            "device=%s arch=%s trial=%s epoch=%s/%s loss=%.4f lr=%.3e",
            device, arch_record["arch_index"], trial.trial_id, epoch + 1,
            target_epochs, train_loss, optimizer.param_groups[0]["lr"],
        )

    val_acc1 = evaluate(model, val_loader, device)
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
        "device=%s arch=%s trial=%s target_epochs=%s val_acc1=%.2f elapsed_min=%.1f",
        device, arch_record["arch_index"], trial.trial_id, target_epochs,
        val_acc1, (time.time() - started_at) / 60,
    )
    return {"val_acc1": val_acc1, "checkpoint_path": str(stage_checkpoint)}
