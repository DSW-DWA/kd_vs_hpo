import logging
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.common.config import TrainConfig
from src.common.dataloader import build_cifar10_dataloaders
from src.common.flops import FlopsBudgetTracker, count_flops_params
from src.common.optim import create_optimizer_and_scheduler
from src.common.utils import (
    accuracy_top1_from_logits,
    checkpoint_path,
    extract_logits,
    set_seed,
    stage_checkpoint_path,
)


logger = logging.getLogger(__name__)


FLOPS_NORM = 1e9


def log_hparams(
    writer: SummaryWriter,
    config: TrainConfig,
    arch_record: dict[str, Any],
    params: int,
    flops_per_sample: int,
) -> None:
    writer.add_text("run/arch_str", str(arch_record.get("arch_str", "")))
    writer.add_text("run/arch_record", str(arch_record))
    writer.add_text("run/config", str(asdict(config)))
    writer.add_scalar("model/params", params, 0)
    writer.add_scalar("model/Gflops_per_sample", flops_per_sample / FLOPS_NORM, 0)

def log_epoch(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    train_acc1: float,
    val_loss: float,
    val_acc1: float,
    lr: float,
    cumulative_flops: int,
    flops_budget: int,
) -> None:
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/acc1", train_acc1, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/acc1", val_acc1, epoch)
    writer.add_scalar("optim/lr", lr, epoch)
    writer.add_scalar("budget/cumulative_Gflops", cumulative_flops / FLOPS_NORM, epoch)
    writer.add_scalar("budget/remaining_Gflops", max(0, flops_budget - cumulative_flops) / FLOPS_NORM, epoch)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    *,
    tracker: FlopsBudgetTracker,
    train_step_flops_per_sample: int,
    grad_clip_norm: float | None,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float, int, bool]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    flops_spent = 0
    stopped_early = False

    use_amp = scaler.is_enabled() and device.type == "cuda"

    for images, targets in loader:
        batch_size = int(targets.size(0))
        batch_flops = int(train_step_flops_per_sample * batch_size)

        if not tracker.can_spend(batch_flops):
            stopped_early = True
            break

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

        tracker.spend(batch_flops)
        flops_spent += batch_flops

        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_top1_from_logits(logits.detach(), targets)
        total_examples += batch_size

    avg_loss = total_loss / max(1, total_examples)
    top1_acc = 100.0 * total_correct / max(1, total_examples)
    return avg_loss, top1_acc, flops_spent, stopped_early


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    *,
    tracker: FlopsBudgetTracker,
    eval_flops_per_sample: int,
    device: torch.device,
) -> tuple[float, float, int, bool]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    flops_spent = 0
    stopped_early = False

    for images, targets in loader:
        batch_size = int(targets.size(0))
        batch_flops = int(eval_flops_per_sample * batch_size)

        if not tracker.can_spend(batch_flops):
            stopped_early = True
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = extract_logits(model(images))
        loss = criterion(logits, targets)

        tracker.spend(batch_flops)
        flops_spent += batch_flops

        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_top1_from_logits(logits, targets)
        total_examples += batch_size

    avg_loss = total_loss / max(1, total_examples)
    top1_acc = 100.0 * total_correct / max(1, total_examples)
    return avg_loss, top1_acc, flops_spent, stopped_early



def run_training_pipeline(
    *,
    model: nn.Module,
    config: TrainConfig,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    flops_budget: int,
    run_name: str,
    arch_record: dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train/evaluate a single model under a cumulative FLOPs budget.
    Stops early before the budget is exceeded.
    """
    set_seed(config.seed, deterministic=config.deterministic)

    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_cifar10_dataloaders(config, device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        schedule_max_epochs=max_epochs,
        momentum=config.momentum,
    )

    scaler = GradScaler(enabled=(config.amp and device.type == "cuda"))
    tracker = FlopsBudgetTracker(budget=flops_budget)


    forward_flops_per_sample, params = count_flops_params(
        model,
        input_shape=(1, 3, 32, 32),
        device=str(device),
    )
    train_step_flops_per_sample = int(forward_flops_per_sample * config.train_step_multiplier)
    eval_flops_per_sample = int(forward_flops_per_sample)

    log_hparams(writer, config, arch_record, params, forward_flops_per_sample)
    

    ckpt_path = checkpoint_path(config.checkpoint_dir, arch_record["arch_index"], int(arch_record.get("trial_id", 0)))
    stage_ckpt_path = stage_checkpoint_path(
        config.checkpoint_dir,
        arch_record["arch_index"],
        int(arch_record.get("trial_id", 0)),
        max_epochs,
    )

    start_epoch = 0
    best_val_acc1 = float("-inf")
    best_state: dict[str, Any] | None = None

    started_at = time.time()

    with writer:
        for epoch in tqdm(range(start_epoch, max_epochs), desc=f"Training model {arch_record['arch_index']}", unit="epoch", disable=not verbose):
            epoch_start = time.time()

            train_loss, train_acc1, train_flops, train_stopped = train_one_epoch(
                model,
                optimizer,
                criterion,
                train_loader,
                tracker=tracker,
                train_step_flops_per_sample=train_step_flops_per_sample,
                grad_clip_norm=config.grad_clip_norm,
                scaler=scaler,
                device=device,
            )

            if train_stopped:
                scheduler.step()
                break

            estimated_val_flops = eval_flops_per_sample * n_val
            if not tracker.can_spend(estimated_val_flops):
                scheduler.step()
                break

            val_loss, val_acc1, val_flops, val_stopped = evaluate(
                model,
                criterion,
                val_loader,
                tracker=tracker,
                eval_flops_per_sample=eval_flops_per_sample,
                device=device,
            )

            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]

    
            log_epoch(
                writer=writer,
                epoch=epoch,
                train_loss=train_loss,
                train_acc1=train_acc1,
                val_loss=val_loss,
                val_acc1=val_acc1,
                lr=lr_now,
                cumulative_flops=tracker.spent,
                flops_budget=flops_budget,
            )

            logger.info(
                f"[{run_name}] epoch={epoch + 1}/{max_epochs} "
                f"train_loss={train_loss:.4f} train_acc1={train_acc1:.2f} "
                f"val_loss={val_loss:.4f} val_acc1={val_acc1:.2f} "
                f"Gflops={tracker.spent / FLOPS_NORM:,.2f}/{tracker.budget / FLOPS_NORM:,.2f} "
                f"lr={lr_now:.3e} elapsed_min={(time.time() - epoch_start) / 60:.1f}"
            )

            if val_acc1 > best_val_acc1:
                best_val_acc1 = val_acc1
                best_state = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_loss": train_loss,
                    "train_acc1": train_acc1,
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "cumulative_flops": tracker.spent,
                    "flops_budget": tracker.budget,
                    "config": asdict(config),
                    "arch_record": arch_record,
                    "counts": {
                        "n_train": n_train,
                        "n_val": n_val,
                        "n_test": n_test,
                        "params": params,
                        "forward_flops_per_sample": forward_flops_per_sample,
                        "train_step_flops_per_sample": train_step_flops_per_sample,
                        "eval_flops_per_sample": eval_flops_per_sample,
                    },
                }

            if tracker.spent >= tracker.budget:
                break

        final_state = best_state or {
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "val_acc1": float("nan"),
            "cumulative_flops": tracker.spent,
            "flops_budget": tracker.budget,
            "config": asdict(config),
            "arch_record": arch_record,
        }

        torch.save(final_state, ckpt_path)
        torch.save(final_state, stage_ckpt_path)

        test_loss, test_acc1, test_flops, test_stopped = (float("nan"), float("nan"), 0, False)
        if tracker.can_spend(eval_flops_per_sample * len(test_dataset := test_loader.dataset)):
            test_loss, test_acc1, test_flops, test_stopped = evaluate(
                model,
                criterion,
                test_loader,
                tracker=tracker,
                eval_flops_per_sample=eval_flops_per_sample,
                device=device,
            )

        final_state["test_loss"] = test_loss
        final_state["test_acc1"] = test_acc1
        final_state["test_flops"] = test_flops
        final_state["elapsed_sec"] = time.time() - started_at

        torch.save(final_state, ckpt_path)
        torch.save(final_state, stage_ckpt_path)

        logger.info(
            f"[{run_name}] done val_acc1={final_state.get('val_acc1', float('nan')):.2f} "
            f"test_acc1={test_acc1:.2f} "
            f"Gflops={tracker.spent / FLOPS_NORM:,.2f}/{tracker.budget / FLOPS_NORM:,.2f} "
            f"elapsed_min={(time.time() - started_at) / 60:.1f}"
        )

        return {
            "val_acc1": final_state.get("val_acc1", float("nan")),
            "test_acc1": test_acc1,
            "checkpoint_path": str(stage_ckpt_path),
            "best_checkpoint_path": str(ckpt_path),
            "cumulative_flops": tracker.spent,
            "flops_budget": tracker.budget,
        }
