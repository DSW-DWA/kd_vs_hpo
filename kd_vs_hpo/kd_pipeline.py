import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from kd_vs_hpo.config import TrainConfig
from kd_vs_hpo.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.flops import FlopsBudgetTracker, count_flops_params
from kd_vs_hpo.pipeline import log_epoch, log_hparams
from kd_vs_hpo.utils import accuracy_top1_from_logits, checkpoint_path, extract_logits, set_seed, stage_checkpoint_path
import logging


logger = logging.getLogger(__name__)


FLOPS_NORM = 1e9



def set_eval_and_freeze(models):
    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)




def kd_loss(student_logits, teacher_soft_targets, temperature=4.0):
    """
    KL divergence between student logits and teacher soft targets.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_soft_targets, reduction="batchmean") * (temperature ** 2)



def kld_loss(images, teachers, student_logits, teacher_weights, kd_temperature):
    if len(teachers) > 0:
        with torch.no_grad():
            ensemble_probs = None
            n_teachers = len(teachers)

            if teacher_weights is None:
                teacher_weights_eff = [1.0 / n_teachers] * n_teachers
            else:
                teacher_weights_eff = teacher_weights

            for teacher, w in zip(teachers, teacher_weights_eff):
                teacher_logits = extract_logits(teacher(images))
                teacher_probs = F.softmax(teacher_logits / kd_temperature, dim=-1)
                if ensemble_probs is None:
                    ensemble_probs = teacher_probs.mul(w)
                else:
                    ensemble_probs.add_(teacher_probs, alpha=w)

        student_log_probs = F.log_softmax(student_logits / kd_temperature, dim=-1)
        kd_loss = F.kl_div(
            student_log_probs,
            ensemble_probs,
            reduction="batchmean",
        ) * (kd_temperature ** 2)

    return kd_loss


def mse_prob_loss(images, teachers, student_logits, teacher_weights, **kwargs):
    if len(teachers) > 0:
        with torch.no_grad():
            ensemble_probs = None
            n_teachers = len(teachers)

            if teacher_weights is None:
                teacher_weights_eff = [1.0 / n_teachers] * n_teachers
            else:
                teacher_weights_eff = teacher_weights

            for teacher, w in zip(teachers, teacher_weights_eff):
                teacher_logits = extract_logits(teacher(images))
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                if ensemble_probs is None:
                    ensemble_probs = teacher_probs.mul(w)
                else:
                    ensemble_probs.add_(teacher_probs, alpha=w)

        student_probs = F.softmax(student_logits, dim=-1)
        kd_loss = F.mse_loss(student_probs, ensemble_probs, reduction="mean")
    return kd_loss


def mse_logit_loss(images, teachers, student_logits, teacher_weights, **kwargs):
    if len(teachers) > 0:
        with torch.no_grad():
            ensemble_logits = None
            n_teachers = len(teachers)

            if teacher_weights is None:
                teacher_weights_eff = [1.0 / n_teachers] * n_teachers
            else:
                teacher_weights_eff = teacher_weights

            for teacher, w in zip(teachers, teacher_weights_eff):
                teacher_logits = extract_logits(teacher(images))
                if ensemble_logits is None:
                    ensemble_logits = teacher_logits.mul(w)
                else:
                    ensemble_logits.add_(teacher_logits, alpha=w)

        kd_loss = F.mse_loss(student_logits, ensemble_logits, reduction="mean")
    return kd_loss



def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    *,
    tracker: "FlopsBudgetTracker",
    train_step_flops_per_sample: int,
    grad_clip_norm: float | None,
    scaler: GradScaler,
    device: torch.device,
    teachers: list[nn.Module] | None = None,
    kd_alpha: float = 0.5,
    kd_temperature: float = 2.0,
    teacher_weights: list[float] | None = None,
    kd_loss_fn: callable = kld_loss
) -> tuple[float, float, int, bool]:
    """
    KD training epoch.

    loss = kd_alpha * hard_loss + (1 - kd_alpha) * kd_loss

    teachers:
        list of frozen teacher models. If None or empty, behaves like normal training.
    teacher_weights:
        optional weights for teacher ensemble. If provided, must match len(teachers).
    """
    model.train()

    teachers = teachers or []

    if teacher_weights is not None:
        if len(teacher_weights) != len(teachers):
            raise ValueError("teacher_weights must have the same length as teachers")
        w_sum = float(sum(teacher_weights))
        if w_sum <= 0:
            raise ValueError("teacher_weights must sum to a positive value")
        teacher_weights = [w / w_sum for w in teacher_weights]

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

        with torch.autocast(device_type=device.type, enabled=use_amp):
            student_logits = extract_logits(model(images))
            hard_loss = criterion(student_logits, targets)

            if len(teachers) > 0:
                kd_loss = kd_loss_fn(images, teachers, student_logits, teacher_weights)

                loss = kd_alpha * hard_loss + (1.0 - kd_alpha) * kd_loss
            else:
                loss = hard_loss

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
        total_correct += accuracy_top1_from_logits(student_logits.detach(), targets)
        total_examples += batch_size

    avg_loss = total_loss / max(1, total_examples)
    top1_acc = 100.0 * total_correct / max(1, total_examples)
    return avg_loss, top1_acc, flops_spent, stopped_early




def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    schedule_max_epochs: int,
    momentum: float,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, schedule_max_epochs),
    )
    return optimizer, scheduler


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



def run_kd_training_pipeline(
    *,
    model: nn.Module,
    teachers: list[nn.Module],
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
    kd_loss_fn: callable = kd_loss,
) -> dict[str, Any]:
    """
    Train/evaluate a single model under a cumulative FLOPs budget.
    Stops early before the budget is exceeded.
    """
    set_seed(config.seed, deterministic=config.deterministic)

    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_cifar10_dataloaders(config, device)

    model = model.to(device)
    for teacher in teachers:
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)


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
    teacher_forward_flops_per_sample = 0
    for teacher in teachers:
        flops, _ = count_flops_params(
            teacher,
            input_shape=(1, 3, 32, 32),
            device=str(device),
        )
        teacher_forward_flops_per_sample += flops

    train_step_flops_per_sample = int(forward_flops_per_sample * config.train_step_multiplier + teacher_forward_flops_per_sample)
    eval_flops_per_sample = int(forward_flops_per_sample)

    log_hparams(writer, config, arch_record, params, forward_flops_per_sample)
    

    ckpt_path = checkpoint_path(config.checkpoint_dir, arch_record["arch_index"], int(arch_record.get("trial_id", 0)))


    start_epoch = 0
    best_val_acc1 = float("-inf")
    best_state: dict[str, Any] | None = None

    started_at = time.time()

    with writer:
        for epoch in tqdm(range(start_epoch, max_epochs), desc=f"Training model {arch_record['arch_index']} with teachers", unit="epoch", disable=not verbose):
            epoch_start = time.time()

            train_loss, train_acc1, train_flops, train_stopped = train_one_epoch(
                model=model,
                teachers=teachers,
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                tracker=tracker,
                train_step_flops_per_sample=train_step_flops_per_sample,
                grad_clip_norm=config.grad_clip_norm,
                scaler=scaler,
                device=device,
                kd_temperature=config.kd_temperature,
                kd_loss_fn=kd_loss_fn
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

        logger.info(
            f"[{run_name}] done val_acc1={final_state.get('val_acc1', float('nan')):.2f} "
            f"test_acc1={test_acc1:.2f} "
            f"Gflops={tracker.spent / FLOPS_NORM:,.2f}/{tracker.budget / FLOPS_NORM:,.2f} "
            f"elapsed_min={(time.time() - started_at) / 60:.1f}"
        )

        return {
            "val_acc1": final_state.get("val_acc1", float("nan")),
            "test_acc1": test_acc1,
            "best_checkpoint_path": str(ckpt_path),
            "cumulative_flops": tracker.spent,
            "flops_budget": tracker.budget,
        }
