"""Lightning training and validation helpers used by HPO studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.common.flops import FlopsBudgetTracker
from kd_vs_hpo.common.train_pipeline import (
    DEFAULT_OPTIMIZER_CLS,
    DEFAULT_SCHEDULER_CLS,
)
from kd_vs_hpo.common.train_modules import KDLightningModule, build_trainer


@dataclass(frozen=True)
class LightningTrialResult:
    completed_epochs: int
    best_epoch: int
    best_val_acc1: float
    train_flops: int
    validation_flops: int
    pruned: bool


class OptunaMetricsCallback(L.Callback):
    def __init__(
        self,
        *,
        trial: optuna.Trial,
        study_name: str,
        checkpoint_path: Path,
        epoch_records: list[dict[str, Any]],
    ) -> None:
        self.trial = trial
        self.study_name = study_name
        self.checkpoint_path = checkpoint_path
        self.epoch_records = epoch_records
        self.completed_epochs = 0
        self.best_epoch = 0
        self.best_val_acc1 = float("-inf")
        self.train_flops = 0
        self.validation_flops = 0
        self.pruned = False

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: KDLightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "val_acc" not in metrics:
            return

        epoch = trainer.current_epoch + 1
        val_acc1 = 100.0 * _metric_value(metrics["val_acc"])
        train_loss = _metric_value(metrics.get("train_loss", float("nan")))
        current_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        self.completed_epochs = epoch
        self.train_flops += int(pl_module.train_epoch_flops)
        self.validation_flops += int(pl_module.eval_epoch_flops)

        if val_acc1 > self.best_val_acc1:
            self.best_val_acc1 = val_acc1
            self.best_epoch = epoch
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": pl_module.model.state_dict(),
                    "arch_record": getattr(pl_module, "arch_record"),
                },
                self.checkpoint_path,
            )

        self.trial.report(self.best_val_acc1, step=epoch)
        self.epoch_records.append(
            {
                "study_name": self.study_name,
                "trial_id": self.trial.number,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc1": val_acc1,
                "best_val_acc1": self.best_val_acc1,
                "learning_rate": current_lr,
                "cumulative_trial_flops": (
                    pl_module.flops_tracker.spent
                    if pl_module.flops_tracker is not None
                    else self.train_flops + self.validation_flops
                ),
            }
        )
        if self.trial.should_prune():
            self.pruned = True
            trainer.should_stop = True

    def result(self) -> LightningTrialResult:
        return LightningTrialResult(
            completed_epochs=self.completed_epochs,
            best_epoch=self.best_epoch,
            best_val_acc1=self.best_val_acc1,
            train_flops=self.train_flops,
            validation_flops=self.validation_flops,
            pruned=self.pruned,
        )


def build_lightning_module(
    *,
    model: nn.Module,
    architecture: dict[str, Any],
    train_config: TrainConfig,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    forward_flops_per_sample: int,
    flops_tracker: FlopsBudgetTracker | None,
    optimizer_cls: type[torch.optim.Optimizer] = DEFAULT_OPTIMIZER_CLS,
    scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = (
        DEFAULT_SCHEDULER_CLS
    ),
) -> KDLightningModule:
    module = KDLightningModule(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer_cls=optimizer_cls,
        optimizer_kwargs={
            "lr": lr,
            "momentum": train_config.momentum,
            "weight_decay": weight_decay,
        },
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=(
            None if scheduler_cls is None else {"T_max": max(1, max_epochs)}
        ),
        train_step_flops=int(
            forward_flops_per_sample * train_config.train_step_multiplier
        ),
        eval_step_flops=int(forward_flops_per_sample),
        num_classes=int(architecture.get("num_classes", 10)),
        flops_tracker=flops_tracker,
    )
    module.arch_record = architecture
    return module


def fit_lightning_trial(
    *,
    lightning_module: KDLightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trial: optuna.Trial,
    study_name: str,
    checkpoint_path: Path,
    epoch_records: list[dict[str, Any]],
    train_config: TrainConfig,
    max_epochs: int,
) -> LightningTrialResult:
    callback = OptunaMetricsCallback(
        trial=trial,
        study_name=study_name,
        checkpoint_path=checkpoint_path,
        epoch_records=epoch_records,
    )
    trainer = build_trainer(
        run_name=f"{study_name}__trial_{trial.number:03d}",
        checkpoint_dir=str(train_config.checkpoint_dir),
        log_dir=str(train_config.log_dir),
        max_epochs=max_epochs,
        deterministic=train_config.deterministic,
        amp=train_config.amp,
        grad_clip_norm=_gradient_clip_value(train_config.grad_clip_norm),
    )
    trainer.num_sanity_val_steps = 0
    trainer.callbacks.append(callback)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    return callback.result()


def validate_lightning_module(
    *,
    lightning_module: KDLightningModule,
    val_loader: DataLoader,
    run_name: str,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    deterministic: bool,
    amp: bool,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    """Validate a KDLightningModule and return scalar Lightning metrics."""
    trainer = build_trainer(
        run_name=run_name,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        max_epochs=1,
        deterministic=deterministic,
        amp=amp,
        grad_clip_norm=_gradient_clip_value(grad_clip_norm),
    )
    results = trainer.validate(model=lightning_module, dataloaders=val_loader)
    if not results:
        raise RuntimeError("Lightning validation returned no metrics")
    return {name: float(value) for name, value in results[0].items()}


def _metric_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _gradient_clip_value(value: float | None) -> float:
    return 0.0 if value is None else float(value)
