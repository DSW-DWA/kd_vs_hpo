import gc
import logging
import time
from pathlib import Path
from typing import Any

import optuna
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.common.nats import create_nats_model
from src.common.optim import create_optimizer_and_scheduler
from src.common.utils import set_seed
from src.hpo.domain.config import (
    HPOExperimentConfig,
    PrunerName,
    SamplerName,
)
from src.hpo.domain.stopping import AccuracyGrowthStopper
from src.hpo.infrastructure.training import evaluate, train_one_epoch
from src.hpo.infrastructure.event_log import ExperimentEventLog

logger = logging.getLogger(__name__)

SAMPLER_OFFSETS = {"tpe": 0, "grid": 1, "cmaes": 2, "gp": 3}


def run_study(
    *,
    architecture: dict[str, Any],
    sampler_name: SamplerName,
    pruner_name: PrunerName,
    experiment: HPOExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    event_log: ExperimentEventLog,
    forward_flops_per_sample: int,
    n_train: int,
    n_val: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = experiment.train.seed + architecture["arch_row"] * 100 + SAMPLER_OFFSETS[sampler_name]
    strategy = f"{sampler_name}_{pruner_name}"
    study_name = f"arch_{architecture['arch_index']}__{strategy}"
    trial_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    train_flops_per_epoch = int(
        experiment.train.train_step_multiplier * forward_flops_per_sample * n_train
    )
    validation_flops_per_epoch = int(forward_flops_per_sample * n_val)
    total_flops_per_epoch = train_flops_per_epoch + validation_flops_per_epoch
    study_train_flops = 0
    study_validation_flops = 0
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=_create_sampler(sampler_name, seed, experiment),
        pruner=_create_pruner(pruner_name, experiment),
    )
    study_started = time.perf_counter()
    event_log.emit(
        "study_started",
        study_name=study_name,
        strategy=strategy,
        sampler=sampler_name,
        pruner=pruner_name,
        arch_row=architecture["arch_row"],
        arch_index=architecture["arch_index"],
        seed=seed,
        planned_trials=experiment.optuna.n_trials,
        max_epochs=experiment.optuna.max_epochs,
        forward_flops_per_sample=forward_flops_per_sample,
        train_flops_per_epoch=train_flops_per_epoch,
        validation_flops_per_epoch=validation_flops_per_epoch,
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal study_train_flops, study_validation_flops
        lr = trial.suggest_float("lr", *experiment.search_space.lr, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", *experiment.search_space.weight_decay, log=True
        )
        trial_seed = seed * 10_000 + trial.number
        trial_started = time.perf_counter()
        event_log.emit(
            "trial_started",
            study_name=study_name,
            strategy=strategy,
            arch_index=architecture["arch_index"],
            trial_id=trial.number,
            trial_seed=trial_seed,
            lr=lr,
            weight_decay=weight_decay,
        )
        set_seed(trial_seed, deterministic=experiment.train.deterministic)
        model = create_nats_model(architecture).to(device)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            lr=lr,
            weight_decay=weight_decay,
            schedule_max_epochs=experiment.optuna.max_epochs,
            momentum=experiment.train.momentum,
        )
        scaler = GradScaler(enabled=experiment.train.amp and device.type == "cuda")
        stopper = AccuracyGrowthStopper(experiment.early_stopping)
        checkpoint = _checkpoint_path(experiment.output_dir, study_name, trial.number)
        best_val_acc1 = float("-inf")
        best_epoch = 0
        completed_epochs = 0
        stop_reason = "MAX_EPOCHS"
        trial_train_flops = 0
        trial_validation_flops = 0
        event_log.emit(
            "trial_initialized",
            study_name=study_name,
            strategy=strategy,
            arch_index=architecture["arch_index"],
            trial_id=trial.number,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            amp_enabled=scaler.is_enabled(),
            optimizer=type(optimizer).__name__,
            scheduler=type(scheduler).__name__,
            checkpoint_path=str(checkpoint),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        try:
            for epoch in range(1, experiment.optuna.max_epochs + 1):
                epoch_started = time.perf_counter()
                current_lr = float(optimizer.param_groups[0]["lr"])
                event_log.emit(
                    "epoch_started",
                    study_name=study_name,
                    strategy=strategy,
                    arch_index=architecture["arch_index"],
                    trial_id=trial.number,
                    epoch=epoch,
                    learning_rate=current_lr,
                )
                train_started = time.perf_counter()
                train_loss = train_one_epoch(
                    model,
                    optimizer,
                    train_loader,
                    scaler,
                    experiment.train.grad_clip_norm,
                    device,
                )
                train_seconds = time.perf_counter() - train_started
                validation_started = time.perf_counter()
                val_acc1 = evaluate(model, val_loader, device)
                validation_seconds = time.perf_counter() - validation_started
                scheduler.step()
                completed_epochs = epoch
                trial_train_flops += train_flops_per_epoch
                trial_validation_flops += validation_flops_per_epoch
                study_train_flops += train_flops_per_epoch
                study_validation_flops += validation_flops_per_epoch
                improved = val_acc1 > best_val_acc1
                if improved:
                    best_val_acc1 = val_acc1
                    best_epoch = epoch
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "arch_record": architecture},
                        checkpoint,
                    )
                    event_log.emit(
                        "checkpoint_saved",
                        study_name=study_name,
                        strategy=strategy,
                        arch_index=architecture["arch_index"],
                        trial_id=trial.number,
                        epoch=epoch,
                        val_acc1=val_acc1,
                        checkpoint_path=str(checkpoint),
                    )

                trial.report(val_acc1, step=epoch)
                growth_stop = stopper.update(val_acc1, epoch)
                pruner_decision = trial.should_prune()
                epoch_seconds = time.perf_counter() - epoch_started
                peak_gpu_memory = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                epoch_record = {
                        "study_name": study_name,
                        "strategy": strategy,
                        "sampler": sampler_name,
                        "pruner": pruner_name,
                        "arch_row": architecture["arch_row"],
                        "arch_index": architecture["arch_index"],
                        "trial_id": trial.number,
                        "epoch": epoch,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "train_loss": train_loss,
                        "val_acc1": val_acc1,
                        "best_val_acc1": best_val_acc1,
                        "accuracy_growth": stopper.last_growth,
                        "growth_threshold": experiment.early_stopping.min_growth,
                        "improved": improved,
                        "pruner_decision": pruner_decision,
                        "growth_stop_reason": growth_stop,
                        "current_lr": current_lr,
                        "next_lr": float(optimizer.param_groups[0]["lr"]),
                        "train_seconds": train_seconds,
                        "validation_seconds": validation_seconds,
                        "epoch_seconds": epoch_seconds,
                        "trial_seconds": time.perf_counter() - trial_started,
                        "peak_gpu_memory_bytes": peak_gpu_memory,
                        "forward_flops_per_sample": forward_flops_per_sample,
                        "train_flops_epoch": train_flops_per_epoch,
                        "validation_flops_epoch": validation_flops_per_epoch,
                        "total_flops_epoch": total_flops_per_epoch,
                        "cumulative_trial_train_flops": trial_train_flops,
                        "cumulative_trial_validation_flops": trial_validation_flops,
                        "cumulative_trial_flops": (
                            trial_train_flops + trial_validation_flops
                        ),
                        "cumulative_study_flops": (
                            study_train_flops + study_validation_flops
                        ),
                    }
                epoch_records.append(epoch_record)
                event_log.emit("epoch_completed", **epoch_record)
                if growth_stop is not None:
                    stop_reason = growth_stop
                    break
                if pruner_decision:
                    stop_reason = f"PRUNED_{pruner_name.upper()}"
                    raise optuna.TrialPruned(stop_reason)

            record = _trial_record(
                architecture, study_name, strategy, sampler_name, pruner_name,
                trial.number, trial_seed, lr, weight_decay, "COMPLETE",
                stop_reason, completed_epochs, best_epoch, best_val_acc1, checkpoint,
                trial_train_flops, trial_validation_flops,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            event_log.emit("trial_completed", **record)
            return best_val_acc1
        except optuna.TrialPruned:
            record = _trial_record(
                architecture, study_name, strategy, sampler_name, pruner_name,
                trial.number, trial_seed, lr, weight_decay, "PRUNED",
                stop_reason, completed_epochs, best_epoch, best_val_acc1, checkpoint,
                trial_train_flops, trial_validation_flops,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            event_log.emit("trial_pruned", **record)
            raise
        except Exception as error:
            event_log.emit(
                "trial_failed",
                study_name=study_name,
                strategy=strategy,
                arch_index=architecture["arch_index"],
                trial_id=trial.number,
                completed_epochs=completed_epochs,
                error_type=type(error).__name__,
                error_message=str(error),
                trial_seconds=time.perf_counter() - trial_started,
                train_flops=trial_train_flops,
                validation_flops=trial_validation_flops,
                total_flops=trial_train_flops + trial_validation_flops,
            )
            raise
        finally:
            del scaler, scheduler, optimizer, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    try:
        study.optimize(objective, n_trials=experiment.optuna.n_trials)
    except Exception as error:
        event_log.emit(
            "study_failed",
            study_name=study_name,
            strategy=strategy,
            error_type=type(error).__name__,
            error_message=str(error),
            study_seconds=time.perf_counter() - study_started,
        )
        raise
    complete_values = [
        record["best_val_acc1"]
        for record in trial_records
        if record["state"] == "COMPLETE"
    ]
    event_log.emit(
        "study_completed",
        study_name=study_name,
        strategy=strategy,
        arch_index=architecture["arch_index"],
        complete_trials=sum(record["state"] == "COMPLETE" for record in trial_records),
        pruned_trials=sum(record["state"] == "PRUNED" for record in trial_records),
        best_val_acc1=max(complete_values) if complete_values else None,
        study_seconds=time.perf_counter() - study_started,
        total_train_flops=study_train_flops,
        total_validation_flops=study_validation_flops,
        total_study_flops=study_train_flops + study_validation_flops,
    )
    return trial_records, epoch_records


def _create_sampler(
    name: SamplerName, seed: int, experiment: HPOExperimentConfig
) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=experiment.optuna.startup_trials
        )
    if name == "grid":
        return optuna.samplers.GridSampler(
            {
                "lr": experiment.search_space.grid_lr,
                "weight_decay": experiment.search_space.grid_weight_decay,
            },
            seed=seed,
        )
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(
            seed=seed, n_startup_trials=experiment.optuna.startup_trials
        )
    if name == "gp":
        return optuna.samplers.GPSampler(
            seed=seed,
            n_startup_trials=experiment.optuna.startup_trials,
        )
    raise ValueError(f"Unsupported sampler: {name}")


def _create_pruner(
    name: PrunerName, experiment: HPOExperimentConfig
) -> optuna.pruners.BasePruner:
    config = experiment.optuna
    if name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=config.min_resource,
            reduction_factor=config.reduction_factor,
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=config.min_resource,
            max_resource=config.max_epochs,
            reduction_factor=config.reduction_factor,
        )
    raise ValueError(f"Unsupported pruner: {name}")


def _checkpoint_path(output_dir: Path, study_name: str, trial_id: int) -> Path:
    return output_dir / "checkpoints" / study_name / f"trial_{trial_id:03d}.pt"


def _trial_record(
    architecture: dict[str, Any],
    study_name: str,
    strategy: str,
    sampler: str,
    pruner: str,
    trial_id: int,
    trial_seed: int,
    lr: float,
    weight_decay: float,
    state: str,
    stop_reason: str,
    completed_epochs: int,
    best_epoch: int,
    best_val_acc1: float,
    checkpoint: Path,
    train_flops: int,
    validation_flops: int,
) -> dict[str, Any]:
    return {
        "study_name": study_name,
        "strategy": strategy,
        "sampler": sampler,
        "pruner": pruner,
        "arch_row": architecture["arch_row"],
        "arch_index": architecture["arch_index"],
        "arch_str": architecture["arch_str"],
        "trial_id": trial_id,
        "trial_seed": trial_seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "state": state,
        "stop_reason": stop_reason,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_val_acc1": best_val_acc1,
        "checkpoint_path": str(checkpoint),
        "train_flops": train_flops,
        "validation_flops": validation_flops,
        "total_flops": train_flops + validation_flops,
    }
