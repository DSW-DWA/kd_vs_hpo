import gc
import time
import warnings
from pathlib import Path
from typing import Any

import optuna
import torch
from torch.amp import GradScaler
from torch.utils.data import Dataset

from src.common.dataloader import build_dataloader
from src.common.nats import create_nats_model
from src.common.optim import create_optimizer_and_scheduler
from src.common.utils import set_seed
from src.hpo.config import (
    HPOExperimentConfig,
    PrunerName,
    SAMPLER_NAMES,
    SamplerName,
)
from src.hpo.persistence import save_study_progress
from src.hpo.training import evaluate, train_one_epoch

SAMPLER_OFFSETS = {name: offset for offset, name in enumerate(SAMPLER_NAMES)}


def run_study(
    *,
    architecture: dict[str, Any],
    sampler_name: SamplerName,
    pruner_name: PrunerName,
    experiment: HPOExperimentConfig,
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    device: torch.device,
    forward_flops_per_sample: int,
    n_train: int,
    n_val: int,
    progress_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampler_seed = (
        experiment.train.seed
        + architecture["arch_row"] * 100
        + SAMPLER_OFFSETS[sampler_name]
    )
    trial_seed_base = experiment.train.seed + architecture["arch_row"] * 100
    strategy = f"{sampler_name}_{pruner_name}"
    study_name = f"arch_{architecture['arch_index']}__{strategy}"
    trial_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    study_record = {
        "study_name": study_name,
        "sampler": sampler_name,
        "pruner": pruner_name,
        "arch_row": architecture["arch_row"],
        "arch_index": architecture["arch_index"],
        "arch_str": architecture["arch_str"],
    }
    train_flops_per_epoch = int(
        experiment.train.train_step_multiplier * forward_flops_per_sample * n_train
    )
    validation_flops_per_epoch = int(forward_flops_per_sample * n_val)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=_create_sampler(sampler_name, sampler_seed, experiment),
        pruner=_create_pruner(pruner_name, experiment),
    )
    initial_trial = _initial_trial_parameters(experiment)
    if sampler_name != "grid":
        study.enqueue_trial(initial_trial)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *experiment.search_space.lr, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", *experiment.search_space.weight_decay, log=True
        )
        trial_seed = trial_seed_base * 10_000 + trial.number
        trial_started = time.perf_counter()
        set_seed(trial_seed, deterministic=experiment.train.deterministic)
        train_loader = build_dataloader(
            train_dataset,
            experiment.train,
            device,
            shuffle=True,
            seed=trial_seed,
        )
        val_loader = build_dataloader(
            val_dataset,
            experiment.train,
            device,
            shuffle=False,
            seed=trial_seed + 1,
        )
        model = create_nats_model(architecture).to(device)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            lr=lr,
            weight_decay=weight_decay,
            schedule_max_epochs=experiment.optuna.max_epochs,
            momentum=experiment.train.momentum,
        )
        scaler = GradScaler(enabled=experiment.train.amp and device.type == "cuda")
        checkpoint = _checkpoint_path(experiment.output_dir, study_name, trial.number)
        best_val_acc1 = float("-inf")
        best_epoch = 0
        completed_epochs = 0
        stop_reason = "MAX_EPOCHS"
        trial_train_flops = 0
        trial_validation_flops = 0
        trial_record_context = {
            "study_name": study_name,
            "sampler": sampler_name,
            "pruner": pruner_name,
            "arch_row": architecture["arch_row"],
            "arch_index": architecture["arch_index"],
            "arch_str": architecture["arch_str"],
            "trial_id": trial.number,
            "trial_seed": trial_seed,
            "lr": lr,
            "weight_decay": weight_decay,
        }

        try:
            for epoch in range(1, experiment.optuna.max_epochs + 1):
                current_lr = float(optimizer.param_groups[0]["lr"])
                train_loss = train_one_epoch(
                    model,
                    optimizer,
                    train_loader,
                    scaler,
                    experiment.train.grad_clip_norm,
                    device,
                )
                val_acc1 = evaluate(model, val_loader, device)
                scheduler.step()
                completed_epochs = epoch
                trial_train_flops += train_flops_per_epoch
                trial_validation_flops += validation_flops_per_epoch
                improved = val_acc1 > best_val_acc1
                if improved:
                    best_val_acc1 = val_acc1
                    best_epoch = epoch
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "arch_record": architecture},
                        checkpoint,
                    )

                trial.report(best_val_acc1, step=epoch)
                pruner_decision = trial.should_prune()
                epoch_record = {
                    "study_name": study_name,
                    "trial_id": trial.number,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_acc1": val_acc1,
                    "best_val_acc1": best_val_acc1,
                    "learning_rate": current_lr,
                    "cumulative_trial_flops": (
                        trial_train_flops + trial_validation_flops
                    ),
                }
                epoch_records.append(epoch_record)
                if pruner_decision:
                    stop_reason = f"PRUNED_{pruner_name.upper()}"
                    raise optuna.TrialPruned(stop_reason)

            record = _trial_record(
                context=trial_record_context,
                state="COMPLETE",
                stop_reason=stop_reason,
                completed_epochs=completed_epochs,
                best_epoch=best_epoch,
                best_val_acc1=best_val_acc1,
                checkpoint=checkpoint,
                train_flops=trial_train_flops,
                validation_flops=trial_validation_flops,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            return best_val_acc1
        except optuna.TrialPruned:
            record = _trial_record(
                context=trial_record_context,
                state="PRUNED",
                stop_reason=stop_reason,
                completed_epochs=completed_epochs,
                best_epoch=best_epoch,
                best_val_acc1=best_val_acc1,
                checkpoint=checkpoint,
                train_flops=trial_train_flops,
                validation_flops=trial_validation_flops,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            raise
        except Exception as error:
            record = _trial_record(
                context=trial_record_context,
                state="FAILED",
                stop_reason=f"FAILED_{type(error).__name__.upper()}",
                completed_epochs=completed_epochs,
                best_epoch=best_epoch,
                best_val_acc1=best_val_acc1,
                checkpoint=checkpoint,
                train_flops=trial_train_flops,
                validation_flops=trial_validation_flops,
                error=error,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            raise
        except BaseException as error:
            record = _trial_record(
                context=trial_record_context,
                state="INTERRUPTED",
                stop_reason=f"INTERRUPTED_{type(error).__name__.upper()}",
                completed_epochs=completed_epochs,
                best_epoch=best_epoch,
                best_val_acc1=best_val_acc1,
                checkpoint=checkpoint,
                train_flops=trial_train_flops,
                validation_flops=trial_validation_flops,
                error=error,
            )
            record["trial_seconds"] = time.perf_counter() - trial_started
            trial_records.append(record)
            raise
        finally:
            if progress_dir is not None:
                try:
                    save_study_progress(
                        progress_dir,
                        study_record,
                        trial_records,
                        epoch_records,
                    )
                except OSError as error:
                    warnings.warn(
                        f"Could not save recovery data for {study_name}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            del train_loader, val_loader, scaler, scheduler, optimizer, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    study.optimize(objective, n_trials=experiment.optuna.n_trials)
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
            seed=seed,
            n_startup_trials=experiment.optuna.startup_trials,
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
    if name == "none":
        return optuna.pruners.NopPruner()
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


def _initial_trial_parameters(experiment: HPOExperimentConfig) -> dict[str, float]:
    return {
        "lr": experiment.search_space.initial_lr,
        "weight_decay": experiment.search_space.initial_weight_decay,
    }


def _checkpoint_path(output_dir: Path, study_name: str, trial_id: int) -> Path:
    return output_dir / "checkpoints" / study_name / f"trial_{trial_id:03d}.pt"


def _trial_record(
    *,
    context: dict[str, Any],
    state: str,
    stop_reason: str,
    completed_epochs: int,
    best_epoch: int,
    best_val_acc1: float,
    checkpoint: Path,
    train_flops: int,
    validation_flops: int,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        **context,
        "state": state,
        "stop_reason": stop_reason,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_val_acc1": best_val_acc1,
        "checkpoint_path": str(checkpoint),
        "train_flops": train_flops,
        "validation_flops": validation_flops,
        "total_flops": train_flops + validation_flops,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
