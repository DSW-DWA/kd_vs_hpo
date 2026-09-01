import csv
import gc
import time
import warnings
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from torch.utils.data import DataLoader

from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.flops import CounterMode, FlopsBudgetTracker
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.common.utils import set_seed
from kd_vs_hpo.hpo.config import (
    SAMPLER_NAMES,
    HPOExperimentConfig,
    PrunerName,
    SamplerName,
)
from kd_vs_hpo.hpo.persistence import save_study_progress
from kd_vs_hpo.hpo.training import build_lightning_module, fit_lightning_trial

SAMPLER_OFFSETS = {name: offset for offset, name in enumerate(SAMPLER_NAMES)}


def run_study(
    *,
    architecture: dict[str, Any],
    sampler_name: SamplerName,
    pruner_name: PrunerName,
    experiment: HPOExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    forward_flops_per_sample: int,
    n_train: int,
    n_val: int,
    progress_dir: Path | None = None,
    storage_path: Path | None = None,
    n_trials: int | None = None,
    sampler_seed_offset: int = 0,
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
    study = _create_study(
        study_name=study_name,
        sampler_name=sampler_name,
        pruner_name=pruner_name,
        sampler_seed=sampler_seed + sampler_seed_offset,
        experiment=experiment,
        storage_path=storage_path,
    )
    initial_trial = _initial_trial_parameters(experiment)
    if storage_path is None and sampler_name != "grid":
        study.enqueue_trial(initial_trial)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *experiment.search_space.lr, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", *experiment.search_space.weight_decay, log=True
        )
        trial_seed = (
            experiment.train.seed
            if experiment.optuna.fixed_trial_seed
            else trial_seed_base * 10_000 + trial.number
        )
        trial_started = time.perf_counter()
        set_seed(trial_seed, deterministic=experiment.train.deterministic)
        model = create_nats_model(architecture)
        objective_train_loader = train_loader
        objective_val_loader = val_loader
        if experiment.optuna.fresh_dataloaders_per_trial:
            objective_train_loader, objective_val_loader, _, *_ = (
                build_cifar10_dataloaders(
                    experiment.train.checkpoint_dir,
                    experiment.train.log_dir,
                    experiment.train.data_root,
                    trial_seed,
                    experiment.train.batch_size,
                    experiment.train.num_workers,
                    experiment.train.validation_fraction,
                    device,
                )
            )
        elif objective_train_loader.generator is not None:
            objective_train_loader.generator.manual_seed(trial_seed)
        checkpoint = _checkpoint_path(experiment.output_dir, study_name, trial.number)
        flops_tracker = FlopsBudgetTracker(
            budget=(train_flops_per_epoch + validation_flops_per_epoch)
            * experiment.optuna.max_epochs,
            mode=CounterMode.SILENT,
        )
        lightning_module = build_lightning_module(
            model=model,
            architecture=architecture,
            train_config=experiment.train,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=experiment.optuna.max_epochs,
            forward_flops_per_sample=forward_flops_per_sample,
            flops_tracker=flops_tracker,
        )
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
            outcome = fit_lightning_trial(
                lightning_module=lightning_module,
                train_loader=objective_train_loader,
                val_loader=objective_val_loader,
                trial=trial,
                study_name=study_name,
                checkpoint_path=checkpoint,
                epoch_records=epoch_records,
                train_config=experiment.train,
                max_epochs=experiment.optuna.max_epochs,
                device=device,
            )
            completed_epochs = outcome.completed_epochs
            best_epoch = outcome.best_epoch
            best_val_acc1 = outcome.best_val_acc1
            trial_train_flops = outcome.train_flops
            trial_validation_flops = outcome.validation_flops
            if outcome.pruned:
                stop_reason = f"PRUNED_{pruner_name.upper()}"
                raise optuna.TrialPruned(stop_reason)
            if completed_epochs < experiment.optuna.max_epochs:
                stop_reason = "EARLY_STOPPING"

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
            del lightning_module, model
            if experiment.optuna.fresh_dataloaders_per_trial:
                del objective_train_loader, objective_val_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    study.optimize(
        objective,
        n_trials=experiment.optuna.n_trials if n_trials is None else n_trials,
    )
    return trial_records, epoch_records


def import_trials(
    *,
    architecture: dict[str, Any],
    sampler_name: SamplerName,
    pruner_name: PrunerName,
    experiment: HPOExperimentConfig,
    storage_path: Path,
    forward_flops_per_sample: int,
    n_train: int,
    n_val: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    imported_trials = experiment.imported_trials
    if not imported_trials:
        raise ValueError("No imported trials are configured")
    for imported in imported_trials:
        if not imported.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Imported checkpoint was not found: {imported.checkpoint_path}"
            )
        if not imported.metrics_path.is_file():
            raise FileNotFoundError(
                f"Imported metrics were not found: {imported.metrics_path}"
            )
    if storage_path.exists():
        raise FileExistsError(
            f"Optuna journal already exists: {storage_path}. "
            "Use a new output_dir for a new experiment."
        )

    sampler_seed = (
        experiment.train.seed
        + architecture["arch_row"] * 100
        + SAMPLER_OFFSETS[sampler_name]
    )
    study_name = f"arch_{architecture['arch_index']}__{sampler_name}_{pruner_name}"
    study = _create_study(
        study_name=study_name,
        sampler_name=sampler_name,
        pruner_name=pruner_name,
        sampler_seed=sampler_seed,
        experiment=experiment,
        storage_path=storage_path,
    )
    records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    for trial_id, imported in enumerate(imported_trials):
        curve = _read_validation_curve(imported.metrics_path, study_name, trial_id)
        parameters = {"lr": imported.lr, "weight_decay": imported.weight_decay}
        study.enqueue_trial(
            parameters,
            user_attrs={
                "imported": True,
                "source_checkpoint": str(imported.checkpoint_path),
            },
        )

        def replay_imported(
            trial: optuna.Trial,
            validation_curve: list[dict[str, Any]] = curve,
        ) -> float:
            trial.suggest_float("lr", *experiment.search_space.lr, log=True)
            trial.suggest_float(
                "weight_decay", *experiment.search_space.weight_decay, log=True
            )
            for row in validation_curve:
                trial.report(row["val_acc1"], step=row["epoch"])
                if trial.should_prune():
                    raise RuntimeError(
                        f"Imported trial_{trial.number} was unexpectedly pruned"
                    )
            return max(row["val_acc1"] for row in validation_curve)

        study.optimize(replay_imported, n_trials=1)
        frozen = study.trials[trial_id]
        if (
            frozen.number != trial_id
            or frozen.state != optuna.trial.TrialState.COMPLETE
        ):
            raise RuntimeError(
                f"The imported checkpoint did not become COMPLETE trial_{trial_id}"
            )

        checkpoint = _checkpoint_path(experiment.output_dir, study_name, trial_id)
        _save_imported_checkpoint(imported.checkpoint_path, checkpoint, architecture)
        best_row = max(curve, key=lambda row: row["val_acc1"])
        completed_epochs = max(int(row["epoch"]) for row in curve)
        train_flops = int(
            experiment.train.train_step_multiplier
            * forward_flops_per_sample
            * n_train
            * completed_epochs
        )
        validation_flops = int(forward_flops_per_sample * n_val * completed_epochs)
        context = {
            "study_name": study_name,
            "sampler": sampler_name,
            "pruner": pruner_name,
            "arch_row": architecture["arch_row"],
            "arch_index": architecture["arch_index"],
            "arch_str": architecture["arch_str"],
            "trial_id": trial_id,
            "trial_seed": experiment.train.seed,
            "lr": parameters["lr"],
            "weight_decay": parameters["weight_decay"],
        }
        record = _trial_record(
            context=context,
            state="COMPLETE",
            stop_reason="IMPORTED_CHECKPOINT",
            completed_epochs=completed_epochs,
            best_epoch=int(best_row["epoch"]),
            best_val_acc1=float(best_row["val_acc1"]),
            checkpoint=checkpoint,
            train_flops=train_flops,
            validation_flops=validation_flops,
        )
        record["trial_seconds"] = 0.0
        records.append(record)
        epoch_records.extend(curve)
    return records, epoch_records


def _create_study(
    *,
    study_name: str,
    sampler_name: SamplerName,
    pruner_name: PrunerName,
    sampler_seed: int,
    experiment: HPOExperimentConfig,
    storage_path: Path | None,
) -> optuna.Study:
    storage = None
    if storage_path is not None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = JournalStorage(JournalFileBackend(str(storage_path)))
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=_create_sampler(
            sampler_name,
            sampler_seed,
            experiment,
            distributed=storage is not None,
        ),
        pruner=_create_pruner(pruner_name, experiment),
        storage=storage,
        load_if_exists=storage is not None,
    )


def _read_validation_curve(
    metrics_path: Path,
    study_name: str,
    trial_id: int,
) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, str]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("epoch") or not row.get("val_acc"):
                continue
            by_epoch[int(row["epoch"])] = row
    if not by_epoch:
        raise ValueError(f"No validation metrics were found in {metrics_path}")

    records = []
    best = float("-inf")
    train_flops_per_epoch = 0
    validation_flops_per_epoch = 0
    for epoch_index in sorted(by_epoch):
        source = by_epoch[epoch_index]
        val_acc1 = 100.0 * float(source["val_acc"])
        best = max(best, val_acc1)
        train_flops_per_epoch = int(float(source.get("train_flops") or 0))
        validation_flops_per_epoch = int(float(source.get("val_flops") or 0))
        epoch = epoch_index + 1
        records.append(
            {
                "study_name": study_name,
                "trial_id": trial_id,
                "epoch": epoch,
                "train_loss": float("nan"),
                "val_acc1": val_acc1,
                "best_val_acc1": best,
                "learning_rate": float(source.get("lr") or "nan"),
                "cumulative_trial_flops": epoch
                * (train_flops_per_epoch + validation_flops_per_epoch),
            }
        )
    return records


def _save_imported_checkpoint(
    source_path: Path,
    target_path: Path,
    architecture: dict[str, Any],
) -> None:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model_state = source.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Model weights were not found in {source_path}")
    model = create_nats_model(architecture)
    model.load_state_dict(model_state, strict=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(".tmp")
    torch.save(
        {
            "model": {
                name: value.detach().cpu() for name, value in model_state.items()
            },
            "arch_record": architecture,
        },
        temporary_path,
    )
    temporary_path.replace(target_path)


def _create_sampler(
    name: SamplerName,
    seed: int,
    experiment: HPOExperimentConfig,
    *,
    distributed: bool = False,
) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=experiment.optuna.startup_trials,
            constant_liar=distributed,
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
