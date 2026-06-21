from __future__ import annotations

import gc
import json
import logging
import math
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
    TPESampler,
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.common.optim import create_optimizer_and_scheduler
from kd_vs_hpo.common.utils import (
    accuracy_top1_from_logits,
    extract_logits,
    set_seed,
)

logger = logging.getLogger(__name__)

SamplerName = Literal["grid", "tpe", "gp", "cmaes", "qmc", "nsgaii"]
PrunerName = Literal["successive_halving", "hyperband"]

DEFAULT_SAMPLERS: tuple[SamplerName, ...] = (
    "grid",
    "tpe",
    "gp",
    "cmaes",
    "qmc",
    "nsgaii",
)
DEFAULT_PRUNERS: tuple[PrunerName, ...] = (
    "successive_halving",
    "hyperband",
)

DEFAULT_GRID_LR: tuple[float, ...] = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
)
DEFAULT_GRID_WEIGHT_DECAY: tuple[float, ...] = (
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
)


@dataclass(frozen=True)
class OptunaSearchSpace:
    lr: tuple[float, float] = (1e-4, 1.0)
    weight_decay: tuple[float, float] = (1e-7, 1e-2)
    grid_lr: tuple[float, ...] = DEFAULT_GRID_LR
    grid_weight_decay: tuple[float, ...] = DEFAULT_GRID_WEIGHT_DECAY


@dataclass(frozen=True)
class PlateauConfig:
    warmup_epochs: int = 30
    patience: int = 25
    min_delta: float = 0.05
    smoothing_window: int = 5
    max_epochs: int = 300


@dataclass(frozen=True)
class StudyStopConfig:
    min_started_trials: int = 80
    min_complete_trials: int = 40
    stagnation_window: int = 40
    min_improvement: float = 0.05
    max_started_trials: int = 300
    qmc_max_started_trials: int = 256


@dataclass(frozen=True)
class PruningConfig:
    min_resource: int = 20
    reduction_factor: int = 3
    min_early_stopping_rate: int = 0
    bootstrap_count: int = 0


@dataclass(frozen=True)
class OptunaExperimentConfig:
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(num_workers=4),
    )
    search_space: OptunaSearchSpace = field(default_factory=OptunaSearchSpace)
    plateau: PlateauConfig = field(default_factory=PlateauConfig)
    study_stop: StudyStopConfig = field(default_factory=StudyStopConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    architectures_path: Path = Path("experiments/nats_architectures_10.json")
    costs_path: Path = Path("experiments/sampled_architecture_costs.csv")
    output_dir: Path = Path("optuna_output")
    arch_rows: tuple[int, ...] | None = None
    samplers: tuple[SamplerName, ...] = DEFAULT_SAMPLERS
    pruners: tuple[PrunerName, ...] = DEFAULT_PRUNERS
    sampler_seeds: tuple[int, ...] = (42,)
    gpu_ids: tuple[int, ...] | None = None
    workers_per_gpu: int = 8
    torch_threads_per_worker: int = 1
    tpe_startup_trials: int = 20
    gp_startup_trials: int = 20
    cmaes_startup_trials: int = 20
    nsgaii_population_size: int = 20
    write_parquet: bool = True


@dataclass(frozen=True)
class OptunaExperimentResult:
    epoch_metrics_path: Path
    trial_summary_path: Path
    study_summary_path: Path
    optimization_history_path: Path
    parquet_paths: tuple[Path, ...]
    failed_studies: tuple[str, ...]


@dataclass(frozen=True)
class _StudyTask:
    arch_record: dict[str, Any]
    cost: dict[str, Any]
    sampler_name: SamplerName
    pruner_name: PrunerName
    sampler_seed: int

    @property
    def strategy_id(self) -> str:
        return f"{self.sampler_name}_{self.pruner_name}"

    @property
    def study_name(self) -> str:
        return (
            f"arch_{self.arch_record['arch_row']:02d}_"
            f"{self.arch_record['arch_index']}__"
            f"{self.strategy_id}__seed_{self.sampler_seed}"
        )


@dataclass
class _PlateauTracker:
    config: PlateauConfig
    values: deque[float] = field(init=False)
    best_smoothed_value: float = float("-inf")
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.config.smoothing_window)

    def update(self, value: float, epoch: int) -> tuple[float, bool]:
        self.values.append(value)
        smoothed = float(np.mean(self.values))
        if len(self.values) < self.config.smoothing_window:
            return smoothed, False

        if smoothed > self.best_smoothed_value + self.config.min_delta:
            self.best_smoothed_value = smoothed
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (
            epoch >= self.config.warmup_epochs
            and self.epochs_without_improvement >= self.config.patience
        )
        return smoothed, should_stop


_worker_device: torch.device | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _append_jsonl(file: TextIO, record: dict[str, Any]) -> None:
    file.write(
        json.dumps(
            record,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    file.flush()


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    temporary.replace(path)


def _to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _validate_config(config: OptunaExperimentConfig) -> None:
    if config.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1")
    if config.torch_threads_per_worker < 1:
        raise ValueError("torch_threads_per_worker must be at least 1")
    if config.train.num_workers < 0:
        raise ValueError("train.num_workers cannot be negative")
    if not config.samplers:
        raise ValueError("At least one sampler must be selected")
    if not config.pruners:
        raise ValueError("At least one pruner must be selected")
    if not config.sampler_seeds:
        raise ValueError("At least one sampler seed must be selected")

    low_lr, high_lr = config.search_space.lr
    low_wd, high_wd = config.search_space.weight_decay
    if low_lr <= 0 or high_lr <= low_lr:
        raise ValueError("Invalid learning-rate search space")
    if low_wd <= 0 or high_wd <= low_wd:
        raise ValueError("Invalid weight-decay search space")
    if any(value < low_lr or value > high_lr for value in config.search_space.grid_lr):
        raise ValueError("Grid learning rates must be inside the search space")
    if any(
        value < low_wd or value > high_wd
        for value in config.search_space.grid_weight_decay
    ):
        raise ValueError("Grid weight decays must be inside the search space")

    plateau = config.plateau
    if plateau.max_epochs < 1:
        raise ValueError("plateau.max_epochs must be at least 1")
    if plateau.warmup_epochs < 1 or plateau.warmup_epochs > plateau.max_epochs:
        raise ValueError("plateau.warmup_epochs must be within max_epochs")
    if plateau.patience < 1:
        raise ValueError("plateau.patience must be at least 1")
    if plateau.smoothing_window < 1:
        raise ValueError("plateau.smoothing_window must be at least 1")
    if plateau.min_delta < 0:
        raise ValueError("plateau.min_delta cannot be negative")

    stop = config.study_stop
    if stop.min_started_trials < 1:
        raise ValueError("min_started_trials must be at least 1")
    if stop.min_complete_trials < 1:
        raise ValueError("min_complete_trials must be at least 1")
    if stop.stagnation_window < 1:
        raise ValueError("stagnation_window must be at least 1")
    if stop.max_started_trials < stop.min_started_trials:
        raise ValueError("max_started_trials must be >= min_started_trials")

    pruning = config.pruning
    if pruning.min_resource < 1:
        raise ValueError("pruning.min_resource must be at least 1")
    if pruning.min_resource > plateau.max_epochs:
        raise ValueError("pruning.min_resource cannot exceed max_epochs")
    if pruning.reduction_factor < 2:
        raise ValueError("pruning.reduction_factor must be at least 2")


def _load_architectures(
    path: Path,
    rows: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
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


def _prepare_cifar10_data(data_root: Path) -> None:
    logger.info("Ensuring CIFAR-10 is available at %s", data_root)
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)


def _make_storage(journal_path: Path) -> JournalStorage:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock = JournalFileOpenLock(str(journal_path)) if os.name == "nt" else None
    return JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))


def _make_sampler(
    name: SamplerName,
    seed: int,
    config: OptunaExperimentConfig,
) -> optuna.samplers.BaseSampler:
    if name == "grid":
        return GridSampler(
            {
                "lr": list(config.search_space.grid_lr),
                "weight_decay": list(config.search_space.grid_weight_decay),
            },
            seed=seed,
        )
    if name == "tpe":
        return TPESampler(
            seed=seed,
            n_startup_trials=config.tpe_startup_trials,
            multivariate=True,
            constant_liar=True,
        )
    if name == "gp":
        return GPSampler(
            seed=seed,
            n_startup_trials=config.gp_startup_trials,
            deterministic_objective=False,
        )
    if name == "cmaes":
        return CmaEsSampler(
            seed=seed,
            n_startup_trials=config.cmaes_startup_trials,
            consider_pruned_trials=True,
        )
    if name == "qmc":
        return QMCSampler(
            qmc_type="sobol",
            scramble=True,
            seed=seed,
        )
    if name == "nsgaii":
        return NSGAIISampler(
            population_size=config.nsgaii_population_size,
            seed=seed,
        )
    raise ValueError(f"Unknown sampler: {name}")


def _make_pruner(
    name: PrunerName,
    config: OptunaExperimentConfig,
) -> optuna.pruners.BasePruner:
    pruning = config.pruning
    if name == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=pruning.min_resource,
            reduction_factor=pruning.reduction_factor,
            min_early_stopping_rate=pruning.min_early_stopping_rate,
            bootstrap_count=pruning.bootstrap_count,
        )
    if name == "hyperband":
        return HyperbandPruner(
            min_resource=pruning.min_resource,
            max_resource=config.plateau.max_epochs,
            reduction_factor=pruning.reduction_factor,
            bootstrap_count=pruning.bootstrap_count,
        )
    raise ValueError(f"Unknown pruner: {name}")


def _study_trial_limit(
    task: _StudyTask,
    config: OptunaExperimentConfig,
) -> int:
    if task.sampler_name == "grid":
        return len(config.search_space.grid_lr) * len(
            config.search_space.grid_weight_decay
        )
    if task.sampler_name == "qmc":
        return min(
            config.study_stop.max_started_trials,
            config.study_stop.qmc_max_started_trials,
        )
    return config.study_stop.max_started_trials


class _StudyStopper:
    def __init__(
        self,
        config: StudyStopConfig,
        *,
        trial_limit: int,
        stop_on_stagnation: bool,
    ) -> None:
        self._config = config
        self._trial_limit = trial_limit
        self._stop_on_stagnation = stop_on_stagnation

    def __call__(
        self,
        study: optuna.Study,
        _: optuna.trial.FrozenTrial,
    ) -> None:
        trials = study.get_trials(deepcopy=False)
        if len(trials) >= self._trial_limit:
            study.stop()
            return
        if not self._stop_on_stagnation:
            return
        if len(trials) < self._config.min_started_trials:
            return

        complete = [
            trial
            for trial in trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ]
        required = max(
            self._config.min_complete_trials,
            self._config.stagnation_window + 1,
        )
        if len(complete) < required:
            return

        values = [float(trial.value) for trial in complete]
        previous_best = max(values[: -self._config.stagnation_window])
        current_best = max(values)
        if current_best - previous_best < self._config.min_improvement:
            logger.info(
                "Stopping study=%s after %s trials: improvement over last "
                "%s COMPLETE trials is %.4f pp",
                study.study_name,
                len(trials),
                self._config.stagnation_window,
                current_best - previous_best,
            )
            study.stop()


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    scaler: GradScaler,
    grad_clip_norm: float | None,
    device: torch.device,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    use_amp = scaler.is_enabled() and device.type == "cuda"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
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
        total_loss / max(1, total_examples),
        100.0 * total_correct / max(1, total_examples),
        total_examples,
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
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
        total_loss / max(1, total_examples),
        100.0 * total_correct / max(1, total_examples),
        total_examples,
    )


def _trial_paths(
    config: OptunaExperimentConfig,
    task: _StudyTask,
    trial_number: int,
) -> tuple[Path, Path]:
    relative = (
        Path(
            f"arch_{task.arch_record['arch_row']:02d}_{task.arch_record['arch_index']}"
        )
        / task.strategy_id
        / f"seed_{task.sampler_seed}"
    )
    metrics = (
        config.output_dir / "metrics" / relative / f"trial_{trial_number:05d}.jsonl"
    )
    summary = (
        config.output_dir
        / "trial_summaries"
        / relative
        / f"trial_{trial_number:05d}.json"
    )
    metrics.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    return metrics, summary


def _set_trial_summary_attrs(
    trial: optuna.Trial,
    summary: dict[str, Any],
) -> None:
    attr_keys = (
        "stop_reason",
        "completed_epochs",
        "best_epoch",
        "best_val_acc1",
        "final_val_acc1",
        "total_train_flops",
        "total_validation_flops",
        "total_trial_flops",
        "total_seconds",
    )
    for key in attr_keys:
        value = summary.get(key)
        if value is not None and not (
            isinstance(value, float) and not math.isfinite(value)
        ):
            trial.set_user_attr(key, value)


def _build_objective(
    task: _StudyTask,
    config: OptunaExperimentConfig,
    device: torch.device,
) -> Any:
    forward_flops = int(float(task.cost["forward_flops_per_sample"]))

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float(
            "lr",
            config.search_space.lr[0],
            config.search_space.lr[1],
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            config.search_space.weight_decay[0],
            config.search_space.weight_decay[1],
            log=True,
        )
        train_seed = (
            config.train.seed
            + int(task.arch_record["arch_row"]) * 10_000
            + trial.number
        )
        trial_train_config = config.train
        metrics_path, summary_path = _trial_paths(config, task, trial.number)

        started_at = _utc_now()
        started_monotonic = time.monotonic()
        stop_reason = "FAIL"
        best_val_acc1 = float("-inf")
        best_epoch = 0
        final_val_acc1 = float("nan")
        completed_epochs = 0
        cumulative_train_flops = 0
        cumulative_validation_flops = 0
        cumulative_seconds = 0.0
        plateau = _PlateauTracker(config.plateau)

        model: nn.Module | None = None
        optimizer: torch.optim.Optimizer | None = None
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        scaler: GradScaler | None = None
        train_loader: DataLoader | None = None
        val_loader: DataLoader | None = None

        base_record = {
            "experiment_id": config.output_dir.name,
            "study_name": task.study_name,
            "strategy_id": task.strategy_id,
            "sampler": task.sampler_name,
            "pruner": task.pruner_name,
            "sampler_seed": task.sampler_seed,
            "architecture_row": int(task.arch_record["arch_row"]),
            "architecture_index": int(task.arch_record["arch_index"]),
            "architecture_str": str(task.arch_record["arch_str"]),
            "trial_number": trial.number,
            "trial_seed": train_seed,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": config.train.momentum,
            "forward_flops_per_sample": forward_flops,
            "params": _finite_or_none(task.cost.get("params")),
            "gpu_id": device.index if device.type == "cuda" else None,
        }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        try:
            set_seed(train_seed, deterministic=config.train.deterministic)
            train_loader, val_loader, _, n_train, n_val, _ = build_cifar10_dataloaders(
                trial_train_config, device
            )
            model = create_nats_model(task.arch_record).to(device)
            optimizer, scheduler = create_optimizer_and_scheduler(
                model,
                lr=lr,
                weight_decay=weight_decay,
                schedule_max_epochs=config.plateau.max_epochs,
                momentum=config.train.momentum,
            )
            scaler = GradScaler(
                enabled=config.train.amp and device.type == "cuda",
            )
            criterion = nn.CrossEntropyLoss()

            with metrics_path.open("w", encoding="utf-8") as metrics_file:
                for epoch_index in range(config.plateau.max_epochs):
                    epoch = epoch_index + 1
                    epoch_started_at = _utc_now()
                    epoch_started_monotonic = time.monotonic()
                    current_lr = float(optimizer.param_groups[0]["lr"])

                    train_loss, train_acc1, train_examples = _train_one_epoch(
                        model,
                        optimizer,
                        criterion,
                        train_loader,
                        scaler,
                        config.train.grad_clip_norm,
                        device,
                    )
                    val_loss, val_acc1, val_examples = _evaluate(
                        model,
                        criterion,
                        val_loader,
                        device,
                    )
                    scheduler.step()

                    epoch_seconds = time.monotonic() - epoch_started_monotonic
                    cumulative_seconds += epoch_seconds
                    train_flops = int(
                        train_examples
                        * forward_flops
                        * config.train.train_step_multiplier
                    )
                    validation_flops = int(val_examples * forward_flops)
                    total_flops = train_flops + validation_flops
                    cumulative_train_flops += train_flops
                    cumulative_validation_flops += validation_flops

                    final_val_acc1 = val_acc1
                    if val_acc1 > best_val_acc1:
                        best_val_acc1 = val_acc1
                        best_epoch = epoch

                    smoothed_val_acc1, plateau_detected = plateau.update(
                        val_acc1,
                        epoch,
                    )
                    trial.report(val_acc1, step=epoch)
                    pruner_decision = trial.should_prune()
                    completed_epochs = epoch

                    peak_gpu_memory = (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    )
                    epoch_record = {
                        **base_record,
                        "epoch": epoch,
                        "epoch_started_at": epoch_started_at,
                        "epoch_ended_at": _utc_now(),
                        "current_lr": current_lr,
                        "next_lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": train_loss,
                        "train_acc1": train_acc1,
                        "val_loss": val_loss,
                        "val_acc1": val_acc1,
                        "smoothed_val_acc1": smoothed_val_acc1,
                        "best_val_acc1": best_val_acc1,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": (
                            plateau.epochs_without_improvement
                        ),
                        "plateau_detected": plateau_detected,
                        "pruner_decision": pruner_decision,
                        "train_examples": train_examples,
                        "validation_examples": val_examples,
                        "expected_train_examples": n_train,
                        "expected_validation_examples": n_val,
                        "train_flops_epoch": train_flops,
                        "validation_flops_epoch": validation_flops,
                        "total_flops_epoch": total_flops,
                        "cumulative_train_flops": cumulative_train_flops,
                        "cumulative_validation_flops": (cumulative_validation_flops),
                        "cumulative_trial_flops": (
                            cumulative_train_flops + cumulative_validation_flops
                        ),
                        "epoch_seconds": epoch_seconds,
                        "cumulative_trial_seconds": cumulative_seconds,
                        "samples_per_second": (
                            (train_examples + val_examples) / epoch_seconds
                            if epoch_seconds > 0
                            else None
                        ),
                        "peak_gpu_memory_bytes": peak_gpu_memory,
                    }
                    _append_jsonl(metrics_file, epoch_record)

                    logger.info(
                        "study=%s trial=%s epoch=%s val_acc1=%.3f "
                        "best=%.3f no_improve=%s Gflops=%.2f",
                        task.study_name,
                        trial.number,
                        epoch,
                        val_acc1,
                        best_val_acc1,
                        plateau.epochs_without_improvement,
                        (cumulative_train_flops + cumulative_validation_flops) / 1e9,
                    )

                    if pruner_decision:
                        stop_reason = f"PRUNED_{task.pruner_name.upper()}"
                        summary = {
                            **base_record,
                            "state": "PRUNED",
                            "stop_reason": stop_reason,
                            "started_at": started_at,
                            "ended_at": _utc_now(),
                            "completed_epochs": completed_epochs,
                            "best_epoch": best_epoch,
                            "best_val_acc1": best_val_acc1,
                            "final_val_acc1": final_val_acc1,
                            "best_smoothed_val_acc1": _finite_or_none(
                                plateau.best_smoothed_value
                            ),
                            "epochs_without_improvement": (
                                plateau.epochs_without_improvement
                            ),
                            "total_train_flops": cumulative_train_flops,
                            "total_validation_flops": (cumulative_validation_flops),
                            "total_trial_flops": (
                                cumulative_train_flops + cumulative_validation_flops
                            ),
                            "total_seconds": (time.monotonic() - started_monotonic),
                            "metrics_path": str(metrics_path),
                        }
                        _set_trial_summary_attrs(trial, summary)
                        _write_json(summary_path, summary)
                        raise optuna.TrialPruned(stop_reason)

                    if plateau_detected:
                        stop_reason = "PLATEAU"
                        break
                else:
                    stop_reason = "MAX_EPOCHS"

            summary = {
                **base_record,
                "state": "COMPLETE",
                "stop_reason": stop_reason,
                "started_at": started_at,
                "ended_at": _utc_now(),
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_val_acc1": best_val_acc1,
                "final_val_acc1": final_val_acc1,
                "best_smoothed_val_acc1": _finite_or_none(plateau.best_smoothed_value),
                "epochs_without_improvement": (plateau.epochs_without_improvement),
                "total_train_flops": cumulative_train_flops,
                "total_validation_flops": cumulative_validation_flops,
                "total_trial_flops": (
                    cumulative_train_flops + cumulative_validation_flops
                ),
                "total_seconds": time.monotonic() - started_monotonic,
                "metrics_path": str(metrics_path),
            }
            _set_trial_summary_attrs(trial, summary)
            _write_json(summary_path, summary)
            return best_val_acc1
        except optuna.TrialPruned:
            raise
        except Exception as error:
            stop_reason = "FAIL"
            summary = {
                **base_record,
                "state": "FAIL",
                "stop_reason": stop_reason,
                "started_at": started_at,
                "ended_at": _utc_now(),
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_val_acc1": (
                    best_val_acc1 if math.isfinite(best_val_acc1) else None
                ),
                "final_val_acc1": (
                    final_val_acc1 if math.isfinite(final_val_acc1) else None
                ),
                "best_smoothed_val_acc1": (
                    plateau.best_smoothed_value
                    if math.isfinite(plateau.best_smoothed_value)
                    else None
                ),
                "epochs_without_improvement": (plateau.epochs_without_improvement),
                "total_train_flops": cumulative_train_flops,
                "total_validation_flops": cumulative_validation_flops,
                "total_trial_flops": (
                    cumulative_train_flops + cumulative_validation_flops
                ),
                "total_seconds": time.monotonic() - started_monotonic,
                "metrics_path": str(metrics_path),
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
            }
            _write_json(summary_path, summary)
            logger.exception(
                "Trial failed: study=%s trial=%s",
                task.study_name,
                trial.number,
            )
            raise
        finally:
            del train_loader, val_loader, scaler, scheduler, optimizer, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def _initialize_study_worker(
    gpu_id: int | None,
    torch_threads_per_worker: int,
) -> None:
    global _worker_device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_num_threads(torch_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if gpu_id is None:
        _worker_device = torch.device("cpu")
    else:
        torch.cuda.set_device(gpu_id)
        _worker_device = torch.device(f"cuda:{gpu_id}")
    logger.info("Optuna worker initialized on %s", _worker_device)


def _run_study(
    task: _StudyTask,
    config: OptunaExperimentConfig,
) -> dict[str, Any]:
    if _worker_device is None:
        raise RuntimeError("Study worker was not initialized")

    storage = _make_storage(config.output_dir / "optuna_journal.log")
    sampler = _make_sampler(task.sampler_name, task.sampler_seed, config)
    pruner = _make_pruner(task.pruner_name, config)
    study = optuna.create_study(
        study_name=task.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    study.set_user_attr("architecture_row", int(task.arch_record["arch_row"]))
    study.set_user_attr("architecture_index", int(task.arch_record["arch_index"]))
    study.set_user_attr("sampler", task.sampler_name)
    study.set_user_attr("pruner", task.pruner_name)
    study.set_user_attr("sampler_seed", task.sampler_seed)
    study.set_user_attr("config", _to_jsonable(asdict(config)))

    trial_limit = _study_trial_limit(task, config)
    existing_trials = len(study.get_trials(deepcopy=False))
    remaining_trials = max(0, trial_limit - existing_trials)
    if remaining_trials:
        logger.info(
            "Starting study=%s existing_trials=%s remaining_limit=%s device=%s",
            task.study_name,
            existing_trials,
            remaining_trials,
            _worker_device,
        )
        stopper = _StudyStopper(
            config.study_stop,
            trial_limit=trial_limit,
            stop_on_stagnation=task.sampler_name != "grid",
        )
        study.optimize(
            _build_objective(task, config, _worker_device),
            n_trials=remaining_trials,
            callbacks=[stopper],
            catch=(Exception,),
            gc_after_trial=False,
            show_progress_bar=False,
        )

    trials = study.get_trials(deepcopy=False)
    counts = {
        state.name: sum(trial.state == state for trial in trials)
        for state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.RUNNING,
        )
    }
    complete = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    best = max(complete, key=lambda trial: float(trial.value)) if complete else None
    summary = {
        "study_name": task.study_name,
        "strategy_id": task.strategy_id,
        "sampler": task.sampler_name,
        "pruner": task.pruner_name,
        "sampler_seed": task.sampler_seed,
        "architecture_row": int(task.arch_record["arch_row"]),
        "architecture_index": int(task.arch_record["arch_index"]),
        "started_trials": len(trials),
        "complete_trials": counts["COMPLETE"],
        "pruned_trials": counts["PRUNED"],
        "failed_trials": counts["FAIL"],
        "running_trials": counts["RUNNING"],
        "best_trial_number": best.number if best is not None else None,
        "best_val_acc1": float(best.value) if best is not None else None,
        "best_lr": best.params.get("lr") if best is not None else None,
        "best_weight_decay": (
            best.params.get("weight_decay") if best is not None else None
        ),
    }
    _write_json(
        config.output_dir / "study_summaries" / f"{task.study_name}.json",
        summary,
    )
    logger.info("Finished study=%s counts=%s", task.study_name, counts)
    return summary


def _read_jsonl_files(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return pd.DataFrame.from_records(records)


def _read_json_files(paths: list[Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            records.append(json.load(file))
    return pd.DataFrame.from_records(records)


def _build_study_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()

    keys = [
        "study_name",
        "strategy_id",
        "sampler",
        "pruner",
        "sampler_seed",
        "architecture_row",
        "architecture_index",
    ]
    summaries: list[dict[str, Any]] = []
    for values, group in trials.groupby(keys, dropna=False):
        row = dict(zip(keys, values, strict=True))
        complete = group.loc[group["state"] == "COMPLETE"].copy()
        best = (
            complete.sort_values("best_val_acc1", ascending=False).iloc[0]
            if not complete.empty
            else None
        )
        group_flops = pd.to_numeric(
            group["total_trial_flops"],
            errors="coerce",
        ).fillna(0)
        group_seconds = pd.to_numeric(
            group["total_seconds"],
            errors="coerce",
        ).fillna(0)
        ordered = group.sort_values(["ended_at", "trial_number"]).copy()
        ordered["total_trial_flops"] = pd.to_numeric(
            ordered["total_trial_flops"],
            errors="coerce",
        ).fillna(0)
        ordered["cumulative_study_flops"] = ordered["total_trial_flops"].cumsum()
        first_started = pd.to_datetime(
            ordered["started_at"],
            utc=True,
            errors="coerce",
        ).min()
        last_ended = pd.to_datetime(
            ordered["ended_at"],
            utc=True,
            errors="coerce",
        ).max()
        best_finished = (
            ordered.loc[ordered["trial_number"] == best["trial_number"]].iloc[0]
            if best is not None
            else None
        )
        best_ended = (
            pd.to_datetime(best_finished["ended_at"], utc=True, errors="coerce")
            if best_finished is not None
            else None
        )

        row.update(
            {
                "started_trials": len(group),
                "complete_trials": int((group["state"] == "COMPLETE").sum()),
                "pruned_trials": int((group["state"] == "PRUNED").sum()),
                "failed_trials": int((group["state"] == "FAIL").sum()),
                "pruned_ratio": float((group["state"] == "PRUNED").mean()),
                "total_study_flops": int(group_flops.sum()),
                "sum_trial_seconds": float(group_seconds.sum()),
                "study_wall_clock_seconds": (
                    float((last_ended - first_started).total_seconds())
                    if not pd.isna(first_started) and not pd.isna(last_ended)
                    else None
                ),
                "mean_trial_flops": float(group_flops.mean()),
                "median_trial_flops": float(group_flops.median()),
                "mean_completed_epochs": float(
                    pd.to_numeric(
                        group["completed_epochs"],
                        errors="coerce",
                    ).mean()
                ),
                "best_trial_number": (
                    int(best["trial_number"]) if best is not None else None
                ),
                "best_val_acc1": (
                    float(best["best_val_acc1"]) if best is not None else None
                ),
                "best_lr": float(best["lr"]) if best is not None else None,
                "best_weight_decay": (
                    float(best["weight_decay"]) if best is not None else None
                ),
                "flops_until_best_trial": (
                    int(best_finished["cumulative_study_flops"])
                    if best_finished is not None
                    else None
                ),
                "time_until_best_trial_seconds": (
                    float((best_ended - first_started).total_seconds())
                    if best_ended is not None
                    and not pd.isna(best_ended)
                    and not pd.isna(first_started)
                    else None
                ),
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def _build_optimization_history(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()

    histories: list[pd.DataFrame] = []
    for _, group in trials.groupby("study_name", dropna=False):
        ordered = group.sort_values(["ended_at", "trial_number"]).copy()
        ordered["total_trial_flops"] = pd.to_numeric(
            ordered["total_trial_flops"],
            errors="coerce",
        ).fillna(0)
        ordered["cumulative_study_flops"] = ordered["total_trial_flops"].cumsum()
        candidate = pd.to_numeric(
            ordered["best_val_acc1"].where(ordered["state"] == "COMPLETE"),
            errors="coerce",
        )
        ordered["best_val_acc1_so_far"] = candidate.cummax()
        ordered["finished_trial_index"] = np.arange(1, len(ordered) + 1)
        first_started = pd.to_datetime(
            ordered["started_at"],
            utc=True,
            errors="coerce",
        ).min()
        ended = pd.to_datetime(ordered["ended_at"], utc=True, errors="coerce")
        ordered["wall_clock_seconds"] = (ended - first_started).dt.total_seconds()
        histories.append(ordered)
    return pd.concat(histories, ignore_index=True)


def _write_table(
    frame: pd.DataFrame,
    csv_path: Path,
    *,
    write_parquet: bool,
) -> Path | None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    if not write_parquet:
        return None
    parquet_path = csv_path.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "Parquet export skipped because pyarrow/fastparquet is unavailable"
        )
        return None
    return parquet_path


def aggregate_optuna_metrics(
    output_dir: Path,
    *,
    write_parquet: bool = True,
) -> OptunaExperimentResult:
    tables_dir = output_dir / "tables"
    epoch_metrics = _read_jsonl_files(
        sorted((output_dir / "metrics").glob("**/trial_*.jsonl"))
    )
    trial_summary = _read_json_files(
        sorted((output_dir / "trial_summaries").glob("**/trial_*.json"))
    )
    if not epoch_metrics.empty and not trial_summary.empty:
        trial_outcomes = trial_summary[
            [
                "study_name",
                "trial_number",
                "state",
                "stop_reason",
            ]
        ].drop_duplicates(["study_name", "trial_number"])
        epoch_metrics = epoch_metrics.merge(
            trial_outcomes,
            on=["study_name", "trial_number"],
            how="left",
            validate="many_to_one",
        )
    study_summary = _build_study_summary(trial_summary)
    optimization_history = _build_optimization_history(trial_summary)

    epoch_metrics_path = tables_dir / "epoch_metrics.csv"
    trial_summary_path = tables_dir / "trial_summary.csv"
    study_summary_path = tables_dir / "study_summary.csv"
    optimization_history_path = tables_dir / "optimization_history.csv"
    parquet_paths = tuple(
        path
        for path in (
            _write_table(
                epoch_metrics,
                epoch_metrics_path,
                write_parquet=write_parquet,
            ),
            _write_table(
                trial_summary,
                trial_summary_path,
                write_parquet=write_parquet,
            ),
            _write_table(
                study_summary,
                study_summary_path,
                write_parquet=write_parquet,
            ),
            _write_table(
                optimization_history,
                optimization_history_path,
                write_parquet=write_parquet,
            ),
        )
        if path is not None
    )
    return OptunaExperimentResult(
        epoch_metrics_path=epoch_metrics_path,
        trial_summary_path=trial_summary_path,
        study_summary_path=study_summary_path,
        optimization_history_path=optimization_history_path,
        parquet_paths=parquet_paths,
        failed_studies=(),
    )


def _resolve_gpu_ids(config: OptunaExperimentConfig) -> tuple[int | None, ...]:
    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; running one CPU study worker")
        return (None,)
    gpu_ids = (
        config.gpu_ids
        if config.gpu_ids is not None
        else tuple(range(torch.cuda.device_count()))
    )
    if not gpu_ids:
        raise ValueError("No GPU IDs were selected")
    invalid = [
        gpu_id
        for gpu_id in gpu_ids
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count()
    ]
    if invalid:
        raise ValueError(f"Invalid GPU IDs: {invalid}")
    return tuple(gpu_ids)


def _build_study_tasks(
    architectures: list[dict[str, Any]],
    costs: dict[int, dict[str, Any]],
    config: OptunaExperimentConfig,
) -> list[_StudyTask]:
    strategies = [
        (sampler, pruner) for pruner in config.pruners for sampler in config.samplers
    ]
    tasks: list[_StudyTask] = []
    for seed in config.sampler_seeds:
        for strategy_offset in range(len(strategies)):
            for architecture_offset, architecture in enumerate(architectures):
                sampler, pruner = strategies[
                    (architecture_offset + strategy_offset) % len(strategies)
                ]
                tasks.append(
                    _StudyTask(
                        arch_record=architecture,
                        cost=costs[int(architecture["arch_row"])],
                        sampler_name=sampler,
                        pruner_name=pruner,
                        sampler_seed=seed,
                    )
                )
    return tasks


def run_optuna_experiment(
    config: OptunaExperimentConfig,
) -> OptunaExperimentResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "experiment_config.json", asdict(config))
    _prepare_cifar10_data(config.train.data_root)

    architectures = _load_architectures(
        config.architectures_path,
        config.arch_rows,
    )
    costs = _load_costs(config.costs_path)
    missing_costs = [
        int(architecture["arch_row"])
        for architecture in architectures
        if int(architecture["arch_row"]) not in costs
    ]
    if missing_costs:
        raise KeyError(f"No FLOPs records for architecture rows: {missing_costs}")
    tasks = _build_study_tasks(architectures, costs, config)

    gpu_ids = _resolve_gpu_ids(config)
    workers_per_device = config.workers_per_gpu if gpu_ids[0] is not None else 1
    logger.info(
        "Starting %s studies on devices=%s workers_per_device=%s",
        len(tasks),
        gpu_ids,
        workers_per_device,
    )

    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(
            max_workers=workers_per_device,
            mp_context=context,
            initializer=_initialize_study_worker,
            initargs=(gpu_id, config.torch_threads_per_worker),
        )
        for gpu_id in gpu_ids
    ]
    futures: dict[Future[dict[str, Any]], _StudyTask] = {}
    failed_studies: list[str] = []
    try:
        for index, task in enumerate(tasks):
            executor = executors[index % len(executors)]
            futures[executor.submit(_run_study, task, config)] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                summary = future.result()
                logger.info(
                    "Study completed: %s best_val_acc1=%s",
                    task.study_name,
                    summary["best_val_acc1"],
                )
            except Exception:
                failed_studies.append(task.study_name)
                logger.exception("Study process failed: %s", task.study_name)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

    result = aggregate_optuna_metrics(
        config.output_dir,
        write_parquet=config.write_parquet,
    )
    return replace(result, failed_studies=tuple(sorted(failed_studies)))


__all__ = [
    "DEFAULT_PRUNERS",
    "DEFAULT_SAMPLERS",
    "OptunaExperimentConfig",
    "OptunaExperimentResult",
    "OptunaSearchSpace",
    "PlateauConfig",
    "PruningConfig",
    "StudyStopConfig",
    "aggregate_optuna_metrics",
    "run_optuna_experiment",
]
