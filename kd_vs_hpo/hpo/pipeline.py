import json
import shutil
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.flops import CounterMode, FlopsBudgetTracker, count_flops_params
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.hpo.config import HPOExperimentConfig, validate_experiment
from kd_vs_hpo.hpo.persistence import save_result_tables, save_run_config
from kd_vs_hpo.hpo.results import HPOExperimentResult, build_experiment_result
from kd_vs_hpo.hpo.worker import (
    build_study_tasks,
    resolve_worker_devices,
    run_study_tasks,
)


def run_hpo_experiment(
    experiment: HPOExperimentConfig,
) -> HPOExperimentResult:
    validate_experiment(experiment)
    device = _resolve_device(experiment.device)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir = experiment.output_dir / "recovery"
    shutil.rmtree(recovery_dir, ignore_errors=True)

    architectures = _load_architectures(
        experiment.architectures_path,
        experiment.arch_rows,
    )
    costs = _measure_architecture_costs(architectures)
    devices = resolve_worker_devices(experiment, device)
    evaluation_device = torch.device(devices[0])
    train_loader, val_loader, test_loader, n_train, n_val, n_test = (
        build_cifar10_dataloaders(
            experiment.train.checkpoint_dir,
            experiment.train.log_dir,
            experiment.train.data_root,
            experiment.train.seed,
            experiment.train.batch_size,
            experiment.train.num_workers,
            experiment.train.validation_fraction,
            evaluation_device,
        )
    )
    flops_tracker = FlopsBudgetTracker(
        budget=_maximum_experiment_flops(
            experiment,
            costs,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        ),
        mode=CounterMode.SILENT,
    )
    run_config = {
        "device": str(device),
        "calflops_version": version("calflops"),
        "experiment": asdict(experiment),
        "architecture_costs": list(costs.values()),
        "flops_tracker": _flops_tracker_payload(flops_tracker),
    }
    save_run_config(experiment.output_dir, run_config)

    tasks = build_study_tasks(
        architectures,
        costs,
        experiment,
        devices,
        recovery_dir,
    )
    local_loaders = (
        (train_loader, val_loader, n_train, n_val) if len(devices) == 1 else None
    )
    if local_loaders is None:
        del train_loader, val_loader
    trial_records, epoch_records = run_study_tasks(
        tasks,
        devices,
        local_loaders=local_loaders,
    )
    for trial_record in trial_records:
        flops_tracker.spend(int(trial_record["total_flops"]))
    result = build_experiment_result(
        trial_records=trial_records,
        epoch_records=epoch_records,
        costs=costs,
        n_test=n_test,
        test_loader=test_loader,
        evaluation_device=evaluation_device,
        output_dir=experiment.output_dir,
        experiment=experiment,
    )
    if not result.summary.empty:
        flops_tracker.spend(int(result.summary["test_flops"].sum()))
    run_config["flops_tracker"] = _flops_tracker_payload(flops_tracker)
    save_run_config(experiment.output_dir, run_config)
    save_result_tables(
        experiment.output_dir,
        result.epochs,
        result.trials,
        result.studies,
        result.summary,
    )
    shutil.rmtree(recovery_dir, ignore_errors=True)
    return result


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device(requested)


def _maximum_experiment_flops(
    experiment: HPOExperimentConfig,
    costs: dict[int, dict[str, int]],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
) -> int:
    studies_per_architecture = len(experiment.optuna.samplers) * len(
        experiment.optuna.pruners
    )
    maximum = 0
    for cost in costs.values():
        forward_flops = int(cost["forward_flops_per_sample"])
        train_epoch_flops = int(
            experiment.train.train_step_multiplier * forward_flops * n_train
        )
        validation_epoch_flops = forward_flops * n_val
        maximum += (
            studies_per_architecture
            * experiment.optuna.n_trials
            * experiment.optuna.max_epochs
            * (train_epoch_flops + validation_epoch_flops)
        )
        maximum += forward_flops * n_test
    return maximum


def _flops_tracker_payload(tracker: FlopsBudgetTracker) -> dict[str, int | str]:
    return {
        "mode": tracker.mode.value,
        "budget": tracker.budget,
        "spent": tracker.spent,
        "remaining": tracker.remaining(),
    }


def _load_architectures(
    path: Path, rows: tuple[int, ...] | None
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    selected = []
    for row, source in enumerate(records):
        if rows is not None and row not in rows:
            continue
        record = dict(source)
        record.update(arch_row=row, arch_index=int(record["arch_index"]))
        selected.append(record)
    if not selected:
        raise ValueError("No architectures selected")
    duplicate_indices = sorted(
        arch_index
        for arch_index, count in Counter(
            architecture["arch_index"] for architecture in selected
        ).items()
        if count > 1
    )
    if duplicate_indices:
        raise ValueError(f"Architecture indices must be unique: {duplicate_indices}")
    return selected


def _measure_architecture_costs(
    architectures: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    costs = {}
    for architecture in architectures:
        model = create_nats_model(architecture)
        forward_flops, params = count_flops_params(model)
        arch_row = int(architecture["arch_row"])
        costs[arch_row] = {
            "arch_row": arch_row,
            "arch_index": int(architecture["arch_index"]),
            "forward_flops_per_sample": forward_flops,
            "params": params,
        }
    return costs
