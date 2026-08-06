import json
import shutil
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from src.common.dataloader import build_cifar10_datasets, build_dataloader
from src.common.flops import count_flops_params
from src.common.nats import create_nats_model
from src.hpo.config import HPOExperimentConfig, validate_experiment
from src.hpo.persistence import save_result_tables, save_run_config
from src.hpo.results import HPOExperimentResult, build_experiment_result
from src.hpo.worker import (
    build_study_tasks,
    resolve_worker_devices,
    run_study_tasks,
)


def run_hpo_experiment(
    experiment: HPOExperimentConfig,
    device: torch.device,
) -> HPOExperimentResult:
    validate_experiment(experiment)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir = experiment.output_dir / "recovery"
    shutil.rmtree(recovery_dir, ignore_errors=True)

    architectures = _load_architectures(
        experiment.architectures_path,
        experiment.arch_rows,
    )
    costs = _measure_architecture_costs(architectures)
    save_run_config(
        experiment.output_dir,
        {
            "device": str(device),
            "calflops_version": version("calflops"),
            "experiment": asdict(experiment),
            "architecture_costs": list(costs.values()),
        },
    )

    devices = resolve_worker_devices(experiment, device)
    evaluation_device = torch.device(devices[0])
    train_dataset, val_dataset, test_dataset = build_cifar10_datasets(experiment.train)
    test_loader = build_dataloader(
        test_dataset,
        experiment.train,
        evaluation_device,
        shuffle=False,
        seed=experiment.train.seed + 2,
    )
    n_test = len(test_dataset)
    del train_dataset, val_dataset

    tasks = build_study_tasks(
        architectures,
        costs,
        experiment,
        devices,
        recovery_dir,
    )
    trial_records, epoch_records = run_study_tasks(tasks, devices)
    result = build_experiment_result(
        trial_records=trial_records,
        epoch_records=epoch_records,
        costs=costs,
        n_test=n_test,
        test_loader=test_loader,
        evaluation_device=evaluation_device,
        output_dir=experiment.output_dir,
    )
    save_result_tables(
        experiment.output_dir,
        result.epochs,
        result.trials,
        result.studies,
        result.summary,
    )
    shutil.rmtree(recovery_dir, ignore_errors=True)
    return result


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
        forward_flops, params = count_flops_params(model, device="cpu")
        arch_row = int(architecture["arch_row"])
        costs[arch_row] = {
            "arch_row": arch_row,
            "arch_index": int(architecture["arch_index"]),
            "forward_flops_per_sample": forward_flops,
            "params": params,
        }
    return costs
