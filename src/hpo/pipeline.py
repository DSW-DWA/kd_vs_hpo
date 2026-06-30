import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.common.dataloader import build_cifar10_dataloaders
from src.hpo.config import HPOExperimentConfig
from src.hpo.plotting import save_hpo_plots
from src.hpo.results import HPOExperimentResult, build_summary
from src.hpo.search import run_architecture
from src.hpo.workers import ParallelStageRunner

logger = logging.getLogger(__name__)


def _load_architectures(path: Path, rows: tuple[int, ...] | None) -> list[dict[str, Any]]:
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


def _gpu_ids(experiment: HPOExperimentConfig) -> tuple[int, ...]:
    gpu_ids = experiment.gpu_ids or tuple(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise ValueError("No GPU IDs were selected for CUDA execution")
    invalid = [gpu_id for gpu_id in gpu_ids if not 0 <= gpu_id < torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Invalid GPU IDs: {invalid}")
    return gpu_ids


def run_hpo_experiment(
    experiment: HPOExperimentConfig,
    device: torch.device,
) -> HPOExperimentResult:
    if experiment.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1")
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    (experiment.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    architectures = _load_architectures(experiment.architectures_path, experiment.arch_rows)
    costs = _load_costs(experiment.costs_path)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = (
        build_cifar10_dataloaders(experiment.train, device)
    )
    logger.info(
        "Loaded %s architectures; dataset sizes: train=%s validation=%s test=%s",
        len(architectures), n_train, n_val, n_test,
    )

    runner = None
    evaluation_device = device
    if device.type == "cuda":
        gpu_ids = _gpu_ids(experiment)
        runner = ParallelStageRunner(experiment, gpu_ids)
        evaluation_device = torch.device(f"cuda:{gpu_ids[0]}")

    try:
        results = []
        for architecture in architectures:
            arch_row = architecture["arch_row"]
            if arch_row not in costs:
                raise KeyError(f"No FLOPs record for architecture row {arch_row}")
            results.append(
                run_architecture(
                    arch_record=architecture,
                    cost=costs[arch_row],
                    experiment=experiment,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_train=n_train,
                    n_val=n_val,
                    device=device,
                    parallel_runner=runner,
                )
            )
    finally:
        if runner:
            runner.close()

    stages = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    summary = build_summary(stages, test_loader, n_test, evaluation_device)
    stages_path = experiment.output_dir / "hpo_results.csv"
    summary_path = experiment.output_dir / "hpo_summary.csv"
    stages.to_csv(stages_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_paths = (
        save_hpo_plots(stages, summary, experiment.output_dir)
        if experiment.generate_plots else ()
    )
    return HPOExperimentResult(stages, summary, stages_path, summary_path, plot_paths)
