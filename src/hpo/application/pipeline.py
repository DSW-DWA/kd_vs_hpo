import json
import multiprocessing as mp
import shutil
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.common.dataloader import build_cifar10_dataloaders
from src.hpo.domain.config import HPOExperimentConfig
from src.hpo.infrastructure.event_log import ExperimentEventLog
from src.hpo.infrastructure.worker import StudyTask, run_study_process
from src.hpo.reporting.plotting import save_plots
from src.hpo.reporting.results import (
    HPOExperimentResult,
    build_architecture_summary,
    build_study_summary,
)


def run_hpo_experiment(
    experiment: HPOExperimentConfig,
    device: torch.device,
) -> HPOExperimentResult:
    _validate(experiment)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_logs_dir = experiment.output_dir / "logs" / "runs" / run_id
    main_log_path = run_logs_dir / "main.jsonl"
    merged_log_path = run_logs_dir / "events.jsonl"
    started = time.perf_counter()
    result: HPOExperimentResult | None = None

    try:
        with ExperimentEventLog(main_log_path, {"run_id": run_id}) as event_log:
            event_log.emit(
                "experiment_started",
                device=str(device),
                config=asdict(experiment),
                output_dir=str(experiment.output_dir),
            )
            architectures = _load_architectures(
                experiment.architectures_path, experiment.arch_rows
            )
            costs = _load_costs(experiment.costs_path)
            missing_costs = [
                architecture["arch_row"]
                for architecture in architectures
                if architecture["arch_row"] not in costs
            ]
            if missing_costs:
                raise KeyError(f"Missing FLOPs costs for architecture rows: {missing_costs}")
            event_log.emit(
                "architectures_loaded",
                count=len(architectures),
                architecture_rows=[item["arch_row"] for item in architectures],
                architecture_indices=[item["arch_index"] for item in architectures],
            )

            devices = _resolve_worker_devices(experiment, device)
            evaluation_device = torch.device(devices[0])
            event_log.emit("dataloaders_started", device=str(evaluation_device))
            parent_train_loader, parent_val_loader, test_loader, n_train, n_val, n_test = (
                build_cifar10_dataloaders(experiment.train, evaluation_device)
            )
            del parent_train_loader, parent_val_loader
            event_log.emit(
                "dataloaders_completed",
                train_examples=n_train,
                validation_examples=n_val,
                test_examples=n_test,
                batch_size=experiment.train.batch_size,
                dataloader_workers=experiment.train.num_workers,
            )

            tasks = _build_tasks(
                architectures, costs, experiment, devices, run_id, run_logs_dir
            )
            event_log.emit(
                "parallel_execution_started",
                num_processes=experiment.num_processes,
                devices=devices,
                studies=len(tasks),
            )
            trial_records, epoch_records = _run_tasks(
                tasks, devices, event_log
            )
            event_log.emit(
                "parallel_execution_completed",
                trials=len(trial_records),
                epochs=len(epoch_records),
                total_hpo_flops=sum(
                    int(record["total_flops"]) for record in trial_records
                ),
            )

            trials = pd.DataFrame(trial_records)
            epochs = pd.DataFrame(epoch_records)
            studies = build_study_summary(trials)
            event_log.emit(
                "test_evaluation_started",
                architectures=len(architectures),
                device=str(evaluation_device),
            )
            test_started = time.perf_counter()
            summary = build_architecture_summary(
                trials, test_loader, evaluation_device
            )
            test_flops = sum(
                int(costs[architecture["arch_row"]]["forward_flops_per_sample"])
                * n_test
                for architecture in architectures
            )
            total_hpo_flops = int(trials["total_flops"].sum())
            if not summary.empty:
                summary["test_flops"] = summary["arch_row"].map(
                    lambda row: int(costs[int(row)]["forward_flops_per_sample"])
                    * n_test
                )
                summary["total_experiment_flops"] = (
                    summary["total_hpo_flops"] + summary["test_flops"]
                )
            event_log.emit(
                "test_evaluation_completed",
                evaluated_architectures=len(summary),
                test_seconds=time.perf_counter() - test_started,
                test_flops=test_flops,
                total_experiment_flops=total_hpo_flops + test_flops,
            )
            _save_tables(experiment.output_dir, epochs, trials, studies, summary)
            event_log.emit(
                "tables_saved",
                tables_dir=str(experiment.output_dir / "tables"),
                epoch_rows=len(epochs),
                trial_rows=len(trials),
                study_rows=len(studies),
                summary_rows=len(summary),
            )
            plot_paths = (
                save_plots(epochs, studies, experiment.output_dir)
                if experiment.generate_plots
                else ()
            )
            event_log.emit(
                "experiment_completed",
                experiment_seconds=time.perf_counter() - started,
                plot_paths=[str(path) for path in plot_paths],
                event_log_path=str(merged_log_path),
                total_hpo_flops=total_hpo_flops,
                total_test_flops=test_flops,
                total_experiment_flops=total_hpo_flops + test_flops,
            )
            result = HPOExperimentResult(
                epochs,
                trials,
                studies,
                summary,
                experiment.output_dir,
                merged_log_path,
                plot_paths,
            )
    except Exception as error:
        with ExperimentEventLog(main_log_path, {"run_id": run_id}) as event_log:
            event_log.emit(
                "experiment_failed",
                error_type=type(error).__name__,
                error_message=str(error),
                experiment_seconds=time.perf_counter() - started,
            )
        _merge_event_logs(run_logs_dir, merged_log_path, experiment.output_dir)
        raise

    _merge_event_logs(run_logs_dir, merged_log_path, experiment.output_dir)
    if result is None:
        raise RuntimeError("Experiment completed without a result")
    return result


def _run_tasks(
    tasks: list[StudyTask],
    worker_devices: tuple[str, ...],
    event_log: ExperimentEventLog,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    if len(worker_devices) == 1:
        for task in tasks:
            event_log.emit(
                "study_task_started",
                study_name=task.study_name,
                device=task.device,
            )
            trials, epochs = run_study_process(task)
            trial_records.extend(trials)
            epoch_records.extend(epochs)
            event_log.emit(
                "study_task_completed",
                study_name=task.study_name,
                device=task.device,
                trials=len(trials),
                epochs=len(epochs),
                total_flops=sum(int(record["total_flops"]) for record in trials),
            )
        return trial_records, epoch_records

    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(
            max_workers=count,
            mp_context=context,
        )
        for device, count in Counter(worker_devices).items()
    }
    try:
        futures = {
            executors[task.device].submit(run_study_process, task): task
            for task in tasks
        }
        for task in tasks:
            event_log.emit(
                "study_task_submitted",
                study_name=task.study_name,
                device=task.device,
            )
        for future in as_completed(futures):
            task = futures[future]
            trials, epochs = future.result()
            trial_records.extend(trials)
            epoch_records.extend(epochs)
            event_log.emit(
                "study_task_completed",
                study_name=task.study_name,
                device=task.device,
                trials=len(trials),
                epochs=len(epochs),
                total_flops=sum(int(record["total_flops"]) for record in trials),
            )
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
    return trial_records, epoch_records


def _build_tasks(
    architectures: list[dict[str, Any]],
    costs: dict[int, dict[str, Any]],
    experiment: HPOExperimentConfig,
    devices: tuple[str, ...],
    run_id: str,
    run_logs_dir: Path,
) -> list[StudyTask]:
    tasks = []
    task_index = 0
    for architecture in architectures:
        for sampler in experiment.optuna.samplers:
            for pruner in experiment.optuna.pruners:
                study_name = f"arch_{architecture['arch_index']}__{sampler}_{pruner}"
                tasks.append(
                    StudyTask(
                        architecture=architecture,
                        forward_flops_per_sample=int(
                            float(
                                costs[architecture["arch_row"]][
                                    "forward_flops_per_sample"
                                ]
                            )
                        ),
                        sampler=sampler,
                        pruner=pruner,
                        experiment=experiment,
                        device=devices[task_index % len(devices)],
                        run_id=run_id,
                        log_path=run_logs_dir / "studies" / f"{study_name}.jsonl",
                    )
                )
                task_index += 1
    return tasks


def _resolve_worker_devices(
    experiment: HPOExperimentConfig,
    requested_device: torch.device,
) -> tuple[str, ...]:
    if requested_device.type != "cuda":
        return tuple("cpu" for _ in range(experiment.num_processes))
    gpu_ids = experiment.gpu_ids
    if gpu_ids is None:
        gpu_ids = (
            (requested_device.index,)
            if requested_device.index is not None
            else tuple(range(torch.cuda.device_count()))
        )
    if not gpu_ids:
        raise ValueError("No CUDA devices are available")
    invalid = [gpu_id for gpu_id in gpu_ids if not 0 <= gpu_id < torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Invalid GPU IDs: {invalid}")
    return tuple(
        f"cuda:{gpu_ids[index % len(gpu_ids)]}"
        for index in range(experiment.num_processes)
    )


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
    return selected


def _load_costs(path: Path) -> dict[int, dict[str, Any]]:
    costs = pd.read_csv(path)
    required = {"arch_row", "forward_flops_per_sample"}
    missing = required - set(costs.columns)
    if missing:
        raise ValueError(f"Costs CSV is missing columns: {sorted(missing)}")
    return {
        int(row): values
        for row, values in costs.set_index("arch_row").to_dict(orient="index").items()
    }


def _validate(experiment: HPOExperimentConfig) -> None:
    early = experiment.early_stopping
    optuna_config = experiment.optuna
    if experiment.num_processes < 1:
        raise ValueError("num_processes must be positive")
    if early.min_growth < 0:
        raise ValueError("early_stopping.min_growth must be non-negative")
    if early.patience < 1 or early.warmup_epochs < early.patience:
        raise ValueError("warmup_epochs must be at least patience")
    if optuna_config.n_trials < 1 or optuna_config.max_epochs < 1:
        raise ValueError("n_trials and max_epochs must be positive")
    if not optuna_config.samplers or not optuna_config.pruners:
        raise ValueError("At least one sampler and pruner must be selected")
    unsupported_samplers = set(optuna_config.samplers) - {
        "tpe", "grid", "cmaes", "gp"
    }
    unsupported_pruners = set(optuna_config.pruners) - {
        "successive_halving", "hyperband"
    }
    if unsupported_samplers:
        raise ValueError(f"Unsupported samplers: {sorted(unsupported_samplers)}")
    if unsupported_pruners:
        raise ValueError(f"Unsupported pruners: {sorted(unsupported_pruners)}")
    search = experiment.search_space
    if not search.grid_lr or not search.grid_weight_decay:
        raise ValueError("Grid search values cannot be empty")
    if any(value < search.lr[0] or value > search.lr[1] for value in search.grid_lr):
        raise ValueError("grid_lr values must be within lr bounds")
    if any(
        value < search.weight_decay[0] or value > search.weight_decay[1]
        for value in search.grid_weight_decay
    ):
        raise ValueError("grid_weight_decay values must be within weight_decay bounds")


def _save_tables(output_dir: Path, *tables: pd.DataFrame) -> None:
    names = ("epoch_metrics", "trials", "studies", "architecture_summary")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in zip(names, tables, strict=True):
        table.to_csv(tables_dir / f"{name}.csv", index=False)


def _merge_event_logs(
    run_logs_dir: Path,
    merged_path: Path,
    output_dir: Path,
) -> None:
    records = []
    for path in run_logs_dir.rglob("*.jsonl"):
        if path == merged_path:
            continue
        with path.open("r", encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file if line.strip())
    records.sort(key=lambda record: record["timestamp"])
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    latest_path = output_dir / "logs" / "events.jsonl"
    shutil.copyfile(merged_path, latest_path)
