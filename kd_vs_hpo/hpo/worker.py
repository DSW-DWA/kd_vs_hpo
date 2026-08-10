import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kd_vs_hpo.hpo.config import HPOExperimentConfig, PrunerName, SamplerName
from kd_vs_hpo.hpo.data import build_cifar10_datasets
from kd_vs_hpo.hpo.optimization import run_study


@dataclass(frozen=True)
class StudyTask:
    architecture: dict[str, Any]
    forward_flops_per_sample: int
    sampler: SamplerName
    pruner: PrunerName
    experiment: HPOExperimentConfig
    device: str
    progress_dir: Path

    @property
    def study_name(self) -> str:
        return f"arch_{self.architecture['arch_index']}__{self.sampler}_{self.pruner}"


def run_study_process(
    task: StudyTask,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device(task.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)

    train_dataset, val_dataset, _ = build_cifar10_datasets(
        task.experiment.train,
        device,
    )
    return run_study(
        architecture=task.architecture,
        sampler_name=task.sampler,
        pruner_name=task.pruner,
        experiment=task.experiment,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        forward_flops_per_sample=task.forward_flops_per_sample,
        n_train=len(train_dataset),
        n_val=len(val_dataset),
        progress_dir=task.progress_dir,
    )


def resolve_worker_devices(
    experiment: HPOExperimentConfig,
    requested_device: torch.device,
) -> tuple[str, ...]:
    if requested_device.type == "mps":
        return tuple("mps" for _ in range(experiment.num_processes))
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

    invalid = [
        gpu_id for gpu_id in gpu_ids if not 0 <= gpu_id < torch.cuda.device_count()
    ]
    if invalid:
        raise ValueError(f"Invalid GPU IDs: {invalid}")
    return tuple(
        f"cuda:{gpu_ids[index % len(gpu_ids)]}"
        for index in range(experiment.num_processes)
    )


def build_study_tasks(
    architectures: list[dict[str, Any]],
    costs: dict[int, dict[str, Any]],
    experiment: HPOExperimentConfig,
    devices: tuple[str, ...],
    recovery_dir: Path,
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
                        progress_dir=recovery_dir / study_name,
                    )
                )
                task_index += 1
    return tasks


def run_study_tasks(
    tasks: list[StudyTask],
    worker_devices: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    if len(worker_devices) == 1:
        for task in tasks:
            trials, epochs = run_study_process(task)
            trial_records.extend(trials)
            epoch_records.extend(epochs)
        return trial_records, epoch_records

    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(max_workers=count, mp_context=context)
        for device, count in Counter(worker_devices).items()
    }
    try:
        futures = [
            executors[task.device].submit(run_study_process, task) for task in tasks
        ]
        for future in as_completed(futures):
            trials, epochs = future.result()
            trial_records.extend(trials)
            epoch_records.extend(epochs)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
    return trial_records, epoch_records
