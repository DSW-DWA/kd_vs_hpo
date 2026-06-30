import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.common.dataloader import build_cifar10_dataloaders
from src.hpo.application.optimization import run_study
from src.hpo.domain.config import HPOExperimentConfig, PrunerName, SamplerName
from src.hpo.infrastructure.event_log import ExperimentEventLog


@dataclass(frozen=True)
class StudyTask:
    architecture: dict[str, Any]
    forward_flops_per_sample: int
    sampler: SamplerName
    pruner: PrunerName
    experiment: HPOExperimentConfig
    device: str
    run_id: str
    log_path: Path

    @property
    def study_name(self) -> str:
        return (
            f"arch_{self.architecture['arch_index']}__"
            f"{self.sampler}_{self.pruner}"
        )


def run_study_process(
    task: StudyTask,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    device = torch.device(task.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)

    context = {
        "run_id": task.run_id,
        "worker_pid": os.getpid(),
        "worker_device": str(device),
    }
    with ExperimentEventLog(task.log_path, context) as event_log:
        event_log.emit(
            "worker_started",
            study_name=task.study_name,
            arch_index=task.architecture["arch_index"],
            sampler=task.sampler,
            pruner=task.pruner,
        )
        try:
            event_log.emit("worker_dataloaders_started", study_name=task.study_name)
            train_loader, val_loader, _, n_train, n_val, _ = (
                build_cifar10_dataloaders(task.experiment.train, device)
            )
            event_log.emit(
                "worker_dataloaders_completed",
                study_name=task.study_name,
                train_examples=n_train,
                validation_examples=n_val,
            )
            result = run_study(
                architecture=task.architecture,
                sampler_name=task.sampler,
                pruner_name=task.pruner,
                experiment=task.experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                event_log=event_log,
                forward_flops_per_sample=task.forward_flops_per_sample,
                n_train=n_train,
                n_val=n_val,
            )
            event_log.emit(
                "worker_completed",
                study_name=task.study_name,
                worker_seconds=time.perf_counter() - started,
            )
            return result
        except Exception as error:
            event_log.emit(
                "worker_failed",
                study_name=task.study_name,
                error_type=type(error).__name__,
                error_message=str(error),
                worker_seconds=time.perf_counter() - started,
            )
            raise
