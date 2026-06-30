import logging
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.common.dataloader import build_cifar10_dataloaders
from src.hpo.asha import ASHAPlan, TrialConfig
from src.hpo.config import HPOExperimentConfig
from src.hpo.training import run_training_stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingStageTask:
    trial: TrialConfig
    target_epochs: int
    plan: ASHAPlan
    seed: int
    arch_record: dict[str, Any]
    experiment: HPOExperimentConfig


_train_loader: DataLoader | None = None
_val_loader: DataLoader | None = None
_device: torch.device | None = None


def _initialize_worker(experiment: HPOExperimentConfig, gpu_id: int) -> None:
    global _device, _train_loader, _val_loader
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.cuda.set_device(gpu_id)
    _device = torch.device(f"cuda:{gpu_id}")
    _train_loader, _val_loader, _, *_ = build_cifar10_dataloaders(
        experiment.train, _device
    )
    logger.info("GPU worker ready: pid=%s device=%s", mp.current_process().pid, _device)


def _run_task(task: TrainingStageTask) -> dict[str, Any]:
    if _device is None or _train_loader is None or _val_loader is None:
        raise RuntimeError("GPU worker was not initialized")
    return run_training_stage(
        trial=task.trial,
        target_epochs=task.target_epochs,
        plan=task.plan,
        seed=task.seed,
        arch_record=task.arch_record,
        experiment=task.experiment,
        train_loader=_train_loader,
        val_loader=_val_loader,
        device=_device,
    )


class ParallelStageRunner:
    def __init__(self, experiment: HPOExperimentConfig, gpu_ids: tuple[int, ...]) -> None:
        context = mp.get_context("spawn")
        self._executors = [
            ProcessPoolExecutor(
                max_workers=experiment.workers_per_gpu,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(experiment, gpu_id),
            )
            for gpu_id in gpu_ids
        ]
        self._next_executor = 0

    def submit(self, task: TrainingStageTask) -> Future[dict[str, Any]]:
        executor = self._executors[self._next_executor]
        self._next_executor = (self._next_executor + 1) % len(self._executors)
        return executor.submit(_run_task, task)

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
