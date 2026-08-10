"""Adapters from the current shared CIFAR-10 API to the HPO pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders


def build_cifar10_datasets(
    config: TrainConfig,
    device: torch.device,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build the shared split and expose its datasets to per-trial loaders."""
    train_loader, val_loader, test_loader, *_ = build_cifar10_dataloaders(
        config.checkpoint_dir,
        config.log_dir,
        config.data_root,
        config.seed,
        config.batch_size,
        config.num_workers,
        config.validation_fraction,
        device,
    )
    return train_loader.dataset, val_loader.dataset, test_loader.dataset


def build_dataloader(
    dataset: Dataset,
    config: TrainConfig,
    device: torch.device,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a reproducibly seeded loader for one HPO trial."""
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
