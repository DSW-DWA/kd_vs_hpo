from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from kd_vs_hpo.common.config import TrainConfig


normalize_kwargs = {
    "mean": [0.49139968, 0.48215841, 0.44653091],
    "std": [0.24703223, 0.24348513, 0.26158784],
}

train_transform = T.Compose(
    [
        T.RandomCrop(size=32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(**normalize_kwargs),
    ]
)

eval_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(**normalize_kwargs),
    ]
)


def build_cifar10_dataloaders(
        checkpoint_dir: Path,
        log_dir: Path,
        data_root: Path,
        seed: int,
        batch_size: int,
        num_workers: int,
        validation_fraction: float,
        device: torch.device
        ) -> tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_aug_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    train_eval_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    split_gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_aug_dataset), generator=split_gen).tolist()

    n_val = int(round(len(indices) * validation_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(train_aug_dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(train_eval_dataset, val_indices),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(train_indices),
        len(val_indices),
        len(test_dataset),
    )
