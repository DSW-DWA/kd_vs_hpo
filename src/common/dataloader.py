from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from src.common.config import TrainConfig


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


def build_cifar10_datasets(
    cfg: TrainConfig,
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
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

    return (
        Subset(train_aug_dataset, train_indices),
        Subset(train_eval_dataset, val_indices),
        test_dataset,
    )


def build_dataloader(
    dataset: Dataset[Any],
    cfg: TrainConfig,
    device: torch.device,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )


def build_cifar10_dataloaders(cfg: TrainConfig, device: torch.device) -> Any:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, test_dataset = build_cifar10_datasets(cfg)
    train_loader = build_dataloader(
        train_dataset,
        cfg,
        device,
        shuffle=True,
        seed=cfg.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        cfg,
        device,
        shuffle=False,
        seed=cfg.seed + 1,
    )
    test_loader = build_dataloader(
        test_dataset,
        cfg,
        device,
        shuffle=False,
        seed=cfg.seed + 2,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
