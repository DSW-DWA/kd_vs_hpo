from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

DATASETS = {
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
}


class _LabelNoiseDataset(Dataset):
    """Wrap a dataset and override targets with a precomputed noisy target map.

    ``noisy_targets`` is an array aligned with the wrapped dataset's indices
    (``None`` means leave labels untouched). The map is built once and shared
    by all train/split datasets so the same sample always gets the same label.
    """

    def __init__(self, dataset: Dataset, noisy_targets: torch.Tensor | None):
        self.dataset = dataset
        self._noisy_targets = noisy_targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        if self._noisy_targets is not None:
            target = int(self._noisy_targets[index])
        return image, target


def _build_noisy_targets(
    dataset: Dataset,
    num_classes: int,
    noise_fraction: float,
    seed: int,
    *,
    groups: tuple[list[int], ...] | None = None,
) -> torch.Tensor | None:
    """Build a deterministic noisy target map.
    """
    if noise_fraction <= 0:
        return None

    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    noisy_targets = targets.clone()

    if groups is None:
        groups = (list(range(len(targets))),)
        
    gen = torch.Generator().manual_seed(seed + 1)
    total_noisy = 0

    for group in groups:
        group_indices = torch.as_tensor(group, dtype=torch.long)

        if group_indices.numel() == 0:
            continue

        n_noisy = int(noise_fraction * group_indices.numel() + 0.5)

        if n_noisy == 0:
            continue

        perm = torch.randperm(group_indices.numel(), generator=gen)
        noisy_idx = group_indices[perm[:n_noisy]]

        original = targets[noisy_idx]
        replacement = torch.randint(
            0,
            num_classes - 1,
            (n_noisy,),
            dtype=torch.long,
            generator=gen,
        )
        replacement += (replacement >= original).long()

        noisy_targets[noisy_idx] = replacement
        total_noisy += n_noisy

    if total_noisy == 0:
        return None

    return noisy_targets


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


def get_num_classes(dataset_name: str) -> int:
    try:
        return DATASETS[dataset_name][1]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset: {dataset_name!r}. "
            f"Supported datasets: {sorted(DATASETS)}"
        )


def build_dataloaders(
    dataset_name: str,
    checkpoint_dir: Path,
    log_dir: Path,
    data_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    validation_fraction: float,
    device: torch.device,
    *,
    label_noise: float = 0.0,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Build train/val/test loaders for a named dataset.

    The train split is randomly partitioned into train and validation subsets;
    both train and val yield ``(image, label)`` pairs, the test split is the
    dataset's own test set. Returns the three loaders plus their sizes.

    When ``label_noise`` is nonzero, that fraction of train and validation
    labels is randomly switched to another class (symmetric noise, seeded),
    while the test labels are left untouched.
    """

    dataset_cls, _ = DATASETS[dataset_name]
    num_classes = get_num_classes(dataset_name)

    if not 0.0 <= label_noise <= 1.0:
        raise ValueError(f"label_noise must be in [0, 1], got {label_noise}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = dataset_cls(
        root=data_root, train=True, download=True, transform=train_transform
    )
    train_eval_dataset = dataset_cls(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    test_dataset = dataset_cls(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    split_gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=split_gen).tolist()

    n_val = round(len(indices) * validation_fraction)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    if label_noise > 0:
        noisy_targets = _build_noisy_targets(
            train_dataset,
            num_classes,
            label_noise,
            seed,
            groups=(train_indices, val_indices),
        )

        if noisy_targets is not None:
            train_dataset = _LabelNoiseDataset(
                train_dataset, noisy_targets
            )
            train_eval_dataset = _LabelNoiseDataset(
                train_eval_dataset, noisy_targets
            )


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
        Subset(train_dataset, train_indices),
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


def build_cifar10_dataloaders(
    checkpoint_dir: Path,
    log_dir: Path,
    data_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    validation_fraction: float,
    device: torch.device,
    *,
    label_noise: float = 0.0,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int, int]:

    return build_dataloaders(
        "cifar10",
        checkpoint_dir,
        log_dir,
        data_root,
        seed,
        batch_size,
        num_workers,
        validation_fraction,
        device,
        label_noise=label_noise,
    )
