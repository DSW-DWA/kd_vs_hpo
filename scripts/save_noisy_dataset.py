"""Generate and save the noisy target map used during training.

The saved ``.pt`` file contains the original/noisy targets, the train/val
split, and the metadata needed to reproduce it.

Example:
    uv run python scripts/save_noisy_dataset.py \
        --dataset cifar10 --noise 0.2 --output data/cifar10_noise20.pt
"""

import argparse
from pathlib import Path

import torch

from kd_vs_hpo.common.dataloader import (
    DATASETS,
    _build_noisy_targets,
    get_num_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--noise", required=True, type=float, help="label noise fraction in [0, 1]"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="path to save the .pt file"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--validation-fraction", default=0.1, type=float)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.noise <= 1.0:
        raise ValueError(f"--noise must be in [0, 1], got {args.noise}")

    dataset_spec = DATASETS[args.dataset]
    dataset_cls = dataset_spec.dataset_cls
    num_classes = get_num_classes(args.dataset)

    train_dataset = dataset_cls(root=args.data_root, train=True, download=True)
    targets = torch.as_tensor(train_dataset.targets, dtype=torch.long)

    split_gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(targets), generator=split_gen).tolist()
    n_val = round(len(indices) * args.validation_fraction)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    noisy_targets = _build_noisy_targets(
        train_dataset,
        num_classes,
        args.noise,
        args.seed,
        groups=(train_indices, val_indices),
    )
    if noisy_targets is None:
        noisy_targets = targets.clone()

    changed = (noisy_targets != targets).nonzero().flatten()

    payload = {
        "dataset": args.dataset,
        "noise_fraction": args.noise,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "num_classes": num_classes,
        "targets": targets,
        "noisy_targets": noisy_targets,
        "changed_indices": changed,
        "train_indices": torch.as_tensor(train_indices, dtype=torch.long),
        "val_indices": torch.as_tensor(val_indices, dtype=torch.long),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(
        f"Saved {args.dataset} noisy dataset (noise={args.noise}, seed={args.seed}) "
        f"to {args.output}: {changed.numel()} / {len(targets)} labels changed"
    )


if __name__ == "__main__":
    main()
