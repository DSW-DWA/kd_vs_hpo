from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import torch

from src.common import TrainConfig
from src.plain_training import (
    INITIAL_LR,
    INITIAL_WEIGHT_DECAY,
    TRIAL_EPOCHS,
    load_architectures_by_index,
    plain_train_config_payload,
    run_plain_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train each NATS architecture in two independent 100- and 200-epoch "
            "trials without Optuna."
        )
    )
    parser.add_argument(
        "--arch-indices",
        type=int,
        nargs="+",
        default=[3358],
    )
    parser.add_argument("--lr", type=float, default=INITIAL_LR)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=INITIAL_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--architectures-path",
        type=Path,
        default=Path("experiments/nats_architectures_10.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plain_training_output"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without loading data or training.",
    )
    return parser.parse_args(argv)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    if not args.arch_indices:
        raise ValueError("--arch-indices must contain at least one value")
    if len(set(args.arch_indices)) != len(args.arch_indices):
        raise ValueError("--arch-indices values must be unique")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    architecture_path = project_path(args.architectures_path)
    architectures = load_architectures_by_index(
        architecture_path,
        tuple(args.arch_indices),
    )
    device = resolve_device(args.device)
    output_dir = project_path(args.output_dir)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=args.amp,
        data_root=project_path(args.data_root),
        checkpoint_dir=output_dir / "checkpoints",
        log_dir=output_dir / "runs",
    )
    resolved = {
        "experiment": "plain_training_with_kd_protocol",
        "architectures": architectures,
        "trials": [
            {
                "trial_id": trial_id,
                "target_epochs": target_epochs,
                "trial_seed": args.seed + trial_id,
            }
            for trial_id, target_epochs in enumerate(TRIAL_EPOCHS)
        ],
        "initial_lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "output_dir": str(output_dir),
        "train": plain_train_config_payload(train_config),
        "independent_runs": True,
        "uses_distillation": False,
        "uses_optuna": False,
        "uses_pruning": False,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False, default=str))
        return

    result = run_plain_experiment(
        architectures=architectures,
        initial_lr=args.lr,
        weight_decay=args.weight_decay,
        train_config=train_config,
        device=device,
        output_dir=output_dir,
        verbose=args.verbose,
    )
    print(result.runs.to_string(index=False))


if __name__ == "__main__":
    main()
