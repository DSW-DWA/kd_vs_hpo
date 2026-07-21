import argparse
import logging
from pathlib import Path

import torch

from src.common import TrainConfig
from src.hpo import (
    DEFAULT_PRUNERS,
    HPOExperimentConfig,
    OptunaConfig,
    PRUNER_NAMES,
    SAMPLER_NAMES,
    SearchSpace,
    run_hpo_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO experiment")
    parser.add_argument("--arch-rows", type=int, nargs="*", default=[0])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("hpo_output"))
    parser.add_argument(
        "--samplers",
        choices=SAMPLER_NAMES,
        nargs="+",
        default=SAMPLER_NAMES,
    )
    parser.add_argument(
        "--pruners",
        choices=PRUNER_NAMES,
        nargs="+",
        default=DEFAULT_PRUNERS,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. MPS requires a compatible Mac and PyTorch build."
        )

    return torch.device(requested_device)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = resolve_device(args.device)
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    experiment = HPOExperimentConfig(
        train=TrainConfig(
            seed=42,
            deterministic=False,
            amp=True,
            data_root=PROJECT_ROOT / "data",
        ),
        search_space=SearchSpace(),
        optuna=OptunaConfig(
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            samplers=tuple(args.samplers),
            pruners=tuple(args.pruners),
        ),
        architectures_path=PROJECT_ROOT / "experiments/nats_architectures_10.json",
        output_dir=output_dir,
        arch_rows=tuple(args.arch_rows) if args.arch_rows else None,
        num_processes=args.processes,
        gpu_ids=tuple(args.gpu_ids) if args.gpu_ids else None,
    )
    result = run_hpo_experiment(experiment, device)
    print(result.studies.to_string(index=False))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
