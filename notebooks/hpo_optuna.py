import argparse
import logging
from pathlib import Path

import torch

from src.common import TrainConfig
from src.hpo import (
    EarlyStoppingConfig,
    HPOExperimentConfig,
    OptunaConfig,
    SearchSpace,
    run_hpo_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO experiment")
    parser.add_argument("--arch-rows", type=int, nargs="*", default=[0])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lambda-growth", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("hpo_output"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    experiment = HPOExperimentConfig(
        train=TrainConfig(
            seed=42,
            deterministic=False,
            amp=True,
            data_root=PROJECT_ROOT / "data",
        ),
        search_space=SearchSpace(),
        early_stopping=EarlyStoppingConfig(
            min_growth=args.lambda_growth,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
        ),
        optuna=OptunaConfig(n_trials=args.n_trials, max_epochs=args.max_epochs),
        architectures_path=PROJECT_ROOT / "experiments/nats_architectures_10.json",
        costs_path=PROJECT_ROOT / "experiments/sampled_architecture_costs.csv",
        output_dir=output_dir,
        arch_rows=tuple(args.arch_rows) if args.arch_rows else None,
        num_processes=args.processes,
        gpu_ids=tuple(args.gpu_ids) if args.gpu_ids else None,
    )
    result = run_hpo_experiment(experiment, device)
    print(f"Structured event log: {result.event_log_path}")
    print(result.studies.to_string(index=False))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
