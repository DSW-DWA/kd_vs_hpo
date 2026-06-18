import argparse
import logging
import time
from pathlib import Path

import torch

from kd_vs_hpo.common import TrainConfig
from kd_vs_hpo.hpo import (
    ASHAConfig,
    HPOExperimentConfig,
    SearchSpace,
    run_hpo_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLOPs-budgeted HPO for NATS-Bench architectures."
    )
    parser.add_argument(
        "--arch-rows",
        type=int,
        nargs="*",
        default=[0],
        help="Architecture rows to run. Pass no values to run all rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo_output"),
        help="Directory for checkpoints, CSV files and plots.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help="Visible GPU IDs to use. By default, use all visible GPUs.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=4,
        help="Number of models trained concurrently on each GPU.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return torch.device(name)


def build_experiment(args: argparse.Namespace) -> HPOExperimentConfig:
    arch_rows = tuple(args.arch_rows) if args.arch_rows else None
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    return HPOExperimentConfig(
        train=TrainConfig(
            batch_size=256,
            num_workers=2,
            validation_fraction=0.1,
            momentum=0.9,
            grad_clip_norm=5.0,
            seed=42,
            deterministic=False,
            amp=True,
            train_step_multiplier=3.0,
            data_root=PROJECT_ROOT / "data",
        ),
        search_space=SearchSpace(
            lr=(1e-3, 3e-1),
            weight_decay=(1e-6, 1e-3),
        ),
        asha=ASHAConfig(
            budget_flops_per_arch=10**15,
            target_min_epochs=3,
            reduction_factor=3,
            max_initial_configs=12,
            max_epochs=81,
        ),
        architectures_path=PROJECT_ROOT / "experiments/nats_architectures_10.json",
        costs_path=PROJECT_ROOT / "experiments/sampled_architecture_costs.csv",
        output_dir=output_dir,
        arch_rows=arch_rows,
        gpu_ids=tuple(args.gpu_ids) if args.gpu_ids is not None else None,
        workers_per_gpu=args.workers_per_gpu,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("hpo_flops_budget")
    device = select_device(args.device)
    experiment = build_experiment(args)

    logger.info("Starting HPO experiment")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("Visible GPUs: %s", torch.cuda.device_count())
        for gpu_id in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            logger.info(
                "GPU %s: %s, free %.1f/%.1f GiB",
                gpu_id,
                torch.cuda.get_device_name(gpu_id),
                free_bytes / 2**30,
                total_bytes / 2**30,
            )
        logger.info(
            "Parallel workers: gpu_ids=%s workers_per_gpu=%s",
            experiment.gpu_ids or "all",
            experiment.workers_per_gpu,
        )
    logger.info(
        "Architecture rows: %s",
        "all" if experiment.arch_rows is None else experiment.arch_rows,
    )
    logger.info(
        "FLOPs budget per architecture: %s", experiment.asha.budget_flops_per_arch
    )
    logger.info("Output directory: %s", experiment.output_dir.resolve())

    started_at = time.monotonic()
    result = run_hpo_experiment(experiment, device)
    elapsed_minutes = (time.monotonic() - started_at) / 60

    logger.info("HPO completed in %.2f minutes", elapsed_minutes)
    logger.info("Stage results: %s", result.stages_path.resolve())
    logger.info("Summary: %s", result.summary_path.resolve())
    for plot_path in result.plot_paths:
        logger.info("Plot: %s", plot_path.resolve())

    if result.summary.empty:
        logger.warning("The experiment produced no completed results")
    else:
        columns = [
            "arch_index",
            "trial_id",
            "lr",
            "weight_decay",
            "val_acc1",
            "test_acc1",
            "spent_budget_ratio",
        ]
        print("\nBest results:")
        print(result.summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
