import argparse
import logging
import os
from pathlib import Path


def _configure_process_environment() -> None:
    thread_count = os.environ.get("KD_VS_HPO_BLAS_THREADS", "1")
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = thread_count

    if os.name == "nt":
        default_temp = (
            Path(os.environ.get("TEMP", Path.home() / "AppData/Local/Temp"))
            / "kd_vs_hpo"
        )
    else:
        default_temp = Path(f"/tmp/kd_vs_hpo_{os.getuid()}")
    temp_root = Path(os.environ.get("KD_VS_HPO_TMPDIR", default_temp))
    temp_root.mkdir(parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        os.environ[variable] = str(temp_root)


_configure_process_environment()

import torch  # noqa: E402

from kd_vs_hpo.common.config import TrainConfig  # noqa: E402
from kd_vs_hpo.hpo.optuna_experiment import (  # noqa: E402
    DEFAULT_PRUNERS,
    DEFAULT_SAMPLERS,
    OptunaExperimentConfig,
    OptunaSearchSpace,
    PlateauConfig,
    PruningConfig,
    StudyStopConfig,
    aggregate_optuna_metrics,
    run_optuna_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna HPO for NATS-Bench architectures and export "
            "epoch/trial/study metrics."
        )
    )
    parser.add_argument(
        "--arch-rows",
        type=int,
        nargs="*",
        default=None,
        help="Architecture rows to run. Omit values to run all 10 rows.",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=DEFAULT_SAMPLERS,
        default=list(DEFAULT_SAMPLERS),
    )
    parser.add_argument(
        "--pruners",
        nargs="+",
        choices=DEFAULT_PRUNERS,
        default=list(DEFAULT_PRUNERS),
    )
    parser.add_argument("--sampler-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--plateau-warmup-epochs", type=int, default=30)
    parser.add_argument("--plateau-patience", type=int, default=25)
    parser.add_argument("--plateau-min-delta", type=float, default=0.05)
    parser.add_argument("--plateau-smoothing-window", type=int, default=5)
    parser.add_argument("--pruner-min-resource", type=int, default=20)
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--min-started-trials", type=int, default=80)
    parser.add_argument("--min-complete-trials", type=int, default=40)
    parser.add_argument("--study-stagnation-window", type=int, default=40)
    parser.add_argument("--study-min-improvement", type=float, default=0.05)
    parser.add_argument("--max-started-trials", type=int, default=300)
    parser.add_argument("--qmc-max-started-trials", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("optuna_output"))
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Enable cuDNN benchmark mode instead of deterministic kernels.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Write CSV tables only.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild CSV/Parquet tables from existing JSON/JSONL metrics.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OptunaExperimentConfig:
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    arch_rows = tuple(args.arch_rows) if args.arch_rows else None
    gpu_ids = tuple(args.gpu_ids) if args.gpu_ids else None
    train = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        validation_fraction=0.1,
        momentum=0.9,
        grad_clip_norm=5.0,
        seed=42,
        deterministic=not args.non_deterministic,
        amp=True,
        train_step_multiplier=3.0,
        data_root=PROJECT_ROOT / "data",
        checkpoint_dir=output_dir / "checkpoints",
        log_dir=output_dir / "runs",
    )
    return OptunaExperimentConfig(
        train=train,
        search_space=OptunaSearchSpace(),
        plateau=PlateauConfig(
            warmup_epochs=args.plateau_warmup_epochs,
            patience=args.plateau_patience,
            min_delta=args.plateau_min_delta,
            smoothing_window=args.plateau_smoothing_window,
            max_epochs=args.max_epochs,
        ),
        study_stop=StudyStopConfig(
            min_started_trials=args.min_started_trials,
            min_complete_trials=args.min_complete_trials,
            stagnation_window=args.study_stagnation_window,
            min_improvement=args.study_min_improvement,
            max_started_trials=args.max_started_trials,
            qmc_max_started_trials=args.qmc_max_started_trials,
        ),
        pruning=PruningConfig(
            min_resource=args.pruner_min_resource,
            reduction_factor=args.reduction_factor,
        ),
        architectures_path=PROJECT_ROOT / "experiments/nats_architectures_10.json",
        costs_path=PROJECT_ROOT / "experiments/sampled_architecture_costs.csv",
        output_dir=output_dir,
        arch_rows=arch_rows,
        samplers=tuple(args.samplers),
        pruners=tuple(args.pruners),
        sampler_seeds=tuple(args.sampler_seeds),
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        torch_threads_per_worker=args.torch_threads_per_worker,
        write_parquet=not args.no_parquet,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = build_config(args)

    if args.aggregate_only:
        result = aggregate_optuna_metrics(
            config.output_dir,
            write_parquet=config.write_parquet,
        )
    else:
        if torch.cuda.is_available():
            for gpu_id in (
                config.gpu_ids
                if config.gpu_ids is not None
                else tuple(range(torch.cuda.device_count()))
            ):
                properties = torch.cuda.get_device_properties(gpu_id)
                logging.info(
                    "GPU %s: %s, %.1f GiB",
                    gpu_id,
                    properties.name,
                    properties.total_memory / 1024**3,
                )
        result = run_optuna_experiment(config)

    logging.info("Epoch metrics: %s", result.epoch_metrics_path)
    logging.info("Trial summary: %s", result.trial_summary_path)
    logging.info("Study summary: %s", result.study_summary_path)
    logging.info("Optimization history: %s", result.optimization_history_path)
    if result.parquet_paths:
        logging.info("Parquet tables: %s", result.parquet_paths)
    if result.failed_studies:
        logging.error("Failed studies: %s", result.failed_studies)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
