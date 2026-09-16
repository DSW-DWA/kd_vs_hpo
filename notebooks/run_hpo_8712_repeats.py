"""Train two new 8712 pools and run their independent TPE studies."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path

import torch
from hpo_optuna import build_experiment
from hydra import compose, initialize_config_dir

from kd_vs_hpo.hpo.config import HPOExperimentConfig, ImportedTrialConfig
from kd_vs_hpo.hpo.pipeline import run_hpo_experiment
from kd_vs_hpo.plain_pool import EPOCHS, SEED, _train_one

REPEATS = (("hpo_8712_repeat_1", 43), ("hpo_8712_repeat_2", 44))
ARCH_INDEX = 8712
PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


def _points(seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    quadrants = (
        ((0.092, 0.098), (0.00046, 0.00049)),
        ((0.102, 0.108), (0.00046, 0.00049)),
        ((0.092, 0.098), (0.00051, 0.00054)),
        ((0.102, 0.108), (0.00051, 0.00054)),
    )
    return [
        (round(rng.uniform(*lr_range), 3), round(rng.uniform(*wd_range), 5))
        for lr_range, wd_range in quadrants
    ]


def _validation_epochs(path: Path) -> set[int]:
    with path.open(encoding="utf-8", newline="") as file:
        return {int(row["epoch"]) for row in csv.DictReader(file) if row.get("val_acc")}


def _prepare() -> tuple[
    dict, list[tuple[HPOExperimentConfig, list[ImportedTrialConfig]]]
]:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "conf"),
    ):
        experiments = [
            build_experiment(
                compose(config_name="config", overrides=[f"hpo={config_name}"])
            )
            for config_name, _ in REPEATS
        ]

    with experiments[0].architectures_path.open(encoding="utf-8") as file:
        architectures = json.load(file)
    architecture = dict(architectures[2])
    if int(architecture["arch_index"]) != ARCH_INDEX:
        raise ValueError("Architecture row 2 is not 8712")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("This experiment requires GPU 0")

    prepared = []
    all_points = set()
    output_dirs = set()
    for experiment, (config_name, point_seed) in zip(experiments, REPEATS, strict=True):
        if experiment.arch_rows != (2,) or experiment.train.seed != SEED:
            raise ValueError(
                f"Unexpected architecture or training seed in {config_name}"
            )
        if experiment.optuna.max_epochs != EPOCHS or experiment.optuna.n_trials != 50:
            raise ValueError(f"Unexpected trial count or epochs in {config_name}")
        if experiment.optuna.samplers != ("tpe",) or experiment.optuna.pruners != (
            "none",
        ):
            raise ValueError(f"Expected TPE without a pruner in {config_name}")
        if experiment.num_processes != 1 or experiment.gpu_ids != (0,):
            raise ValueError(f"Expected one process on GPU 0 in {config_name}")
        if experiment.output_dir in output_dirs:
            raise ValueError("The two HPO output directories must differ")
        output_dirs.add(experiment.output_dir)
        if experiment.output_dir.exists() and any(experiment.output_dir.iterdir()):
            raise FileExistsError(
                f"HPO output already contains files: {experiment.output_dir}"
            )

        imported = [
            trial
            for trial in experiment.imported_trials
            if trial.arch_index == ARCH_INDEX
        ]
        if len(imported) != 5:
            raise ValueError(f"Expected five imported trials in {config_name}")
        initial = imported[0]
        if not initial.checkpoint_path.is_file() or not initial.metrics_path.is_file():
            raise FileNotFoundError(
                "The original 8712 initial checkpoint or metrics are missing"
            )
        if (initial.lr, initial.weight_decay) != (0.1, 0.0005):
            raise ValueError("The original initial parameters have changed")
        if _validation_epochs(initial.metrics_path) != set(range(EPOCHS)):
            raise ValueError("The original initial run is not a complete 200-epoch run")

        points = [(trial.lr, trial.weight_decay) for trial in imported[1:]]
        if points != _points(point_seed) or len(set(points)) != 4:
            raise ValueError(f"Starting points differ from seed {point_seed}")
        if all_points.intersection(points):
            raise ValueError("The two repeats share starting points")
        all_points.update(points)

        pool_root = (
            PROJECT_ROOT / "outputs" / config_name.replace("hpo_", "plain_pool_")
        )
        for number, trial in enumerate(imported[1:], start=1):
            label = f"sample_{number}"
            expected_checkpoint = pool_root / "checkpoints" / f"arch_8712_{label}.pt"
            expected_metrics = (
                pool_root
                / label
                / "arch_8712"
                / "checkpoints"
                / f"arch_8712_{label}"
                / "metrics.csv"
            )
            if (
                trial.checkpoint_path != expected_checkpoint
                or trial.metrics_path != expected_metrics
            ):
                raise ValueError(f"Unexpected pool paths in {config_name}: {label}")
            checkpoint_exists = trial.checkpoint_path.is_file()
            metrics_exist = trial.metrics_path.is_file()
            if checkpoint_exists != metrics_exist:
                raise RuntimeError(
                    f"Incomplete pool run for {config_name}/{label}; inspect its files"
                )
            if metrics_exist and _validation_epochs(trial.metrics_path) != set(
                range(EPOCHS)
            ):
                raise ValueError(f"Incomplete validation history: {trial.metrics_path}")
        prepared.append((experiment, imported[1:]))
    return architecture, prepared


def main() -> None:
    if len(sys.argv) != 1:
        raise ValueError("This launcher does not accept command-line arguments")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    architecture, repeats = _prepare()
    device = torch.device("cuda:0")

    for experiment, pool_trials in repeats:
        logger.info("Preparing pool for %s", experiment.output_dir.name)
        for number, trial in enumerate(pool_trials, start=1):
            if trial.checkpoint_path.is_file():
                logger.info("Reusing completed pool model: %s", trial.checkpoint_path)
                continue
            logger.info(
                "Training pool sample_%s: lr=%s wd=%s",
                number,
                trial.lr,
                trial.weight_decay,
            )
            checkpoint = _train_one(
                architecture,
                f"sample_{number}",
                trial.lr,
                trial.weight_decay,
                device,
                output_dir=trial.checkpoint_path.parents[1],
            )
            if checkpoint != trial.checkpoint_path or not trial.metrics_path.is_file():
                raise RuntimeError(f"Pool artifacts were not saved for sample_{number}")
            if _validation_epochs(trial.metrics_path) != set(range(EPOCHS)):
                raise RuntimeError(
                    f"Pool validation history is incomplete: {trial.metrics_path}"
                )

    for experiment, _ in repeats:
        logger.info("Starting HPO: %s", experiment.output_dir)
        result = run_hpo_experiment(experiment)
        logger.info("Completed HPO:\n%s", result.studies.to_string(index=False))


if __name__ == "__main__":
    main()
