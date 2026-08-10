from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.hpo.config import (
    HPOExperimentConfig,
    OptunaConfig,
    SearchSpace,
    validate_experiment,
)
from kd_vs_hpo.hpo.pipeline import run_hpo_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def build_experiment(cfg: DictConfig) -> HPOExperimentConfig:
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("HPO configuration must be a mapping")
    train_values = _mapping(values, "train")
    search_values = _mapping(values, "search_space")
    optuna_values = _mapping(values, "optuna")

    experiment = HPOExperimentConfig(
        train=TrainConfig(
            batch_size=int(train_values["batch_size"]),
            num_workers=int(train_values["num_workers"]),
            validation_fraction=float(train_values["validation_fraction"]),
            momentum=float(train_values["momentum"]),
            grad_clip_norm=(
                None
                if train_values["grad_clip_norm"] is None
                else float(train_values["grad_clip_norm"])
            ),
            seed=int(train_values["seed"]),
            deterministic=bool(train_values["deterministic"]),
            amp=bool(train_values["amp"]),
            train_step_multiplier=float(train_values["train_step_multiplier"]),
            data_root=project_path(str(train_values["data_root"])),
            checkpoint_dir=project_path(str(train_values["checkpoint_dir"])),
            log_dir=project_path(str(train_values["log_dir"])),
        ),
        search_space=SearchSpace(
            lr=_float_pair(search_values, "lr"),
            weight_decay=_float_pair(search_values, "weight_decay"),
            initial_lr=float(search_values["initial_lr"]),
            initial_weight_decay=float(search_values["initial_weight_decay"]),
            grid_lr=tuple(float(value) for value in search_values["grid_lr"]),
            grid_weight_decay=tuple(
                float(value) for value in search_values["grid_weight_decay"]
            ),
        ),
        optuna=OptunaConfig(
            n_trials=int(optuna_values["n_trials"]),
            max_epochs=int(optuna_values["max_epochs"]),
            samplers=tuple(optuna_values["samplers"]),
            pruners=tuple(optuna_values["pruners"]),
            startup_trials=int(optuna_values["startup_trials"]),
            min_resource=int(optuna_values["min_resource"]),
            reduction_factor=int(optuna_values["reduction_factor"]),
        ),
        architectures_path=project_path(str(values["architectures_path"])),
        output_dir=project_path(str(values["output_dir"])),
        arch_rows=(
            None
            if values["arch_rows"] is None
            else tuple(int(value) for value in values["arch_rows"])
        ),
        num_processes=int(values["num_processes"]),
        gpu_ids=(
            None
            if values["gpu_ids"] is None
            else tuple(int(value) for value in values["gpu_ids"])
        ),
        device=str(values["device"]),
    )
    validate_experiment(experiment)
    return experiment


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    result = values[key]
    if not isinstance(result, dict):
        raise TypeError(f"{key} must be a mapping")
    return result


def _float_pair(values: dict[str, Any], key: str) -> tuple[float, float]:
    pair = values[key]
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(f"{key} must contain exactly two values")
    return float(pair[0]), float(pair[1])


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="hpo_experiment",
)
def main(cfg: DictConfig) -> None:
    if len(sys.argv) != 1:
        raise ValueError(
            "hpo_optuna.py takes parameters only from configs/hpo_experiment.yaml; "
            "command-line arguments are not allowed"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    result = run_hpo_experiment(build_experiment(cfg))
    print(result.studies.to_string(index=False))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
