from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig

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
    general_cfg = cfg.general
    hpo_cfg = cfg.hpo

    experiment = HPOExperimentConfig(
        train=TrainConfig(
            batch_size=int(general_cfg.batch_size),
            num_workers=int(general_cfg.num_workers),
            validation_fraction=float(general_cfg.validation_fraction),
            momentum=float(general_cfg.momentum),
            grad_clip_norm=(
                None
                if general_cfg.grad_clip_norm is None
                else float(general_cfg.grad_clip_norm)
            ),
            seed=int(general_cfg.seed),
            deterministic=bool(general_cfg.deterministic),
            amp=bool(general_cfg.amp),
            train_step_multiplier=float(general_cfg.train_step_multiplier),
            data_root=project_path(str(general_cfg.data_root)),
            checkpoint_dir=project_path(str(hpo_cfg.checkpoint_dir)),
            log_dir=project_path(str(hpo_cfg.log_dir)),
        ),
        search_space=SearchSpace(
            lr=_float_pair(hpo_cfg.search_space.lr, "lr"),
            weight_decay=_float_pair(
                hpo_cfg.search_space.weight_decay,
                "weight_decay",
            ),
            initial_lr=float(hpo_cfg.search_space.initial_lr),
            initial_weight_decay=float(hpo_cfg.search_space.initial_weight_decay),
            grid_lr=tuple(float(value) for value in hpo_cfg.search_space.grid_lr),
            grid_weight_decay=tuple(
                float(value) for value in hpo_cfg.search_space.grid_weight_decay
            ),
        ),
        optuna=OptunaConfig(
            n_trials=int(hpo_cfg.optuna.n_trials),
            max_epochs=int(hpo_cfg.optuna.max_epochs),
            samplers=tuple(hpo_cfg.optuna.samplers),
            pruners=tuple(hpo_cfg.optuna.pruners),
            startup_trials=int(hpo_cfg.optuna.startup_trials),
            min_resource=int(hpo_cfg.optuna.min_resource),
            reduction_factor=int(hpo_cfg.optuna.reduction_factor),
        ),
        architectures_path=project_path(str(general_cfg.architectures_path)),
        output_dir=project_path(str(hpo_cfg.output_dir)),
        arch_rows=(
            None
            if hpo_cfg.arch_rows is None
            else tuple(int(value) for value in hpo_cfg.arch_rows)
        ),
        num_processes=int(hpo_cfg.num_processes),
        gpu_ids=(
            None
            if hpo_cfg.gpu_ids is None
            else tuple(int(value) for value in hpo_cfg.gpu_ids)
        ),
        device=str(hpo_cfg.device),
    )
    validate_experiment(experiment)
    return experiment


def _float_pair(pair: ListConfig, name: str) -> tuple[float, float]:
    if not isinstance(pair, ListConfig) or len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return float(pair[0]), float(pair[1])


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    if len(sys.argv) != 1:
        raise ValueError(
            "hpo_optuna.py takes parameters only from conf/config.yaml; "
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
