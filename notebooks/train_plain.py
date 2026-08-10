from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.plain_training import (
    load_architectures_by_rows,
    run_plain_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


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
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def select_student(
    architectures: list[dict],
    student: int | None,
) -> list[dict]:
    if student is None:
        return architectures
    selected = [item for item in architectures if item["arch_index"] == int(student)]
    if not selected:
        raise ValueError(f"Architecture with index {student} not found")
    return selected


def trial_epochs(max_epochs: int) -> tuple[int, int]:
    if max_epochs < 2:
        raise ValueError("max_epochs must be at least 2")
    return max_epochs // 2, max_epochs


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="base_models",
)
def main(cfg: DictConfig) -> None:
    if len(sys.argv) != 1:
        raise ValueError(
            "train_plain.py takes parameters only from configs/base_models.yaml; "
            "command-line arguments are not allowed"
        )
    cfg = hydra.utils.instantiate(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    architectures_path = project_path(cfg.architectures_path)
    architectures = select_student(
        load_architectures_by_rows(architectures_path, None),
        cfg.student,
    )
    device = resolve_device("auto")
    checkpoint_dir = project_path(cfg.checkpoint_dir)
    log_dir = project_path(cfg.log_dir)
    output_dir = checkpoint_dir
    epochs = trial_epochs(int(cfg.max_epochs))
    train_config = TrainConfig(
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        validation_fraction=float(cfg.validation_fraction),
        momentum=float(cfg.optimizer_params.momentum),
        grad_clip_norm=(
            None if cfg.grad_clip_norm is None else float(cfg.grad_clip_norm)
        ),
        seed=int(cfg.seed),
        deterministic=bool(cfg.deterministic),
        amp=bool(cfg.amp),
        train_step_multiplier=float(cfg.train_step_multiplier),
        data_root=project_path(cfg.data_root),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    result = run_plain_experiment(
        architectures=architectures,
        initial_lr=float(cfg.optimizer_params.lr),
        weight_decay=float(cfg.optimizer_params.weight_decay),
        train_config=train_config,
        device=device,
        output_dir=output_dir,
        trial_epochs=epochs,
        verbose=False,
    )
    print(result.runs.to_string(index=False))


if __name__ == "__main__":
    main()
