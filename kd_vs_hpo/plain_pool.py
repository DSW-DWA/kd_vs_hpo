"""Train five plain models for each selected NATS architecture."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.common.flops import CounterMode, FlopsBudgetTracker, count_flops_params
from kd_vs_hpo.common.nats import create_nats_model
from kd_vs_hpo.common.train_pipeline import run_training_pipeline
from kd_vs_hpo.common.utils import get_architectures_from_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES_PATH = PROJECT_ROOT / "experiments/nats_architectures_7.json"
HPO_CONFIG_PATH = PROJECT_ROOT / "conf/hpo/hpo_base.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs/plain_pool_3_archs"
DATA_ROOT = PROJECT_ROOT / "data"

NATS_INDICES = (1342, 11570)
SEED = 42
EPOCHS = 200

logger = logging.getLogger(__name__)


def _load_candidates() -> list[tuple[str, float, float]]:
    search = OmegaConf.load(HPO_CONFIG_PATH).search_space
    corners = [
        (float(lr), float(weight_decay))
        for lr in search.lr
        for weight_decay in search.weight_decay
    ]
    random.Random(SEED).shuffle(corners)
    return [
        ("initial", float(search.initial_lr), float(search.initial_weight_decay)),
        *[
            (f"sample_{number}", lr, weight_decay)
            for number, (lr, weight_decay) in enumerate(corners, start=1)
        ],
    ]


def _load_architectures() -> list[dict]:
    records = get_architectures_from_json(str(ARCHITECTURES_PATH))
    by_index = {int(record["arch_index"]): record for record in records}
    missing = [index for index in NATS_INDICES if index not in by_index]
    if missing:
        raise ValueError(f"Architectures were not found: {missing}")
    return [by_index[index] for index in NATS_INDICES]


def _train_one(
    architecture: dict,
    label: str,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Path:
    arch_index = int(architecture["arch_index"])
    run_root = OUTPUT_DIR / label / f"arch_{arch_index}"
    config = TrainConfig(
        seed=SEED,
        data_root=DATA_ROOT,
        checkpoint_dir=run_root / "checkpoints",
        log_dir=run_root / "logs",
    )
    target = OUTPUT_DIR / "checkpoints" / f"arch_{arch_index}_{label}.pt"

    seed_everything(SEED)
    model = create_nats_model(architecture)
    forward_flops, _ = count_flops_params(model)
    run_training_pipeline(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        train_step_flops=int(forward_flops * config.train_step_multiplier),
        eval_step_flops=forward_flops,
        run_name=f"arch_{arch_index}_{label}",
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        data_root=config.data_root,
        max_epochs=EPOCHS,
        deterministic=config.deterministic,
        amp=config.amp,
        grad_clip_norm=config.grad_clip_norm or 0.0,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_fraction=config.validation_fraction,
        device=device,
        optimizer_kwargs={
            "lr": lr,
            "momentum": config.momentum,
            "weight_decay": weight_decay,
        },
        scheduler_kwargs={"T_max": EPOCHS},
        teacher_ensemble=None,
        kd_loss=None,
        flops_tracker=FlopsBudgetTracker(0, CounterMode.OFF),
        num_classes=int(architecture.get("num_classes", 10)),
        reusable_checkpoint_path=target,
    )

    return target


def run_plain_pool() -> None:
    architectures = _load_architectures()
    candidates = _load_candidates()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(architectures) * len(candidates)
    completed = 0
    logger.info(
        "Training %s models: arches=%s epochs=%s seed=%s device=%s",
        total,
        NATS_INDICES,
        EPOCHS,
        SEED,
        device,
    )

    for architecture in architectures:
        for label, lr, weight_decay in candidates:
            completed += 1
            logger.info(
                "Starting %s/%s: arch=%s %s lr=%g weight_decay=%g",
                completed,
                total,
                architecture["arch_index"],
                label,
                lr,
                weight_decay,
            )
            checkpoint = _train_one(
                architecture,
                label,
                lr,
                weight_decay,
                device,
            )
            logger.info("Reusable checkpoint saved: %s", checkpoint)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_plain_pool()


if __name__ == "__main__":
    main()
