"""Launch five independent plain-training runs for NATS architecture 8712."""

from __future__ import annotations

import logging
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from kd_vs_hpo.common.config import TrainConfig
from kd_vs_hpo.plain_training import (
    PlainExperimentResult,
    load_architectures_by_rows,
    run_plain_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES_PATH = PROJECT_ROOT / "experiments/nats_architectures_10.json"
HPO_CONFIG_PATH = PROJECT_ROOT / "conf/hpo/hpo_base.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs/plain_pool_8712"
DATA_ROOT = PROJECT_ROOT / "data"
NATS_INDEX = 8712
TRAINING_SEED = 42
EPOCHS_PER_MODEL = 200
POOL_SIZE = 5
SAMPLING_SEED = 42

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlainPoolCandidate:
    """One fixed optimizer configuration in the plain-training pool."""

    label: str
    initial_lr: float
    weight_decay: float


@dataclass(frozen=True)
class PlainPoolSearchSpace:
    """Optimizer values read from conf/hpo/hpo_base.yaml."""

    initial_lr: float
    initial_weight_decay: float
    lr_bounds: tuple[float, float]
    weight_decay_bounds: tuple[float, float]


def load_search_space() -> PlainPoolSearchSpace:
    """Load initial values and sampling bounds without starting HPO."""
    config = OmegaConf.load(HPO_CONFIG_PATH)
    try:
        search = config.search_space
        lr_bounds = tuple(float(value) for value in search.lr)
        weight_decay_bounds = tuple(float(value) for value in search.weight_decay)
        initial_lr = float(search.initial_lr)
        initial_weight_decay = float(search.initial_weight_decay)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid search space in {HPO_CONFIG_PATH}") from error

    if len(lr_bounds) != 2 or len(weight_decay_bounds) != 2:
        raise ValueError("lr and weight_decay bounds must contain exactly two values")
    lr_pair = (lr_bounds[0], lr_bounds[1])
    weight_decay_pair = (weight_decay_bounds[0], weight_decay_bounds[1])
    for name, (low, high) in (
        ("lr", lr_pair),
        ("weight_decay", weight_decay_pair),
    ):
        if low <= 0 or high <= low:
            raise ValueError(f"{name} bounds must be positive and increasing")
    if not lr_pair[0] <= initial_lr <= lr_pair[1]:
        raise ValueError("initial_lr must be inside lr bounds")
    if not weight_decay_pair[0] <= initial_weight_decay <= weight_decay_pair[1]:
        raise ValueError("initial_weight_decay must be inside weight_decay bounds")

    return PlainPoolSearchSpace(
        initial_lr=initial_lr,
        initial_weight_decay=initial_weight_decay,
        lr_bounds=lr_pair,
        weight_decay_bounds=weight_decay_pair,
    )


def build_hyperparameter_pool(
    search_space: PlainPoolSearchSpace,
    sampling_seed: int = SAMPLING_SEED,
) -> tuple[PlainPoolCandidate, ...]:
    """Return the configured initial point plus four reproducible samples."""
    generator = random.Random(sampling_seed)
    candidates = [
        PlainPoolCandidate(
            label="initial",
            initial_lr=search_space.initial_lr,
            weight_decay=search_space.initial_weight_decay,
        )
    ]
    for number in range(1, POOL_SIZE):
        initial_lr = math.exp(
            generator.uniform(
                math.log(search_space.lr_bounds[0]),
                math.log(search_space.lr_bounds[1]),
            )
        )
        weight_decay = math.exp(
            generator.uniform(
                math.log(search_space.weight_decay_bounds[0]),
                math.log(search_space.weight_decay_bounds[1]),
            )
        )
        candidates.append(
            PlainPoolCandidate(
                label=f"sample_{number}",
                initial_lr=initial_lr,
                weight_decay=weight_decay,
            )
        )
    return tuple(candidates)


def resolve_auto_device() -> torch.device:
    """Select the best available device without requiring a command-line option."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_architecture_8712() -> dict:
    """Load the single selected NATS architecture from the project experiment set."""
    architectures = load_architectures_by_rows(ARCHITECTURES_PATH, None)
    for architecture in architectures:
        if architecture["arch_index"] == NATS_INDEX:
            return architecture
    raise ValueError(f"Architecture with index {NATS_INDEX} was not found")


def _train_config(run_dir: Path) -> TrainConfig:
    return TrainConfig(
        seed=TRAINING_SEED,
        data_root=DATA_ROOT,
        checkpoint_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
    )


def export_checkpoint(
    candidate: PlainPoolCandidate,
    result: PlainExperimentResult,
) -> Path:
    """Publish the plain-training checkpoint under a stable reusable name."""
    if len(result.runs) != 1:
        raise ValueError("Each pool candidate must produce exactly one run")
    source_path = Path(str(result.runs.iloc[0]["checkpoint_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"Plain-training checkpoint not found: {source_path}")

    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{candidate.label}.pt"
    temporary_path = checkpoint_path.with_suffix(".tmp")
    shutil.copy2(source_path, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def run_plain_pool() -> list[tuple[PlainPoolCandidate, PlainExperimentResult]]:
    """Train architecture 8712 once for each fixed hyperparameter configuration."""
    architecture = load_architecture_8712()
    candidates = build_hyperparameter_pool(load_search_space())
    device = resolve_auto_device()
    logger.info(
        "Plain pool: arch=%s models=%s epochs=%s seed=%s device=%s output=%s",
        NATS_INDEX,
        POOL_SIZE,
        EPOCHS_PER_MODEL,
        TRAINING_SEED,
        device,
        OUTPUT_DIR,
    )
    results: list[tuple[PlainPoolCandidate, PlainExperimentResult]] = []
    for pool_index, candidate in enumerate(candidates, start=1):
        run_dir = OUTPUT_DIR / candidate.label
        logger.info(
            "Starting model %s/%s: %s lr=%.8g weight_decay=%.8g",
            pool_index,
            POOL_SIZE,
            candidate.label,
            candidate.initial_lr,
            candidate.weight_decay,
        )
        result = run_plain_experiment(
            architectures=[architecture],
            initial_lr=candidate.initial_lr,
            weight_decay=candidate.weight_decay,
            train_config=_train_config(run_dir),
            device=device,
            output_dir=run_dir,
            trial_epochs=(EPOCHS_PER_MODEL,),
            verbose=True,
        )
        checkpoint_path = export_checkpoint(candidate, result)
        logger.info("Reusable checkpoint saved: %s", checkpoint_path)
        results.append((candidate, result))
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for candidate, result in run_plain_pool():
        print(f"\n{candidate.label}")
        print(result.runs.to_string(index=False))


if __name__ == "__main__":
    main()
