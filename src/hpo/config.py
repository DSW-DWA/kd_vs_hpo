"""HPO experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.common.config import TrainConfig

SamplerName = Literal["tpe", "grid", "cmaes", "gp"]
PrunerName = Literal["none", "successive_halving", "hyperband"]

SAMPLER_NAMES: tuple[SamplerName, ...] = ("tpe", "grid", "cmaes", "gp")
PRUNER_NAMES: tuple[PrunerName, ...] = (
    "none",
    "successive_halving",
    "hyperband",
)
DEFAULT_PRUNERS: tuple[PrunerName, ...] = ("successive_halving", "hyperband")


@dataclass(frozen=True)
class SearchSpace:
    lr: tuple[float, float] = (4.5e-2, 5.5e-2)
    weight_decay: tuple[float, float] = (4.5e-4, 5.5e-4)
    initial_lr: float = 5.0e-2
    initial_weight_decay: float = 5.0e-4
    grid_lr: tuple[float, ...] = (
        4.5e-2,
        5.0e-2,
        5.5e-2,
    )
    grid_weight_decay: tuple[float, ...] = (
        4.5e-4,
        5.0e-4,
        5.5e-4,
    )


@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int = 20
    max_epochs: int = 200
    samplers: tuple[SamplerName, ...] = SAMPLER_NAMES
    pruners: tuple[PrunerName, ...] = DEFAULT_PRUNERS
    startup_trials: int = 5
    min_resource: int = 10
    reduction_factor: int = 3


@dataclass(frozen=True)
class HPOExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    architectures_path: Path = Path("experiments/nats_architectures_10.json")
    output_dir: Path = Path("hpo_output")
    arch_rows: tuple[int, ...] | None = None
    num_processes: int = 1
    gpu_ids: tuple[int, ...] | None = None


def validate_experiment(experiment: HPOExperimentConfig) -> None:
    optuna = experiment.optuna
    if experiment.num_processes < 1:
        raise ValueError("num_processes must be positive")
    if optuna.n_trials < 1 or optuna.max_epochs < 1:
        raise ValueError("n_trials and max_epochs must be positive")
    if optuna.startup_trials < 0:
        raise ValueError("startup_trials cannot be negative")
    if optuna.min_resource < 1:
        raise ValueError("min_resource must be positive")
    if optuna.reduction_factor < 2:
        raise ValueError("reduction_factor must be at least 2")
    if not optuna.samplers or not optuna.pruners:
        raise ValueError("At least one sampler and pruner must be selected")
    if len(set(optuna.samplers)) != len(optuna.samplers):
        raise ValueError("Sampler names must be unique")
    if len(set(optuna.pruners)) != len(optuna.pruners):
        raise ValueError("Pruner names must be unique")

    unsupported_samplers = set(optuna.samplers) - set(SAMPLER_NAMES)
    unsupported_pruners = set(optuna.pruners) - set(PRUNER_NAMES)
    if unsupported_samplers:
        raise ValueError(f"Unsupported samplers: {sorted(unsupported_samplers)}")
    if unsupported_pruners:
        raise ValueError(f"Unsupported pruners: {sorted(unsupported_pruners)}")

    resource_pruners = {"successive_halving", "hyperband"}
    if (
        resource_pruners.intersection(optuna.pruners)
        and optuna.max_epochs < optuna.min_resource
    ):
        raise ValueError(
            "max_epochs must be at least min_resource when using "
            "Successive Halving or Hyperband"
        )

    search = experiment.search_space
    if not search.lr[0] <= search.initial_lr <= search.lr[1]:
        raise ValueError("initial_lr must be within lr bounds")
    if not (
        search.weight_decay[0] <= search.initial_weight_decay <= search.weight_decay[1]
    ):
        raise ValueError("initial_weight_decay must be within weight_decay bounds")
    if "grid" not in optuna.samplers:
        return

    if not search.grid_lr or not search.grid_weight_decay:
        raise ValueError("Grid search values cannot be empty")
    if len(set(search.grid_lr)) != len(search.grid_lr):
        raise ValueError("grid_lr values must be unique")
    if len(set(search.grid_weight_decay)) != len(search.grid_weight_decay):
        raise ValueError("grid_weight_decay values must be unique")
    if any(value < search.lr[0] or value > search.lr[1] for value in search.grid_lr):
        raise ValueError("grid_lr values must be within lr bounds")
    if any(
        value < search.weight_decay[0] or value > search.weight_decay[1]
        for value in search.grid_weight_decay
    ):
        raise ValueError("grid_weight_decay values must be within weight_decay bounds")
    if search.initial_lr not in search.grid_lr:
        raise ValueError("initial_lr must be present in grid_lr")
    if search.initial_weight_decay not in search.grid_weight_decay:
        raise ValueError("initial_weight_decay must be present in grid_weight_decay")

    grid_size = len(search.grid_lr) * len(search.grid_weight_decay)
    if optuna.n_trials < grid_size:
        raise ValueError(
            f"n_trials must be at least {grid_size} when using GridSampler "
            "so every grid point, including the initial parameters, is evaluated"
        )
