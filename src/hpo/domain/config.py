from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.common.config import TrainConfig

SamplerName = Literal["tpe", "grid", "cmaes", "gp"]
PrunerName = Literal["successive_halving", "hyperband"]


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
class EarlyStoppingConfig:
    min_growth: float = 0.05
    patience: int = 10
    warmup_epochs: int = 20


@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int = 20
    max_epochs: int = 200
    samplers: tuple[SamplerName, ...] = ("tpe", "grid", "cmaes", "gp")
    pruners: tuple[PrunerName, ...] = ("successive_halving", "hyperband")
    startup_trials: int = 5
    min_resource: int = 10
    reduction_factor: int = 3


@dataclass(frozen=True)
class HPOExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    architectures_path: Path = Path("experiments/nats_architectures_10.json")
    costs_path: Path = Path("experiments/sampled_architecture_costs.csv")
    output_dir: Path = Path("hpo_output")
    arch_rows: tuple[int, ...] | None = None
    generate_plots: bool = True
    num_processes: int = 1
    gpu_ids: tuple[int, ...] | None = None
