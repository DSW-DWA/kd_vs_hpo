from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import TrainConfig


@dataclass(frozen=True)
class SearchSpace:
    lr: tuple[float, float] = (1e-3, 3e-1)
    weight_decay: tuple[float, float] = (1e-6, 1e-3)


@dataclass(frozen=True)
class ASHAConfig:
    budget_flops_per_arch: int = 10**15
    target_min_epochs: int = 3
    reduction_factor: int = 3
    min_initial_configs: int = 1
    max_initial_configs: int | None = 12
    max_epochs: int | None = 81


@dataclass(frozen=True)
class HPOExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    asha: ASHAConfig = field(default_factory=ASHAConfig)
    architectures_path: Path = Path("experiments/nats_architectures_10.json")
    costs_path: Path = Path("experiments/sampled_architecture_costs.csv")
    output_dir: Path = Path("hpo_output")
    arch_rows: tuple[int, ...] | None = None
    generate_plots: bool = True
    gpu_ids: tuple[int, ...] | None = None
    workers_per_gpu: int = 1
