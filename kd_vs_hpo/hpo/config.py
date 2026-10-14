"""HPO experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kd_vs_hpo.common.config import TrainConfig

SamplerName = Literal["tpe", "grid", "cmaes", "gp"]
PrunerName = Literal["none", "successive_halving", "hyperband"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]

SAMPLER_NAMES: tuple[SamplerName, ...] = ("tpe", "grid", "cmaes", "gp")
PRUNER_NAMES: tuple[PrunerName, ...] = (
    "none",
    "successive_halving",
    "hyperband",
)
DEFAULT_PRUNERS: tuple[PrunerName, ...] = ("successive_halving", "hyperband")


@dataclass(frozen=True)
class SearchSpace:
    lr: tuple[float, float] = (1.0e-5, 3.0e-1)
    weight_decay: tuple[float, float] = (1.0e-7, 5.0e-3)
    batch_size: tuple[int, int] = (16, 512)
    batch_size_step: int = 16
    momentum: tuple[float, float] = (0.0, 0.9)
    initial_batch_size: int = 256
    initial_momentum: float = 0.9
    grid_momentum: tuple[float, ...] = (0.0, 0.45, 0.9)
    initial_lr: float = 1.0e-1
    initial_weight_decay: float = 5.0e-4
    grid_lr: tuple[float, ...] = (
        9.0e-2,
        1.0e-1,
        1.1e-1,
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
    fixed_trial_seed: bool = False
    fresh_dataloaders_per_trial: bool = False


@dataclass(frozen=True)
class ImportedTrialConfig:
    arch_index: int
    checkpoint_path: Path
    metrics_path: Path
    lr: float
    weight_decay: float
    batch_size: int = 256
    momentum: float = 0.9


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
    device: DeviceName = "auto"
    imported_trials: tuple[ImportedTrialConfig, ...] = ()


def validate_imported_trial_files(experiment: HPOExperimentConfig) -> None:
    """Reject missing files and LFS pointers before datasets or training are loaded."""
    for trial in experiment.imported_trials:
        for path in (trial.checkpoint_path, trial.metrics_path):
            if not path.is_file():
                raise FileNotFoundError(f"Imported trial file was not found: {path}")
            with path.open("rb") as file:
                header = file.read(128)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"Imported trial file is a Git LFS pointer: {path}. "
                    "Install Git LFS and run git lfs pull for the source files "
                    "before starting the experiment."
                )


def validate_experiment(experiment: HPOExperimentConfig) -> None:
    optuna = experiment.optuna
    if experiment.num_processes < 1:
        raise ValueError("num_processes must be positive")
    if experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {experiment.device}")
    if optuna.n_trials < 1 or optuna.max_epochs < 1:
        raise ValueError("n_trials and max_epochs must be positive")
    if optuna.startup_trials < 0:
        raise ValueError("startup_trials cannot be negative")
    if optuna.min_resource < 1:
        raise ValueError("min_resource must be positive")
    if optuna.reduction_factor < 2:
        raise ValueError("reduction_factor must be at least 2")
    if optuna.fresh_dataloaders_per_trial and not optuna.fixed_trial_seed:
        raise ValueError(
            "fresh_dataloaders_per_trial requires fixed_trial_seed so the "
            "validation split stays unchanged"
        )
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

    if experiment.imported_trials:
        if experiment.arch_rows is None:
            raise ValueError("Imported trials require selected architecture rows")
        if len(optuna.samplers) != 1 or len(optuna.pruners) != 1:
            raise ValueError("Imported trials require exactly one sampler and pruner")
        for arch_index in {trial.arch_index for trial in experiment.imported_trials}:
            imported_count = sum(
                trial.arch_index == arch_index for trial in experiment.imported_trials
            )
            if imported_count > optuna.n_trials:
                raise ValueError(
                    "Imported trials per architecture cannot exceed n_trials"
                )

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
    if search.lr[0] <= 0 or search.lr[1] <= search.lr[0]:
        raise ValueError("lr bounds must be positive and increasing")
    if search.weight_decay[0] <= 0 or search.weight_decay[1] <= search.weight_decay[0]:
        raise ValueError("weight_decay bounds must be positive and increasing")
    if not search.lr[0] <= search.initial_lr <= search.lr[1]:
        raise ValueError("initial_lr must be within lr bounds")
    if not (
        search.weight_decay[0] <= search.initial_weight_decay <= search.weight_decay[1]
    ):
        raise ValueError("initial_weight_decay must be within weight_decay bounds")
    if len(search.batch_size) != 2 or len(search.momentum) != 2:
        raise ValueError("batch_size and momentum must each contain two bounds")
    low, high = search.batch_size
    if low < 1 or high < low or search.batch_size_step < 1:
        raise ValueError("batch_size bounds and step must be positive and increasing")
    if (high - low) % search.batch_size_step:
        raise ValueError("batch_size upper bound must align with batch_size_step")
    if not 0 <= search.momentum[0] < search.momentum[1] <= 1:
        raise ValueError("momentum bounds must be increasing within [0, 1]")

    def validate_extra_parameters(batch_size: int, momentum: float) -> None:
        if not low <= batch_size <= high or (batch_size - low) % search.batch_size_step:
            raise ValueError("batch_size must lie on the configured search steps")
        if not search.momentum[0] <= momentum <= search.momentum[1]:
            raise ValueError("momentum must be within search bounds")

    validate_extra_parameters(search.initial_batch_size, search.initial_momentum)
    for imported in experiment.imported_trials:
        validate_extra_parameters(imported.batch_size, imported.momentum)
        if not search.lr[0] <= imported.lr <= search.lr[1]:
            raise ValueError("Imported trial lr must be within lr bounds")
        if not (
            search.weight_decay[0] <= imported.weight_decay <= search.weight_decay[1]
        ):
            raise ValueError(
                "Imported trial weight_decay must be within weight_decay bounds"
            )
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

    if not search.grid_momentum or len(set(search.grid_momentum)) != len(
        search.grid_momentum
    ):
        raise ValueError("grid_momentum must contain unique values")
    if any(
        not search.momentum[0] <= value <= search.momentum[1]
        for value in search.grid_momentum
    ):
        raise ValueError("grid_momentum values must be within momentum bounds")
    if search.initial_momentum not in search.grid_momentum:
        raise ValueError("initial_momentum must be present in grid_momentum")
    grid_size = (
        len(search.grid_lr)
        * len(search.grid_weight_decay)
        * len(range(low, high + 1, search.batch_size_step))
        * len(search.grid_momentum)
    )
    if optuna.n_trials < grid_size:
        raise ValueError(
            f"n_trials must be at least {grid_size} when using GridSampler "
            "so every grid point, including the initial parameters, is evaluated"
        )
