import math
import random
from dataclasses import dataclass

from src.hpo.domain.config import ASHAConfig, SearchSpace


@dataclass(frozen=True)
class TrialConfig:
    trial_id: int
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class ASHAPlan:
    min_epochs: int
    max_epochs: int
    reduction_factor: int
    num_initial_configs: int
    budget_epochs: int
    rungs: tuple[int, ...]
    planned_train_epochs: int
    planned_validation_stages: int
    planned_flops: int


def sample_trials(count: int, search_space: SearchSpace, seed: int) -> list[TrialConfig]:
    rng = random.Random(seed)

    def log_uniform(bounds: tuple[float, float]) -> float:
        low, high = bounds
        if low <= 0 or high <= low:
            raise ValueError(f"Invalid log-uniform bounds: {bounds}")
        return math.exp(rng.uniform(math.log(low), math.log(high)))

    return [
        TrialConfig(
            trial_id=trial_id,
            lr=log_uniform(search_space.lr),
            weight_decay=log_uniform(search_space.weight_decay),
        )
        for trial_id in range(count)
    ]


def estimate_work(
    num_configs: int,
    rungs: tuple[int, ...] | list[int],
    reduction_factor: int,
) -> tuple[int, int]:
    total_epochs = 0
    total_stages = 0
    alive = num_configs
    completed_epochs = 0
    for target_epochs in rungs:
        total_stages += alive
        total_epochs += alive * (target_epochs - completed_epochs)
        alive = max(1, math.ceil(alive / reduction_factor))
        completed_epochs = target_epochs
    return total_epochs, total_stages


def estimate_flops(
    num_configs: int,
    rungs: tuple[int, ...] | list[int],
    reduction_factor: int,
    epoch_flops: int,
    validation_flops: int,
) -> tuple[int, int, int]:
    train_epochs, validation_stages = estimate_work(num_configs, rungs, reduction_factor)
    total_flops = train_epochs * epoch_flops + validation_stages * validation_flops
    return train_epochs, validation_stages, total_flops


def make_plan(config: ASHAConfig, epoch_flops: int, validation_flops: int) -> ASHAPlan:
    if epoch_flops <= 0 or validation_flops < 0:
        raise ValueError("FLOPs estimates must be positive")
    if config.min_initial_configs < 1:
        raise ValueError("min_initial_configs must be at least 1")
    if config.max_initial_configs is not None and config.max_initial_configs < 1:
        raise ValueError("max_initial_configs must be at least 1")

    budget_flops = config.budget_flops_per_arch
    budget_epochs = budget_flops // epoch_flops
    if budget_epochs < 1:
        raise ValueError("FLOPs budget is smaller than one training epoch")

    reduction_factor = max(2, config.reduction_factor)
    min_epochs = max(1, min(config.target_min_epochs, budget_epochs))
    min_stage_flops = min_epochs * epoch_flops + validation_flops
    max_configs_that_fit = budget_flops // min_stage_flops
    if max_configs_that_fit < config.min_initial_configs:
        raise ValueError("FLOPs budget cannot fit the minimum number of initial configs")
    requested_configs = (
        config.max_initial_configs
        if config.max_initial_configs is not None
        else max_configs_that_fit
    )
    num_initial_configs = max(
        config.min_initial_configs,
        min(requested_configs, max_configs_that_fit),
    )

    rungs = [min_epochs]
    while True:
        next_epoch = rungs[-1] * reduction_factor
        if config.max_epochs is not None and next_epoch > config.max_epochs:
            break
        candidate = [*rungs, next_epoch]
        *_, candidate_flops = estimate_flops(
            num_initial_configs,
            candidate,
            reduction_factor,
            epoch_flops,
            validation_flops,
        )
        if candidate_flops > budget_flops:
            break
        rungs = candidate

    train_epochs, validation_stages, planned_flops = estimate_flops(
        num_initial_configs,
        rungs,
        reduction_factor,
        epoch_flops,
        validation_flops,
    )
    return ASHAPlan(
        min_epochs=min_epochs,
        max_epochs=rungs[-1],
        reduction_factor=reduction_factor,
        num_initial_configs=num_initial_configs,
        budget_epochs=budget_epochs,
        rungs=tuple(rungs),
        planned_train_epochs=train_epochs,
        planned_validation_stages=validation_stages,
        planned_flops=planned_flops,
    )
