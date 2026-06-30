from dataclasses import dataclass, field

from src.hpo.domain.config import EarlyStoppingConfig


@dataclass
class AccuracyGrowthStopper:
    config: EarlyStoppingConfig
    best_values: list[float] = field(default_factory=list)
    last_growth: float | None = None

    def update(self, value: float, epoch: int) -> str | None:
        previous_best = self.best_values[-1] if self.best_values else float("-inf")
        self.best_values.append(max(previous_best, value))
        if epoch < self.config.warmup_epochs or len(self.best_values) <= self.config.patience:
            return None
        growth = self.best_values[-1] - self.best_values[-self.config.patience - 1]
        self.last_growth = growth
        if growth <= 0:
            return "NO_GROWTH"
        if growth < self.config.min_growth:
            return "LOW_GROWTH"
        return None
