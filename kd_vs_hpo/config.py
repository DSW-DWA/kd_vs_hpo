from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    num_workers: int = 2
    validation_fraction: float = 0.1
    momentum: float = 0.9
    grad_clip_norm: float | None = 5.0
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    train_step_multiplier: float = 3.0
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")
    kd_temperature: float = 2.0