import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic = reproducible; non-deterministic may be faster.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        tensors = [item for item in model_output if torch.is_tensor(item)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Model output does not contain logits tensor: {type(model_output)!r}")


def accuracy_top1_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def checkpoint_path(checkpoint_dir: Path, arch_row: int, trial_id: int) -> Path:
    return checkpoint_dir / f"arch_{arch_row:02d}_trial_{trial_id:02d}.pt"


def stage_checkpoint_path(checkpoint_dir: Path, arch_row: int, trial_id: int, target_epochs: int) -> Path:
    return checkpoint_dir / f"arch_{arch_row:02d}_trial_{trial_id:02d}_epoch_{target_epochs:04d}.pt"
