import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from datetime import datetime

from kd_vs_hpo.common.nats import create_nats_model



DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "conf")

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        tensors = [item for item in model_output if torch.is_tensor(item)]
        if tensors:
            return tensors[-1]
    raise TypeError(
        f"Model output does not contain logits tensor: {type(model_output)!r}"
    )


def accuracy_top1_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def checkpoint_path(
    checkpoint_dir: Path,
    arch_row: int,
    trial_id: int,
) -> Path:
    return checkpoint_dir / f"arch_{arch_row:02d}_trial_{trial_id:02d}.pt"


def stage_checkpoint_path(checkpoint_dir: Path, arch_row: int, trial_id: int, target_epochs: int) -> Path:
    return checkpoint_dir / f"arch_{arch_row:02d}_trial_{trial_id:02d}_epoch_{target_epochs:04d}.pt"


def get_arch_by_idx(idx, architectures):
    return next((a for a in architectures if a['arch_index'] == idx), None)

def get_architectures_from_json(arch_file: str):
    with open(arch_file, 'r') as f:
        architectures = json.load(f)
    return architectures

def get_datetime():
    time = datetime.now()
    return time.strftime("%Y-%m-%d-%H-%M-%S")

def resolve_dir(path: str):
    _path = Path(path)
    if _path.exists():
        return str(_path.parent / (_path.name + '_' + get_datetime()))
    return path


def load_checkpoint(
    path,
    architecture,
    criterion=None,
    optimizer_cls=None,
    optimizer_kwargs=None,
    train_step_flops=0,
    eval_step_flops=0,
    num_classes=None,
    scheduler_cls=None,
    scheduler_kwargs=None,
    teacher_ensemble=None,
    kd_loss=None,
    flops_tracker=None,
):
    from kd_vs_hpo.common.train_modules import KDLightningModule

    try:
        return KDLightningModule.load_from_checkpoint(path, weights_only=False)
    except (TypeError, KeyError):
        checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        if "model" in checkpoint:
            state = checkpoint["model"]
            prefix = ""
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            prefix = "model."
        else:
            raise ValueError(
                f"Checkpoint {path!r} contains neither 'model' nor 'state_dict'"
            )
        model_state = {
            name.removeprefix(prefix): value.detach().cpu()
            for name, value in state.items()
            if name.startswith(prefix)
        }
        model = create_nats_model(architecture)
        model.load_state_dict(model_state, strict=True)
        return KDLightningModule(
            model=model,
            criterion=criterion,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs if optimizer_kwargs is not None else {},
            train_step_flops=train_step_flops,
            eval_step_flops=eval_step_flops,
            num_classes=(
                num_classes
                if num_classes is not None
                else int(architecture.get("num_classes", 10))
            ),
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            teacher_ensemble=teacher_ensemble,
            kd_loss=kd_loss,
            flops_tracker=flops_tracker,

        )
