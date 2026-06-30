import torch
import torch.nn as nn


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    schedule_max_epochs: int,
    momentum: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, schedule_max_epochs),
    )
    return optimizer, scheduler
