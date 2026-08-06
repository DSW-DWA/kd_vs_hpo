from dataclasses import dataclass
from enum import StrEnum

import torch
from calflops import calculate_flops
from torch import nn


class CounterMode(StrEnum):
    SILENT = "silent"
    OFF = "off"
    ON = "on"

@torch.inference_mode()
def count_flops_params(model: nn.Module, input_shape=(1, 3, 32, 32)):
    flops, _, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        print_results=False,
    )
    return int(flops), int(params)


@dataclass
class FlopsBudgetTracker:
    budget: int
    mode: CounterMode
    spent: int = 0

    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def can_spend(self, flops: int) -> bool:
        return self.spent + int(flops) <= self.budget

    def spend(self, flops: int) -> None:
        if self.mode != CounterMode.OFF:
            self.spent += int(flops)

    def reset(self):
        self.spent = 0


def estimate_batch_flops(
    model: nn.Module,
    batch_size: int,
    train: bool,
    train_step_multiplier: float,
    device: torch.device,
) -> int:
    forward_flops_per_sample, _ = count_flops_params(
        model,
        input_shape=(1, 3, 32, 32),
        device=str(device),
    )
    forward_batch_flops = int(forward_flops_per_sample * batch_size)
    if train:
        return int(forward_batch_flops * train_step_multiplier)
    return forward_batch_flops
