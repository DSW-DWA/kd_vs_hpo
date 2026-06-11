from dataclasses import dataclass

import torch
import torch.nn as nn
from calflops import calculate_flops


def count_flops_params(model: nn.Module, input_shape=(1, 3, 32, 32), device="cuda"):
    model = model.to(device).eval()
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        print_results=False,
    )

    return int(flops), int(params)


@dataclass
class FlopsBudgetTracker:
    budget: int
    spent: int = 0

    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def can_spend(self, flops: int) -> bool:
        return self.spent + int(flops) <= self.budget

    def spend(self, flops: int) -> None:
        self.spent += int(flops)


def estimate_batch_flops(
    model: nn.Module,
    batch_size: int,
    train: bool,
    train_step_multiplier: float,
    device: torch.device,
) -> int:
    forward_flops_per_sample, _ = count_flops_params(model, input_shape=(1, 3, 32, 32), device=str(device))
    forward_batch_flops = int(forward_flops_per_sample * batch_size)
    if train:
        return int(forward_batch_flops * train_step_multiplier)
    return forward_batch_flops
