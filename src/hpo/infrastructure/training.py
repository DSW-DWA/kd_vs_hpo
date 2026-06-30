import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.common.utils import accuracy_top1_from_logits, extract_logits


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    scaler: GradScaler,
    grad_clip_norm: float | None,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            enabled=scaler.is_enabled() and device.type == "cuda",
        ):
            loss = criterion(extract_logits(model(images)), targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        correct += accuracy_top1_from_logits(extract_logits(model(images)), targets)
        total += int(targets.size(0))
    return 100.0 * correct / max(total, 1)
