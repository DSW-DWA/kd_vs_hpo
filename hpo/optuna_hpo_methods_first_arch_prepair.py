# %%
# Converted from optuna_hpo_methods_first_arch_prepair.ipynb

# %% [markdown]
# # Optuna HPO Methods Under FLOPs Budget
#
# This notebook compares several Optuna HPO methods on one fixed NATS-Bench architecture under the same FLOPs budget per method.

# %%
# %pip install -q --upgrade pip setuptools wheel
# %pip install -q torch torchvision pandas plotly numpy optuna cmaes
# %pip install -q xautodl --no-deps

# %%
from __future__ import annotations

import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import torch
from optuna.trial import TrialState
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from xautodl.config_utils import dict2config
from xautodl.models import get_cell_based_tiny_net

try:
    from IPython.display import display
except ImportError:
    def display(value):
        if hasattr(value, "to_string"):
            print(value.to_string())
        else:
            print(value)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## Settings

# %%
def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "sampled_architectures_10.jsonl").exists():
            return candidate
        if (candidate / "hpo" / "sampled_architectures_10.jsonl").exists():
            return candidate / "hpo"
    raise FileNotFoundError("Could not find sampled_architectures_10.jsonl")


REPO_ROOT = find_repo_root()
ARCHITECTURES_PATH = REPO_ROOT / "sampled_architectures_10.jsonl"
COSTS_CSV_PATH = REPO_ROOT / "sampled_architecture_costs.csv"
OUTPUT_DIR = REPO_ROOT / "hpo_output" / "optuna_methods"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# This notebook compares methods on one architecture only.
ARCH_ROW = 0

# Every Optuna method gets this separate HPO budget.
BUDGET_FLOPS_PER_METHOD = 1.0e15

# Search space.
LR_RANGE = (1e-3, 3e-1)
WEIGHT_DECAY_RANGE = (1e-6, 1e-3)

# Training / pruning schedule.
MIN_EPOCHS = 3
REDUCTION_FACTOR = 3
MAX_EPOCHS_CAP = 81
MAX_TRIALS_PER_METHOD = 1000

# Fixed training recipe.
BATCH_SIZE = 256
NUM_WORKERS = 2
VALIDATION_FRACTION = 0.1
MOMENTUM = 0.9
GRAD_CLIP_NORM: float | None = 5.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
USE_AMP = torch.cuda.is_available()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
log(f"Repo root: {REPO_ROOT}")
log(f"Device: {DEVICE}")
log(f"AMP device: {AMP_DEVICE}")
log(f"AMP enabled: {USE_AMP}")
log(
    "Run settings: "
    f"ARCH_ROW={ARCH_ROW}, BUDGET_FLOPS_PER_METHOD={BUDGET_FLOPS_PER_METHOD:.3e}, "
    f"MIN_EPOCHS={MIN_EPOCHS}, MAX_EPOCHS_CAP={MAX_EPOCHS_CAP}, "
    f"BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}"
)

# %% [markdown]
# ## Inputs

# %%
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            record = json.loads(line)
            record["arch_row"] = row_index
            record["arch_index"] = int(record["arch_index"])
            records.append(record)
    return records


def load_costs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    costs = pd.read_csv(path)
    required = {"arch_row", "arch_index", "dataset", "forward_flops_per_sample"}
    missing = required - set(costs.columns)
    if missing:
        raise ValueError(f"Costs CSV is missing columns: {sorted(missing)}")
    return costs


arch_records = load_jsonl(ARCHITECTURES_PATH)
arch_record = arch_records[ARCH_ROW]
costs_df = load_costs(COSTS_CSV_PATH)
cost_row = costs_df.set_index("arch_row").loc[ARCH_ROW].to_dict()

display(pd.DataFrame([arch_record]))
display(pd.DataFrame([{**{"arch_row": ARCH_ROW}, **cost_row}]))
log(
    "Loaded architecture: "
    f"arch_row={ARCH_ROW}, arch_index={arch_record['arch_index']}, "
    f"dataset={arch_record.get('dataset')}, search_space={arch_record.get('search_space')}"
)

# %% [markdown]
# ## Dataset

# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


normalize_kwargs = {
    "mean": [0.49139968, 0.48215841, 0.44653091],
    "std": [0.24703223, 0.24348513, 0.26158784],
}

train_transform = T.Compose([
    T.RandomCrop(size=32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(**normalize_kwargs),
])
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(**normalize_kwargs),
])

data_root = REPO_ROOT / "data"
train_aug_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
train_eval_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_transform)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(train_aug_dataset), generator=generator).tolist()
n_val = int(round(len(indices) * VALIDATION_FRACTION))
val_indices = indices[:n_val]
train_indices = indices[n_val:]
N_TRAIN_EXAMPLES = len(train_indices)
N_VALIDATION_EXAMPLES = len(val_indices)
N_TEST_EXAMPLES = len(test_dataset)

loader_kwargs = {
    "num_workers": NUM_WORKERS,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": NUM_WORKERS > 0,
}
if NUM_WORKERS > 0:
    loader_kwargs["prefetch_factor"] = 4

train_loader = DataLoader(Subset(train_aug_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
val_loader = DataLoader(Subset(train_eval_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

log(f"Train examples: {N_TRAIN_EXAMPLES}")
log(f"Validation examples: {N_VALIDATION_EXAMPLES}")
log(f"Test examples: {N_TEST_EXAMPLES}")

# %% [markdown]
# ## FLOPs Accounting

# %%
forward_flops_per_sample = int(float(cost_row["forward_flops_per_sample"]))
epoch_flops = int(3 * forward_flops_per_sample * N_TRAIN_EXAMPLES)
validation_flops = int(forward_flops_per_sample * N_VALIDATION_EXAMPLES)
test_flops = int(forward_flops_per_sample * N_TEST_EXAMPLES)

flops_info = {
    "arch_row": ARCH_ROW,
    "arch_index": arch_record["arch_index"],
    "forward_flops_per_sample": forward_flops_per_sample,
    "epoch_flops": epoch_flops,
    "validation_flops": validation_flops,
    "test_flops": test_flops,
    "budget_flops_per_method": BUDGET_FLOPS_PER_METHOD,
}
display(pd.DataFrame([flops_info]))
log(
    "FLOPs accounting: "
    f"forward_per_sample={forward_flops_per_sample:.3e}, "
    f"epoch_flops={epoch_flops:.3e}, "
    f"validation_flops={validation_flops:.3e}, "
    f"test_flops={test_flops:.3e}"
)

# %% [markdown]
# ## Model And Training Helpers

# %%
def create_nats_model(arch_str: str, num_classes: int = 10) -> nn.Module:
    config = dict2config({"name": "infer.tiny", "C": 16, "N": 5, "arch_str": arch_str, "num_classes": num_classes}, None)
    return get_cell_based_tiny_net(config)


def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        tensors = [item for item in model_output if torch.is_tensor(item)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Model output does not contain logits tensor: {type(model_output)!r}")


def create_optimizer_and_scheduler(model: nn.Module, lr: float, weight_decay: float, schedule_max_epochs: int):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_max_epochs)
    return optimizer, scheduler


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).sum().item())


def train_one_epoch(model: nn.Module, optimizer, criterion, loader: DataLoader, scaler) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            logits = extract_logits(model(images))
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.inference_mode()
def evaluate_top1(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_correct = 0.0
    total_examples = 0
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(AMP_DEVICE, enabled=USE_AMP):
            logits = extract_logits(model(images))
        total_correct += accuracy_top1(logits, targets)
        total_examples += int(targets.size(0))
    return 100.0 * total_correct / max(1, total_examples)

# %% [markdown]
# ## Optuna Methods

# %%
def report_epochs(min_epochs: int, max_epochs: int, reduction_factor: int) -> list[int]:
    epochs = []
    current = min_epochs
    while current < max_epochs:
        epochs.append(current)
        current *= reduction_factor
    epochs.append(max_epochs)
    return sorted(set(epochs))


REPORT_EPOCHS = report_epochs(MIN_EPOCHS, MAX_EPOCHS_CAP, REDUCTION_FACTOR)

OPTUNA_METHOD_SPECS = [
    {"method_name": "random_successive_halving", "sampler": "random", "pruner": "successive_halving"},
    {"method_name": "tpe_successive_halving", "sampler": "tpe", "pruner": "successive_halving"},
    {"method_name": "tpe_hyperband", "sampler": "tpe", "pruner": "hyperband"},
    {"method_name": "gp_successive_halving", "sampler": "gp", "pruner": "successive_halving"},
    {"method_name": "gp_hyperband", "sampler": "gp", "pruner": "hyperband"},
]


def make_sampler(name: str, seed: int):
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "gp":
        if not hasattr(optuna.samplers, "GPSampler"):
            raise RuntimeError("GPSampler is unavailable. Upgrade Optuna to a version that provides optuna.samplers.GPSampler.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="GPSampler is experimental.*")
            return optuna.samplers.GPSampler(seed=seed)
    raise ValueError(name)


def make_pruner(name: str):
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=MIN_EPOCHS, interval_steps=MIN_EPOCHS)
    if name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=MIN_EPOCHS, reduction_factor=REDUCTION_FACTOR)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=MIN_EPOCHS,
            max_resource=MAX_EPOCHS_CAP,
            reduction_factor=REDUCTION_FACTOR,
        )
    raise ValueError(name)


display(pd.DataFrame(OPTUNA_METHOD_SPECS))
log(f"Optuna methods: {[spec['method_name'] for spec in OPTUNA_METHOD_SPECS]}")
log(f"Report epochs: {REPORT_EPOCHS}")

# %% [markdown]
# ## FLOPs-Budgeted Optuna Runner

# %%
@dataclass
class BudgetTracker:
    budget_flops: float
    spent_flops: float = 0.0
    spent_train_flops: float = 0.0
    spent_validation_flops: float = 0.0

    def can_spend(self, flops: float) -> bool:
        return self.spent_flops + flops <= self.budget_flops

    def spend(self, train_flops: float, val_flops: float) -> None:
        self.spent_train_flops += train_flops
        self.spent_validation_flops += val_flops
        self.spent_flops += train_flops + val_flops


def method_checkpoint_path(method_name: str, trial_number: int, target_epochs: int) -> Path:
    return CHECKPOINT_DIR / f"{method_name}_trial_{trial_number:04d}_epoch_{target_epochs:04d}.pt"


def load_model_from_checkpoint(checkpoint_file: str | Path) -> nn.Module:
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model = create_nats_model(checkpoint["arch_record"]["arch_str"]).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    return model


def run_one_optuna_trial(method_name: str, trial, tracker: BudgetTracker, method_seed: int) -> tuple[str, float | None, list[dict[str, Any]]]:
    lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAY_RANGE[0], WEIGHT_DECAY_RANGE[1], log=True)

    log(
        f"{method_name} trial={trial.number} START "
        f"lr={lr:.3e} weight_decay={weight_decay:.3e} "
        f"spent={tracker.spent_flops:.3e}/{tracker.budget_flops:.3e}"
    )

    set_seed(method_seed + trial.number)
    model = create_nats_model(arch_record["arch_str"]).to(DEVICE)
    optimizer, scheduler = create_optimizer_and_scheduler(model, lr, weight_decay, REPORT_EPOCHS[-1])
    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    records = []
    completed_epochs = 0
    last_val_acc1: float | None = None
    last_checkpoint_path: str | None = None
    started_at = time.time()

    for rung, target_epochs in enumerate(REPORT_EPOCHS):
        incremental_epochs = target_epochs - completed_epochs
        if incremental_epochs <= 0:
            continue
        train_stage_flops = incremental_epochs * epoch_flops
        stage_flops = train_stage_flops + validation_flops
        if not tracker.can_spend(stage_flops):
            if last_val_acc1 is None:
                log(
                    f"{method_name} trial={trial.number} SKIP budget exhausted before first stage: "
                    f"needed={stage_flops:.3e}, remaining={tracker.budget_flops - tracker.spent_flops:.3e}"
                )
                return "budget_exhausted", None, records
            log(
                f"{method_name} trial={trial.number} STOP budget exhausted before epoch={target_epochs}: "
                f"needed={stage_flops:.3e}, remaining={tracker.budget_flops - tracker.spent_flops:.3e}, "
                f"last_val_acc1={last_val_acc1:.2f}"
            )
            return "complete_budget_exhausted", last_val_acc1, records

        log(
            f"{method_name} trial={trial.number} rung={rung} TRAIN "
            f"from_epoch={completed_epochs} to_epoch={target_epochs} "
            f"stage_flops={stage_flops:.3e}"
        )

        for local_epoch in range(incremental_epochs):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, scaler)
            scheduler.step()
            current_epoch = completed_epochs + local_epoch + 1
            log(
                f"{method_name} trial={trial.number} epoch={current_epoch}/{target_epochs} "
                f"loss={train_loss:.4f} lr={optimizer.param_groups[0]['lr']:.3e}"
            )
        completed_epochs = target_epochs
        val_acc1 = evaluate_top1(model, val_loader)
        trial.report(val_acc1, step=target_epochs)
        tracker.spend(train_stage_flops, validation_flops)

        checkpoint_file = method_checkpoint_path(method_name, trial.number, target_epochs)
        torch.save(
            {
                "method_name": method_name,
                "trial_number": trial.number,
                "epoch": target_epochs,
                "model": model.state_dict(),
                "val_acc1": val_acc1,
                "lr": lr,
                "weight_decay": weight_decay,
                "arch_record": arch_record,
            },
            checkpoint_file,
        )
        last_val_acc1 = val_acc1
        last_checkpoint_path = str(checkpoint_file)

        records.append(
            {
                "method_name": method_name,
                "trial_number": trial.number,
                "rung": rung,
                "target_epochs": target_epochs,
                "incremental_epochs": incremental_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "train_flops": train_stage_flops,
                "validation_flops": validation_flops,
                "stage_flops": stage_flops,
                "cumulative_flops": tracker.spent_flops,
                "cumulative_train_flops": tracker.spent_train_flops,
                "cumulative_validation_flops": tracker.spent_validation_flops,
                "val_acc1": val_acc1,
                "checkpoint_path": last_checkpoint_path,
                "elapsed_min": (time.time() - started_at) / 60,
                "status": "completed_stage",
            }
        )

        log(
            f"{method_name} trial={trial.number} epoch={target_epochs} "
            f"val_acc1={val_acc1:.2f} spent={tracker.spent_flops:.3e}/{tracker.budget_flops:.3e}"
        )

        if trial.should_prune():
            records[-1]["status"] = "pruned"
            log(f"{method_name} trial={trial.number} PRUNED at epoch={target_epochs} val_acc1={val_acc1:.2f}")
            return "pruned", val_acc1, records

    log(f"{method_name} trial={trial.number} COMPLETE best_or_last_val_acc1={last_val_acc1:.2f}")
    return "complete", last_val_acc1, records


def run_optuna_method(spec: dict[str, str], method_index: int) -> tuple[pd.DataFrame, optuna.Study]:
    method_name = spec["method_name"]
    method_seed = SEED + 1000 * method_index
    log(
        f"{method_name} METHOD START sampler={spec['sampler']} pruner={spec['pruner']} "
        f"budget={BUDGET_FLOPS_PER_METHOD:.3e}"
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=make_sampler(spec["sampler"], method_seed),
        pruner=make_pruner(spec["pruner"]),
        study_name=method_name,
    )
    tracker = BudgetTracker(BUDGET_FLOPS_PER_METHOD)
    all_records = []

    for _ in range(MAX_TRIALS_PER_METHOD):
        min_stage_flops = MIN_EPOCHS * epoch_flops + validation_flops
        if not tracker.can_spend(min_stage_flops):
            log(
                f"{method_name} METHOD STOP budget exhausted before next trial: "
                f"remaining={tracker.budget_flops - tracker.spent_flops:.3e}, "
                f"min_stage_flops={min_stage_flops:.3e}"
            )
            break
        trial = study.ask()
        status, value, records = run_one_optuna_trial(method_name, trial, tracker, method_seed)
        all_records.extend(records)

        if status == "budget_exhausted":
            study.tell(trial, state=TrialState.FAIL)
            break
        if value is None:
            study.tell(trial, state=TrialState.FAIL)
        elif status == "pruned":
            study.tell(trial, state=TrialState.PRUNED)
        else:
            study.tell(trial, value)
        log(f"{method_name} trial={trial.number} TOLD status={status} value={value}")

    method_df = pd.DataFrame(all_records)
    if not method_df.empty:
        method_df["sampler"] = spec["sampler"]
        method_df["pruner"] = spec["pruner"]
        method_df["budget_flops"] = BUDGET_FLOPS_PER_METHOD
        method_df["best_val_acc1_so_far"] = method_df["val_acc1"].cummax()
    best_value = method_df["val_acc1"].max() if not method_df.empty else float("nan")
    log(
        f"{method_name} METHOD END stages={len(method_df)} "
        f"spent={tracker.spent_flops:.3e}/{tracker.budget_flops:.3e} "
        f"best_val_acc1={best_value:.2f}"
    )
    return method_df, study

# %% [markdown]
# ## Run Comparison

# %%
method_results = []
studies = {}

for method_index, spec in enumerate(OPTUNA_METHOD_SPECS):
    log(f"=== Running {spec['method_name']} ===")
    method_df, study = run_optuna_method(spec, method_index)
    method_results.append(method_df)
    studies[spec["method_name"]] = study

all_results = pd.concat(method_results, ignore_index=True) if method_results else pd.DataFrame()
display(all_results)
log(f"All method stages collected: rows={len(all_results)}")

# %% [markdown]
# ## Summary And Test Accuracy

# %%
if all_results.empty:
    method_summary = pd.DataFrame()
else:
    completed = all_results.copy()
    method_summary = (
        completed.sort_values(["method_name", "val_acc1"])
        .groupby("method_name", as_index=False)
        .tail(1)
        .sort_values("val_acc1", ascending=False)
        .copy()
    )
    spent = completed.groupby("method_name")["stage_flops"].sum()
    train_spent = completed.groupby("method_name")["train_flops"].sum()
    val_spent = completed.groupby("method_name")["validation_flops"].sum()
    stages = completed.groupby("method_name").size()
    trials = completed.groupby("method_name")["trial_number"].nunique()
    method_summary["spent_flops"] = method_summary["method_name"].map(spent)
    method_summary["spent_train_flops"] = method_summary["method_name"].map(train_spent)
    method_summary["spent_validation_flops"] = method_summary["method_name"].map(val_spent)
    method_summary["spent_budget_ratio"] = method_summary["spent_flops"] / BUDGET_FLOPS_PER_METHOD
    method_summary["completed_stages"] = method_summary["method_name"].map(stages)
    method_summary["evaluated_trials"] = method_summary["method_name"].map(trials)

    test_rows = []
    for _, row in method_summary.iterrows():
        checkpoint_file = row["checkpoint_path"]
        log(f"Evaluating test accuracy for {row['method_name']} from checkpoint={checkpoint_file}")
        model = load_model_from_checkpoint(checkpoint_file)
        test_acc1 = evaluate_top1(model, test_loader)
        test_rows.append({"method_name": row["method_name"], "test_acc1": test_acc1, "test_flops": test_flops})
        log(f"{row['method_name']} test_acc1={test_acc1:.2f}")
    test_df = pd.DataFrame(test_rows).set_index("method_name")
    method_summary["test_acc1"] = method_summary["method_name"].map(test_df["test_acc1"])
    method_summary["test_flops"] = method_summary["method_name"].map(test_df["test_flops"])

display(method_summary)
log(f"Summary rows: {len(method_summary)}")

results_path = OUTPUT_DIR / "optuna_hpo_method_results.csv"
summary_path = OUTPUT_DIR / "optuna_hpo_method_summary.csv"
all_results.to_csv(results_path, index=False)
method_summary.to_csv(summary_path, index=False)
log(f"Saved results to {results_path}")
log(f"Saved summary to {summary_path}")

# %% [markdown]
# ## Plots

# %%
if not all_results.empty:
    plot_df = all_results.copy()
    plot_df["budget_used_pct"] = 100.0 * plot_df["cumulative_flops"] / plot_df["budget_flops"]
    plot_df["trial_label"] = "trial " + plot_df["trial_number"].astype(str)

    fig_progress = px.line(
        plot_df.sort_values(["method_name", "cumulative_flops"]),
        x="budget_used_pct",
        y="best_val_acc1_so_far",
        color="method_name",
        markers=True,
        hover_data=["trial_number", "target_epochs", "lr", "weight_decay", "val_acc1", "cumulative_flops"],
        title="Best validation accuracy vs used FLOPs budget",
    )
    fig_progress.update_xaxes(title="Used method budget, %")
    fig_progress.update_yaxes(title="Best validation Acc@1 so far")
    fig_progress.show()

    fig_stages = px.scatter(
        plot_df,
        x="target_epochs",
        y="val_acc1",
        color="method_name",
        symbol="rung",
        size="stage_flops",
        hover_data=["trial_number", "lr", "weight_decay", "budget_used_pct"],
        title="All evaluated stages by Optuna method",
    )
    fig_stages.update_xaxes(title="Target epochs")
    fig_stages.update_yaxes(title="Validation Acc@1")
    fig_stages.show()

if not method_summary.empty:
    fig_summary = px.bar(
        method_summary.sort_values("val_acc1", ascending=False),
        x="method_name",
        y="val_acc1",
        color="test_acc1",
        text="val_acc1",
        hover_data=["test_acc1", "evaluated_trials", "completed_stages", "spent_budget_ratio", "lr", "weight_decay"],
        title="Best validation result by Optuna method",
        color_continuous_scale="Viridis",
    )
    fig_summary.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_summary.update_xaxes(title="Optuna method")
    fig_summary.update_yaxes(title="Best validation Acc@1", rangemode="tozero")
    fig_summary.show()

