import logging
import math
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.hpo.asha import ASHAPlan, TrialConfig, make_plan, sample_trials
from src.hpo.config import HPOExperimentConfig
from src.hpo.training import run_training_stage
from src.hpo.workers import ParallelStageRunner, TrainingStageTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledStage:
    trial: TrialConfig
    incremental_epochs: int
    train_flops: int
    stage_flops: int
    cumulative_flops: int
    cumulative_train_flops: int
    cumulative_validation_flops: int
    future: Future[dict[str, Any]] | None


def _stage_record(
    scheduled: ScheduledStage,
    stage: dict[str, Any],
    *,
    arch_record: dict[str, Any],
    cost: dict[str, Any],
    plan: ASHAPlan,
    rung: int,
    target_epochs: int,
    forward_flops: int,
    epoch_flops: int,
    validation_flops: int,
    budget_flops: int,
) -> dict[str, Any]:
    return {
        "arch_row": arch_record["arch_row"],
        "arch_index": arch_record["arch_index"],
        "arch_str": arch_record["arch_str"],
        "dataset": arch_record.get("dataset"),
        **asdict(scheduled.trial),
        "rung": rung,
        "target_epochs": target_epochs,
        "incremental_epochs": scheduled.incremental_epochs,
        "train_flops": scheduled.train_flops,
        "validation_flops": validation_flops,
        "stage_flops": scheduled.stage_flops,
        "cumulative_flops": scheduled.cumulative_flops,
        "cumulative_train_flops": scheduled.cumulative_train_flops,
        "cumulative_validation_flops": scheduled.cumulative_validation_flops,
        "val_acc1": float(stage["val_acc1"]),
        "checkpoint_path": stage["checkpoint_path"],
        "status": "completed",
        "asha_min_epochs": plan.min_epochs,
        "asha_max_epochs": plan.max_epochs,
        "asha_reduction_factor": plan.reduction_factor,
        "asha_num_initial_configs": plan.num_initial_configs,
        "asha_planned_flops": plan.planned_flops,
        "forward_flops_per_sample": forward_flops,
        "epoch_flops": epoch_flops,
        "budget_flops": budget_flops,
        "params": cost.get("params", np.nan),
        "latency": cost.get("latency", np.nan),
    }


def run_architecture(
    *,
    arch_record: dict[str, Any],
    cost: dict[str, Any],
    experiment: HPOExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train: int,
    n_val: int,
    device: torch.device,
    parallel_runner: ParallelStageRunner | None,
) -> pd.DataFrame:
    forward_flops = int(float(cost["forward_flops_per_sample"]))
    epoch_flops = int(experiment.train.train_step_multiplier * forward_flops * n_train)
    validation_flops = forward_flops * n_val
    budget_flops = experiment.asha.budget_flops_per_arch
    plan = make_plan(experiment.asha, epoch_flops, validation_flops)
    seed = experiment.train.seed + arch_record["arch_row"]
    trials = sample_trials(plan.num_initial_configs, experiment.search_space, seed)
    completed_epochs = {trial.trial_id: 0 for trial in trials}
    alive = trials
    spent_flops = spent_train_flops = spent_validation_flops = 0
    records: list[dict[str, Any]] = []

    logger.info(
        "Starting architecture row=%s index=%s: initial_configs=%s rungs=%s planned_flops=%s",
        arch_record["arch_row"], arch_record["arch_index"],
        plan.num_initial_configs, plan.rungs, plan.planned_flops,
    )

    for rung, target_epochs in enumerate(plan.rungs):
        scheduled_stages: list[ScheduledStage] = []
        for trial in alive:
            incremental_epochs = target_epochs - completed_epochs[trial.trial_id]
            train_flops = incremental_epochs * epoch_flops
            stage_flops = train_flops + validation_flops
            if spent_flops + stage_flops > budget_flops:
                continue
            spent_flops += stage_flops
            spent_train_flops += train_flops
            spent_validation_flops += validation_flops
            task = TrainingStageTask(
                trial, target_epochs, plan, seed + trial.trial_id,
                arch_record, experiment,
            )
            scheduled_stages.append(
                ScheduledStage(
                    trial, incremental_epochs, train_flops, stage_flops,
                    spent_flops, spent_train_flops, spent_validation_flops,
                    parallel_runner.submit(task) if parallel_runner else None,
                )
            )

        rung_records: list[dict[str, Any]] = []
        for scheduled in scheduled_stages:
            stage = scheduled.future.result() if scheduled.future else run_training_stage(
                trial=scheduled.trial,
                target_epochs=target_epochs,
                plan=plan,
                seed=seed + scheduled.trial.trial_id,
                arch_record=arch_record,
                experiment=experiment,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
            completed_epochs[scheduled.trial.trial_id] = target_epochs
            record = _stage_record(
                scheduled, stage, arch_record=arch_record, cost=cost, plan=plan,
                rung=rung, target_epochs=target_epochs, forward_flops=forward_flops,
                epoch_flops=epoch_flops, validation_flops=validation_flops,
                budget_flops=budget_flops,
            )
            records.append(record)
            rung_records.append(record)

        if not rung_records:
            break
        promoted_count = max(1, math.ceil(len(rung_records) / plan.reduction_factor))
        promoted_ids = {
            row["trial_id"]
            for row in sorted(rung_records, key=lambda row: row["val_acc1"], reverse=True)[
                :promoted_count
            ]
        }
        alive = [trial for trial in alive if trial.trial_id in promoted_ids]
        logger.info(
            "arch=%s rung=%s completed=%s promoted_trials=%s",
            arch_record["arch_index"], rung, len(rung_records), sorted(promoted_ids),
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result["best_val_acc1_so_far"] = result["val_acc1"].cummax()
    return result
