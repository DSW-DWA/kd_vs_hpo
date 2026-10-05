from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
from omegaconf import OmegaConf
from scipy import stats

from kd_vs_hpo.common.dataloader import build_cifar10_dataloaders
from kd_vs_hpo.common.utils import (
    extract_logits,
    get_arch_by_idx,
    get_architectures_from_json,
    load_checkpoint,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (_REPO_ROOT / path)


def resolve_run_params(params_path: str | Path) -> dict:
    """Load a run's params.yaml and return the fully resolved config as a dict."""
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")
    cfg = OmegaConf.load(params_path)
    return OmegaConf.to_container(cfg, resolve=True)


def find_run_checkpoint(run_dir: str | Path) -> Path:
    """Return the (best) .ckpt checkpoint stored in a run directory."""
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt checkpoint found in {run_dir}")
    if len(checkpoints) == 1:
        return checkpoints[0]

    def _val_acc(name: str) -> float:
        match = re.search(r"val_acc=([0-9.]+)", name)
        return float(match.group(1)) if match else -1.0

    return max(checkpoints, key=lambda p: _val_acc(p.name))


def _load_kwargs_from_params(params: dict | None) -> dict:
    from torch import nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if params is None:
        return {}
    kd = params.get("kd", {})
    return {
        "criterion": nn.CrossEntropyLoss(),
        "optimizer_cls": SGD,
        "optimizer_kwargs": dict(kd.get("optimizer_params") or {}),
        "scheduler_cls": CosineAnnealingLR,
        "scheduler_kwargs": dict(kd.get("scheduler_params") or {}),
    }


def load_model(
    ckpt_path: str | Path,
    arch: dict,
    params: dict | None = None,
) -> torch.nn.Module:
    """Load a KDLightningModule.

    Prefers a full LightningModule load so the checkpoint's own hyper-parameters
    (optimizer, scheduler, num_classes, ...) are restored. Falls back to a raw
    state-dict load built from ``params`` (the model's own params.yaml).
    """
    from kd_vs_hpo.common.train_modules import KDLightningModule

    ckpt_path = Path(ckpt_path)
    try:
        return KDLightningModule.load_from_checkpoint(
            str(ckpt_path),
            weights_only=False,
        )
    except Exception as exc:  # raw state-dict checkpoint
        logger.warning(
            "Full LightningModule load failed for %s (%s); rebuilding from state dict",
            ckpt_path,
            exc,
        )
        return load_checkpoint(str(ckpt_path), arch, **_load_kwargs_from_params(params))


def score_on_validation(
    module: torch.nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Compute top-1 accuracy of a module on the given loader (eval mode)."""
    module.to(device)
    module.eval()
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=num_classes
    ).to(device)
    with torch.inference_mode():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = extract_logits(module(images))
            metric.update(logits, targets)
    return float(metric.compute())


def analyze_run(
    run_dir: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Analyze a single run directory.

    Reads ``params.yaml``, loads the student and all teachers (from
    ``kd.teachers_mapping``), scores every model on the run's validation split
    and returns each model's accuracy together with its full params.
    """
    run_dir = Path(run_dir)
    params_path = run_dir / "params.yaml"
    params = resolve_run_params(params_path)

    general = params["general"]
    kd = params["kd"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architectures = get_architectures_from_json(
        str(_resolve_path(general["architectures_path"]))
    )

    student_idx = kd["student"]
    student_arch = get_arch_by_idx(student_idx, architectures)
    student_ckpt = find_run_checkpoint(run_dir)
    student_module = load_model(student_ckpt, student_arch, params)

    num_classes = int(student_arch.get("num_classes", 10))

    checkpoint_dir = _resolve_path(kd["checkpoint_dir"])
    log_dir = _resolve_path(kd["log_dir"])
    data_root = _resolve_path(general["data_root"])

    _, _, test_loader, _, _, _ = build_cifar10_dataloaders(
        checkpoint_dir,
        log_dir,
        data_root,
        general["seed"],
        general["batch_size"],
        general["num_workers"],
        general["validation_fraction"],
        device,
    )

    student_test_acc = score_on_validation(
        student_module, test_loader, device, num_classes
    )
    result = {
        "run_dir": str(run_dir),
        "params_path": str(params_path),
        "student": {
            "arch_idx": student_idx,
            "checkpoint": str(student_ckpt),
            "test_acc": student_test_acc,
            "params": params,
        },
        "teachers": [],
    }

    teachers_mapping = kd.get("teachers_mapping")
    if teachers_mapping:
        for mapping in teachers_mapping:
            teacher_idx = mapping["idx"]
            teacher_ckpt = _resolve_path(mapping["path"])
            teacher_arch = get_arch_by_idx(teacher_idx, architectures)

            teacher_params_path = teacher_ckpt.parent / "params.yaml"
            teacher_params = (
                resolve_run_params(teacher_params_path)
                if teacher_params_path.exists()
                else None
            )
            teacher_module = load_model(
                teacher_ckpt, teacher_arch, teacher_params
            )
            teacher_num_classes = int(
                teacher_arch.get("num_classes", num_classes)
            )
            teacher_test_acc = score_on_validation(
                teacher_module, test_loader, device, teacher_num_classes
            )
            result["teachers"].append(
                {
                    "arch_idx": teacher_idx,
                    "checkpoint": str(teacher_ckpt),
                    "test_acc": teacher_test_acc,
                    "params": teacher_params or params,
                }
            )

    return result


def analyze_checkpoint_dir(
    base_dir: str | Path,
    device: torch.device | None = None,
) -> list[dict]:
    """Analyze every run directory (a dir containing params.yaml) under base_dir."""
    base_dir = Path(base_dir)
    if (base_dir / "params.yaml").exists():
        run_dirs = [base_dir]
    else:
        run_dirs = [
            d for d in base_dir.iterdir()
            if d.is_dir() and (d / "params.yaml").exists()
        ]

    results = []
    for run_dir in sorted(run_dirs):
        try:
            results.append(analyze_run(run_dir, device=device))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to analyze %s: %s", run_dir, exc)
    return results


def summarize(results: list[dict]) -> list[dict]:
    """Flatten analyze results into a row per model (student + teachers)."""
    rows = []
    for res in results:
        student = res["student"]
        rows.append(
            {
                "run_dir": res["run_dir"],
                "role": "student",
                "arch_idx": student["arch_idx"],
                "checkpoint": student["checkpoint"],
                "test_acc": student["test_acc"],
                "params": student["params"],
            }
        )
        for teacher in res["teachers"]:
            rows.append(
                {
                    "run_dir": res["run_dir"],
                    "role": "teacher",
                    "arch_idx": teacher["arch_idx"],
                    "checkpoint": teacher["checkpoint"],
                    "test_acc": teacher["test_acc"],
                    "params": teacher["params"],
                }
            )
    return rows


def _flatten_params(prefix: tuple, params: dict):
    """Yield ``(path_tuple, value)`` for every leaf of a nested params dict."""
    for key, value in params.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            yield from _flatten_params(path, value)
        else:
            yield path, value


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from analyze results (list of row dicts).

    Scalar fields (``run_dir``, ``role``, ``arch_idx``, ``checkpoint``,
    ``test_acc``, ...) become single-level columns. The nested ``params`` dict is
    expanded recursively so that its structure is reflected as a MultiIndex column
    group::

        run_dir        params
                        general            hpo
                        max_epochs  ...    n_trials  ...

    If ``results`` contains raw ``analyze_run`` records (with a ``student`` key),
    they are flattened via :func:`summarize` first. Nested non-scalar leaves
    (dicts/lists such as ``search_space``) are serialized to JSON strings.
    """
    if results and "student" in results[0]:
        results = summarize(results)
    if not results:
        return pd.DataFrame()

    def _cell(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        return value

    rows = []
    max_depth = 1
    for res in results:
        record = {}
        for key, value in res.items():
            if key == "params" and isinstance(value, dict):
                for path, val in _flatten_params(("params",), value):
                    record[path] = val
                    max_depth = max(max_depth, len(path))
            else:
                record[(key,)] = value
        rows.append(record)

    if max_depth == 1:
        return pd.DataFrame([{k[0]: v for k, v in r.items()} for r in rows])

    records = [
        {
            path + ("",) * (max_depth - len(path)): _cell(value)
            for path, value in record.items()
        }
        for record in rows
    ]
    df = pd.DataFrame(records)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def group_confidence_interval(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute a Student's t confidence interval for each group of a grouped series.

    ``grouped`` must already be grouped (a ``SeriesGroupBy``), e.g.::

        group_confidence_interval(
            results_to_dataframe(summary).groupby('arch_idx')['test_acc']
        )

    Returns a DataFrame indexed by group with ``n``, ``mean``, ``std``, ``sem``,
    ``ci_low`` and ``ci_high`` (``mean +/- t_{1-alpha/2, n-1} * sem``).
    """

    def _stats(values: pd.Series) -> pd.Series:
        vals = np.asarray(values, dtype=float)
        vals = vals[~np.isnan(vals)]
        n = int(vals.size)
        if n == 0:
            return pd.Series(
                {
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "sem": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
        mean = float(vals.mean())
        min_ = float(vals.min())
        max_ = float(vals.max())
        if n == 1:
            return pd.Series(
                {
                    "n": n,
                    "mean": mean,
                    "std": 0.0,
                    "sem": 0.0,
                    "ci_low": min_,
                    "ci_high": max_,
                    "min": min,
                    "max": max,
                }
            )
        std = float(vals.std(ddof=1))
        sem = std / float(np.sqrt(n))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        min_ = float(vals.min())
        max_ = float(vals.max())
        return pd.Series(
            {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_low": mean - t_crit * sem,
                "ci_high": mean + t_crit * sem,
                "min": min_,
                "max": max_,
            }
        )

    result = grouped.apply(_stats).unstack()
    if "n" in result:
        result["n"] = result["n"].astype("Int64")
    return result


def main(base_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = analyze_checkpoint_dir(base_dir, device=device)
    summary = summarize(results)

    if not summary:
        print(f"No run directories with params.yaml found under {base_dir}")
        return

    print(f"{'run_dir':<45} {'role':<8} {'arch_idx':>8} {'val_acc':>8}")
    print("-" * 75)
    for row in summary:
        print(
            f"{row['run_dir']:<45} {row['role']:<8} "
            f"{row['arch_idx']:>8} {row['val_acc']:>8.4f}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m kd_vs_hpo.common.analyze <run_dir_or_checkpoint_dir>")
    main(sys.argv[1])