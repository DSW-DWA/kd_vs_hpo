from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchmetrics
from omegaconf import OmegaConf
from scipy import stats

from kd_vs_hpo.common.dataloader import build_dataloaders, get_num_classes
from kd_vs_hpo.common.utils import (
    extract_logits,
    get_arch_by_idx,
    get_architectures_from_json,
    load_checkpoint,
)
from kd_vs_hpo.kd.teacher import TeacherEnsemble

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelResult:
    """A single scored model (the student or one teacher) inside a run."""

    role: Literal["student", "teacher"]
    arch_idx: int
    seed: int | None
    checkpoint: Path
    test_acc: float
    params: dict
    num_classes: int
    spent_flops: int | None = None


@dataclass(frozen=True)
class EnsembleResult:
    """Accuracy of the teacher ensemble (only present when it existed)."""

    teacher_indices: tuple[int, ...]
    test_acc: float


@dataclass(frozen=True)
class RunResult:
    """Result of one experimental run: the student, its teachers and the ensemble."""

    run_dir: Path
    params_path: Path
    params: dict
    student: ModelResult
    teachers: list[ModelResult]
    ensemble: EnsembleResult | None

    @property
    def student_test_acc(self) -> float:
        return self.student.test_acc

    @property
    def teacher_test_accs(self) -> list[float]:
        return [t.test_acc for t in self.teachers]

    @property
    def ensemble_test_acc(self) -> float | None:
        return self.ensemble.test_acc if self.ensemble else None

    @property
    def n_teachers(self) -> int:
        return len(self.teachers)

    @property
    def seed(self) -> int | None:
        return self.student.seed

    @property
    def teacher_seeds(self) -> list[int | None]:
        return [t.seed for t in self.teachers]

    @property
    def dataset(self) -> str | None:
        return self.params.get("general", {}).get("dataset")

    @property
    def label_noise(self) -> float | None:
        return self.params.get("general", {}).get("label_noise")

    @property
    def spent_flops(self) -> int | None:
        """Cumulative FLOPs spent training this run's student."""
        return self.student.spent_flops

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "params_path": str(self.params_path),
            "params": self.params,
            "student": _model_to_dict(self.student),
            "teachers": [_model_to_dict(t) for t in self.teachers],
            "ensemble": (
                {
                    "teacher_indices": list(self.ensemble.teacher_indices),
                    "test_acc": self.ensemble.test_acc,
                }
                if self.ensemble
                else None
            ),
            "spent_flops": self.spent_flops,
        }

    def to_row(
        self,
        *,
        includes_general: tuple[str, ...] = ("seed", "dataset", "label_noise"),
        include_teachers: bool = True,
        include_ensemble: bool = True,
        extra_params: list[str] | None = None,
    ) -> dict:
        """Flatten the run into one dict (one row) ready for columnar aggregation."""
        general = self.params.get("general", {})
        row: dict = {"run_dir": str(self.run_dir)}
        for key in includes_general:
            row[key] = general.get(key)
        row["student_test_acc"] = self.student_test_acc
        row["student_arch_idx"] = self.student.arch_idx
        row["student_seed"] = self.student.seed
        if include_teachers:
            row["teacher_test_accs"] = self.teacher_test_accs
            row["teacher_arch_idxs"] = [t.arch_idx for t in self.teachers]
            row["teacher_seeds"] = self.teacher_seeds
        if include_ensemble:
            row["ensemble_test_acc"] = self.ensemble_test_acc
        row["spent_flops"] = self.spent_flops
        row["params"] = self.params
        if extra_params:
            for path in extra_params:
                row[f"param_{path.replace('.', '_')}"] = _get_dotted(self.params, path)
        return row

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        device: torch.device | None = None,
    ) -> RunResult:
        """Analyze a single run directory and build its :class:`RunResult`.

        Reads ``params.yaml``, loads the student and every teacher (from
        ``kd.teachers_mapping``), scores them on the run's test split and keeps
        the test accuracies together with each model's own params.
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

        dataset_name = general.get("dataset", "cifar10")
        label_noise = float(general.get("label_noise", 0.0))
        num_classes = int(student_arch.get("num_classes", get_num_classes(dataset_name)))

        _, _, test_loader, _, _, _ = build_dataloaders(
            dataset_name,
            _resolve_path(kd["checkpoint_dir"]),
            _resolve_path(kd["log_dir"]),
            _resolve_path(general["data_root"]),
            general["seed"],
            general["batch_size"],
            general["num_workers"],
            general["validation_fraction"],
            device,
            label_noise=label_noise,
        )

        student_test_acc = score_on_validation(
            student_module, test_loader, device, num_classes
        )
        student = ModelResult(
            role="student",
            arch_idx=student_idx,
            seed=int(general["seed"]),
            checkpoint=student_ckpt,
            test_acc=student_test_acc,
            params=params,
            num_classes=num_classes,
            spent_flops=_read_spent_flops(run_dir),
        )

        teachers: list[ModelResult] = []
        ensemble: EnsembleResult | None = None
        teachers_mapping = kd.get("teachers_mapping")
        if teachers_mapping:
            teacher_modules = []
            teacher_indices = []
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
                teacher_module = load_model(teacher_ckpt, teacher_arch, teacher_params)
                teacher_num_classes = int(
                    teacher_arch.get("num_classes", num_classes)
                )
                teacher_test_acc = score_on_validation(
                    teacher_module, test_loader, device, teacher_num_classes
                )
                teacher_modules.append(teacher_module)
                teacher_indices.append(teacher_idx)
                teachers.append(
                    ModelResult(
                        role="teacher",
                        arch_idx=teacher_idx,
                        seed=(
                            int(teacher_params["general"]["seed"])
                            if teacher_params
                            else None
                        ),
                        checkpoint=teacher_ckpt,
                        test_acc=teacher_test_acc,
                        params=teacher_params or params,
                        num_classes=teacher_num_classes,
                        spent_flops=_read_spent_flops(teacher_ckpt.parent),
                    )
                )

            if len(teacher_modules) > 1:
                ensemble_module = TeacherEnsemble(teacher_modules)
                ensemble_test_acc = score_ensemble_on_validation(
                    ensemble_module, test_loader, device, num_classes
                )
                ensemble = EnsembleResult(
                    teacher_indices=tuple(teacher_indices),
                    test_acc=ensemble_test_acc,
                )

        return cls(
            run_dir=run_dir,
            params_path=params_path,
            params=params,
            student=student,
            teachers=teachers,
            ensemble=ensemble,
        )


def _model_to_dict(model: ModelResult) -> dict:
    return {
        "role": model.role,
        "arch_idx": model.arch_idx,
        "seed": model.seed,
        "checkpoint": str(model.checkpoint),
        "test_acc": model.test_acc,
        "params": model.params,
        "num_classes": model.num_classes,
        "spent_flops": model.spent_flops,
    }


def _read_spent_flops(run_dir: str | Path) -> int | None:
    """Cumulative FLOPs of a run, taken as the last ``cumulative_flops`` value."""
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if "cumulative_flops" not in df.columns:
        return None
    values = pd.to_numeric(df["cumulative_flops"], errors="coerce").dropna()
    return int(values.iloc[-1]) if len(values) else None


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
        return load_checkpoint(
            str(ckpt_path), arch, **_load_kwargs_from_params(params)
        )


def _top1_accuracy(
    logits_fn,
    val_loader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Compute top-1 accuracy from a callable mapping images -> logits."""
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=num_classes
    ).to(device)
    with torch.inference_mode():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = logits_fn(images)
            metric.update(logits, targets)
    return float(metric.compute())


def score_on_validation(
    module: torch.nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Compute top-1 accuracy of a module on the given loader (eval mode)."""
    module.to(device)
    module.eval()
    return _top1_accuracy(
        lambda images: extract_logits(module(images)),
        val_loader,
        device,
        num_classes,
    )


def score_ensemble_on_validation(
    ensemble: TeacherEnsemble,
    val_loader,
    device: torch.device,
    num_classes: int,
) -> float:
    """Compute top-1 accuracy of a TeacherEnsemble on the given loader."""
    ensemble.to(device)
    ensemble.eval()
    return _top1_accuracy(ensemble.logits, val_loader, device, num_classes)


def analyze(run_dir: str | Path, device: torch.device | None = None) -> RunResult:
    """Analyze a single run directory and return its :class:`RunResult`."""
    return RunResult.from_run_dir(run_dir, device=device)


def analyze_dir(
    base_dir: str | Path,
    device: torch.device | None = None,
) -> list[RunResult]:
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
            results.append(RunResult.from_run_dir(run_dir, device=device))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to analyze %s: %s", run_dir, exc)
    return results


def to_model_rows(results: list[RunResult]) -> list[dict]:
    """Flatten run results into a row per model (student + teachers)."""
    rows = []
    for res in results:
        rows.append(
            {
                "run_dir": str(res.run_dir),
                "role": "student",
                "arch_idx": res.student.arch_idx,
                "seed": res.student.seed,
                "checkpoint": str(res.student.checkpoint),
                "test_acc": res.student.test_acc,
                "params": res.student.params,
                "ensemble_acc": res.ensemble_test_acc,
                "spent_flops": res.student.spent_flops,
            }
        )
        for teacher in res.teachers:
            rows.append(
                {
                    "run_dir": str(res.run_dir),
                    "role": "teacher",
                    "arch_idx": teacher.arch_idx,
                    "seed": teacher.seed,
                    "checkpoint": str(teacher.checkpoint),
                    "test_acc": teacher.test_acc,
                    "params": teacher.params,
                    "spent_flops": teacher.spent_flops,
                }
            )
    return rows


def to_rows(
    results: list[RunResult],
    *,
    includes_general: tuple[str, ...] = ("seed", "dataset", "label_noise"),
    include_teachers: bool = True,
    include_ensemble: bool = True,
    extra_params: list[str] | None = None,
) -> list[dict]:
    """Flatten run results into one rich row per run.

    Each row groups the student, its teachers and the teacher ensemble together,
    together with the requested ``general`` keys (e.g. ``seed``, ``dataset``,
    ``label_noise``) and any ``extra_params`` dot-paths from the params.
    """
    return [
        res.to_row(
            includes_general=includes_general,
            include_teachers=include_teachers,
            include_ensemble=include_ensemble,
            extra_params=extra_params,
        )
        for res in results
    ]


def _get_dotted(params: dict, path: str):
    """Return the value at a dot-separated path inside a nested params dict."""
    current = params
    for part in path.split("."):
        current = current[part]
    return current


def to_dataframe(results: list[RunResult] | list[dict]) -> pd.DataFrame:
    """Build a DataFrame from run results (one row per checkpoint/run).

    Each row keeps the run's student and its teachers together: the student test
    accuracy, the teachers' accuracies/seeds and the ensemble accuracy as list
    columns, alongside the run's ``general`` keys. Nested ``params`` dicts are
    normalized into dot-separated columns (``params.general.seed``,
    ``params.kd.student``, ...).
    """
    if not results:
        return pd.DataFrame()
    if isinstance(results[0], RunResult):
        results = to_rows(results)
    return pd.json_normalize([dict(r) for r in results], sep=".")


def confidence_interval(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute a Student's t confidence interval for each group of a grouped series.

    ``grouped`` must already be grouped (a ``SeriesGroupBy``), e.g.::

        confidence_interval(
            to_dataframe(runs).groupby('arch_idx')['test_acc']
        )

    Returns a DataFrame indexed by group with ``n``, ``mean``, ``std``, ``sem``,
    ``ci_low`` and ``ci_high`` (``mean +/- t_{1-alpha/2, n-1} * sem``).
    """

    def _stats(values: pd.Series) -> dict:
        vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        n = int(vals.size)
        if n == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "sem": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        if n == 1:
            return {
                "n": n,
                "mean": float(vals[0]),
                "std": 0.0,
                "sem": 0.0,
                "ci_low": float(vals[0]),
                "ci_high": float(vals[0]),
                "min": float(vals[0]),
                "max": float(vals[0]),
            }
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        sem = std / np.sqrt(n)
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci_low": mean - crit * sem,
            "ci_high": mean + crit * sem,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    result = grouped.apply(_stats).unstack()
    result["n"] = result["n"].astype("Int64")
    return result


def main(base_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = analyze_dir(base_dir, device=device)

    if not results:
        print(f"No run directories with params.yaml found under {base_dir}")
        return

    print(f"{'run_dir':<45} {'role':<8} {'arch_idx':>8} {'test_acc':>8}")
    print("-" * 75)
    for row in to_model_rows(results):
        print(
            f"{row['run_dir']:<45} {row['role']:<8} "
            f"{row['arch_idx']:>8} {row['test_acc']:>8.4f}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m kd_vs_hpo.common.analyze_new <run_dir_or_checkpoint_dir>"
        )
    main(sys.argv[1])