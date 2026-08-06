import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


STUDY_FIELDS = (
    "study_name",
    "sampler",
    "pruner",
    "arch_row",
    "arch_index",
    "arch_str",
)
TRIAL_METADATA_FIELDS = STUDY_FIELDS[1:]


def save_run_config(output_dir: Path, config: dict[str, Any]) -> None:
    path = output_dir / "run_config.json"
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
    temporary_path.replace(path)


def save_study_progress(
    directory: Path,
    study: dict[str, Any],
    trials: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "study.csv", [study])
    if trials:
        normalized_trials = [
            {
                key: value
                for key, value in trial.items()
                if key not in TRIAL_METADATA_FIELDS
            }
            for trial in trials
        ]
        _write_csv(directory / "trials.csv", normalized_trials)
    if epochs:
        _write_csv(directory / "epoch_metrics.csv", epochs)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def save_result_tables(
    output_dir: Path,
    epochs: pd.DataFrame,
    trials: pd.DataFrame,
    studies: pd.DataFrame,
    architecture_summary: pd.DataFrame,
) -> None:
    tables = {
        "epoch_metrics": epochs,
        "trials": trials,
        "studies": studies,
        "architecture_summary": architecture_summary,
    }
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        temporary_path = path.with_suffix(".tmp")
        table.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
