import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build accuracy-vs-FLOPs plots for every Optuna study."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("hpo_output/logs"),
        help="Merged events.jsonl file or HPO logs directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo_output/plots/flops_by_study"),
    )
    parser.add_argument(
        "--metric",
        choices=("val_acc1", "best_val_acc1"),
        default="val_acc1",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID when the input contains multiple runs.",
    )
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help="Embed Plotly JavaScript into every HTML file.",
    )
    return parser.parse_args()


def resolve_event_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Logs path does not exist: {source}")

    latest = source / "events.jsonl"
    if latest.exists():
        return [latest]

    run_dirs = sorted(
        (path for path in (source / "runs").glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No experiment runs found under: {source}")

    latest_run = run_dirs[0]
    merged = latest_run / "events.jsonl"
    if merged.exists():
        return [merged]
    return sorted((latest_run / "studies").glob("*.jsonl"))


def load_epoch_events(paths: list[Path], run_id: str | None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {error}"
                    ) from error
                if record.get("event") != "epoch_completed":
                    continue
                if run_id is not None and record.get("run_id") != run_id:
                    continue
                records.append(record)

    if not records:
        raise ValueError(
            "No epoch_completed events found. Check --events/--run-id and make "
            "sure the selected run contains study logs."
        )

    frame = pd.DataFrame.from_records(records)
    required = {
        "study_name",
        "trial_id",
        "epoch",
        "val_acc1",
        "best_val_acc1",
        "cumulative_trial_flops",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Epoch logs are missing fields: {sorted(missing)}")
    if "run_id" not in frame:
        frame["run_id"] = "unknown"
    return frame.drop_duplicates(
        ["run_id", "study_name", "trial_id", "epoch"],
        keep="last",
    )


def build_study_plot(
    study: pd.DataFrame,
    metric: str,
    output_path: Path,
    self_contained: bool,
) -> None:
    study = study.sort_values(["trial_id", "epoch"])
    study["cumulative_study_tflops"] = study["cumulative_study_flops"] / 1e12
    first = study.iloc[0]
    figure = go.Figure()

    for trial_id, trial in study.groupby("trial_id", sort=True):
        custom_columns = [
            "epoch",
            "cumulative_study_tflops",
            "lr",
            "weight_decay",
            "train_loss",
            "pruner_decision",
            "growth_stop_reason",
        ]
        custom_data = trial.reindex(columns=custom_columns).to_numpy()
        figure.add_trace(
            go.Scatter(
                x=trial["cumulative_trial_flops"] / 1e12,
                y=trial[metric],
                mode="lines+markers",
                name=f"trial {int(trial_id)}",
                customdata=custom_data,
                hovertemplate=(
                    "trial=%{fullData.name}<br>"
                    "epoch=%{customdata[0]}<br>"
                    "trial TFLOPs=%{x:.3f}<br>"
                    "study TFLOPs=%{customdata[1]:.3e}<br>"
                    f"{metric}=%{{y:.3f}}<br>"
                    "lr=%{customdata[2]:.3e}<br>"
                    "weight_decay=%{customdata[3]:.3e}<br>"
                    "train_loss=%{customdata[4]:.4f}<br>"
                    "pruned=%{customdata[5]}<br>"
                    "growth_stop=%{customdata[6]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template="plotly_white",
        title=(
            f"Accuracy vs FLOPs — {first['study_name']}<br>"
            f"architecture={first.get('arch_index', 'unknown')}, "
            f"sampler={first.get('sampler', 'unknown')}, "
            f"pruner={first.get('pruner', 'unknown')}"
        ),
        xaxis_title="Cumulative trial FLOPs, TFLOPs",
        yaxis_title=(
            "Validation accuracy, %"
            if metric == "val_acc1"
            else "Best validation accuracy, %"
        ),
        legend_title="Trials",
        hovermode="closest",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True if self_contained else "cdn",
        full_html=True,
    )


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def write_index(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['study_name']))}</td>"
            f"<td>{html.escape(str(row['arch_index']))}</td>"
            f"<td>{html.escape(str(row['sampler']))}</td>"
            f"<td>{html.escape(str(row['pruner']))}</td>"
            f"<td>{row['trials']}</td>"
            f"<td><a href=\"{html.escape(row['filename'])}\">open</a></td>"
            "</tr>"
        )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>HPO studies: accuracy vs FLOPs</title>"
        "<style>body{font-family:sans-serif;margin:2rem}"
        "table{border-collapse:collapse}th,td{padding:.5rem;border:1px solid #ccc}"
        "</style></head><body><h1>Accuracy vs FLOPs by study</h1>"
        "<table><thead><tr><th>Study</th><th>Architecture</th>"
        "<th>Sampler</th><th>Pruner</th><th>Trials</th><th>Plot</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>"
    )
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    paths = resolve_event_files(args.events)
    epochs = load_epoch_events(paths, args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for study_name, study in epochs.groupby("study_name", sort=True):
        filename = f"{safe_filename(str(study_name))}.html"
        build_study_plot(
            study,
            args.metric,
            args.output_dir / filename,
            args.self_contained,
        )
        first = study.iloc[0]
        index_rows.append(
            {
                "study_name": study_name,
                "arch_index": first.get("arch_index", "unknown"),
                "sampler": first.get("sampler", "unknown"),
                "pruner": first.get("pruner", "unknown"),
                "trials": study["trial_id"].nunique(),
                "filename": filename,
            }
        )

    index_path = write_index(index_rows, args.output_dir)
    print(f"Built {len(index_rows)} study plots")
    print(f"Index: {index_path.resolve()}")


if __name__ == "__main__":
    main()
