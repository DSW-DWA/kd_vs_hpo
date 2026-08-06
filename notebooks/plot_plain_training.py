from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_COLOR = "#E2E8F0"
PALETTE = ("#3978D4", "#169B62", "#D98E04", "#8B5CF6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build interactive graphs from plain-training CSV tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("plain_training_output"),
        help="Directory containing experiment subdirectories with tables/*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plain_training_output/figures"),
        help="Directory for generated HTML graphs.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Centered rolling window used to smooth learning curves.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Embed Plotly into every HTML file instead of using shared JS.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def load_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_paths = sorted(input_dir.glob("*/tables/epoch_metrics.csv"))
    if not epoch_paths:
        raise FileNotFoundError(
            f"No */tables/epoch_metrics.csv files found under {input_dir}"
        )

    epoch_frames: list[pd.DataFrame] = []
    run_frames: list[pd.DataFrame] = []
    for epoch_path in epoch_paths:
        experiment_dir = epoch_path.parents[1]
        runs_path = experiment_dir / "tables" / "runs.csv"
        if not runs_path.is_file():
            raise FileNotFoundError(f"Missing input table: {runs_path}")

        epochs = pd.read_csv(epoch_path)
        runs = pd.read_csv(runs_path)
        require_columns(
            epochs,
            {
                "run_name",
                "arch_index",
                "trial_id",
                "trial_seed",
                "target_epochs",
                "epoch",
                "train_loss",
                "train_acc1",
                "val_loss",
                "val_acc1",
                "best_val_acc1",
                "learning_rate",
            },
            epoch_path,
        )
        require_columns(
            runs,
            {
                "run_name",
                "arch_index",
                "trial_id",
                "trial_seed",
                "target_epochs",
                "completed_epochs",
                "best_epoch",
                "best_val_acc1",
                "final_val_acc1",
                "tested_checkpoint",
                "test_acc1",
                "elapsed_seconds",
            },
            runs_path,
        )
        epochs.insert(0, "experiment_id", experiment_dir.name)
        runs.insert(0, "experiment_id", experiment_dir.name)
        epoch_frames.append(epochs)
        run_frames.append(runs)

    all_epochs = pd.concat(epoch_frames, ignore_index=True)
    all_runs = pd.concat(run_frames, ignore_index=True)
    key = ["experiment_id", "run_name"]
    if all_runs.duplicated(key).any():
        raise ValueError("runs.csv files contain duplicate experiment/run keys")
    if all_epochs.duplicated([*key, "epoch"]).any():
        raise ValueError("epoch_metrics.csv files contain duplicate epoch rows")
    if set(map(tuple, all_epochs[key].drop_duplicates().to_numpy())) != set(
        map(tuple, all_runs[key].to_numpy())
    ):
        raise ValueError("runs.csv and epoch_metrics.csv contain different runs")

    for run in all_runs.itertuples(index=False):
        curve = all_epochs.loc[
            (all_epochs["experiment_id"] == run.experiment_id)
            & (all_epochs["run_name"] == run.run_name)
        ].sort_values("epoch")
        expected = list(range(1, int(run.completed_epochs) + 1))
        if curve["epoch"].astype(int).tolist() != expected:
            raise ValueError(f"{run.run_name} has missing or unexpected epoch rows")
        if int(run.target_epochs) != int(run.completed_epochs):
            raise ValueError(f"{run.run_name} did not reach its target epoch")
        if not math.isclose(
            float(curve["val_acc1"].max()),
            float(run.best_val_acc1),
            abs_tol=1e-9,
        ):
            raise ValueError(f"{run.run_name} has inconsistent best validation metric")
    return all_epochs, all_runs


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def smooth(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def run_curve(
    epochs: pd.DataFrame,
    experiment_id: str,
    run_name: str,
) -> pd.DataFrame:
    return epochs.loc[
        (epochs["experiment_id"] == experiment_id) & (epochs["run_name"] == run_name)
    ].sort_values("epoch")


def trial_label(run: pd.Series, repeated_horizon: bool) -> str:
    label = f"{int(run['target_epochs'])} эпох · seed {int(run['trial_seed'])}"
    if repeated_horizon:
        label += f" · {run['experiment_id']}"
    return label


def add_line(
    figure: go.Figure,
    *,
    row: int,
    col: int,
    x: pd.Series,
    y: pd.Series,
    name: str,
    legendgroup: str,
    color: str,
    dash: str = "solid",
    width: float = 2.5,
    opacity: float = 1.0,
    showlegend: bool = False,
    hover_value: str,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            line={"color": color, "dash": dash, "width": width},
            opacity=opacity,
            hovertemplate=(
                f"<b>{html.escape(name)}</b><br>"
                "Эпоха: %{x}<br>" + hover_value + ": %{y:.4f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def architecture_figure(
    architecture_runs: pd.DataFrame,
    epochs: pd.DataFrame,
    *,
    rolling_window: int,
    colors: dict[int, str],
) -> go.Figure:
    arch_index = int(architecture_runs.iloc[0]["arch_index"])
    figure = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
        subplot_titles=(
            "Validation accuracy",
            "Train и validation accuracy",
            "Train и validation loss",
            "Learning rate",
        ),
    )
    horizon_counts = architecture_runs["target_epochs"].value_counts()

    for _, run in architecture_runs.sort_values("target_epochs").iterrows():
        target_epochs = int(run["target_epochs"])
        color = colors[target_epochs]
        group = f"{run['experiment_id']}::{run['run_name']}"
        label = trial_label(run, int(horizon_counts[target_epochs]) > 1)
        curve = run_curve(
            epochs,
            str(run["experiment_id"]),
            str(run["run_name"]),
        )
        val_smooth = smooth(curve["val_acc1"], rolling_window)
        train_acc_smooth = smooth(curve["train_acc1"], rolling_window)
        train_loss_smooth = smooth(curve["train_loss"], rolling_window)
        val_loss_smooth = smooth(curve["val_loss"], rolling_window)

        add_line(
            figure,
            row=1,
            col=1,
            x=curve["epoch"],
            y=curve["val_acc1"],
            name=f"{label} · сырые значения",
            legendgroup=group,
            color=color,
            width=1,
            opacity=0.20,
            hover_value="Validation accuracy, %",
        )
        add_line(
            figure,
            row=1,
            col=1,
            x=curve["epoch"],
            y=val_smooth,
            name=label,
            legendgroup=group,
            color=color,
            width=3,
            showlegend=True,
            hover_value=f"Validation accuracy, rolling {rolling_window}, %",
        )
        best_row = curve.loc[curve["val_acc1"].idxmax()]
        figure.add_trace(
            go.Scatter(
                x=[int(best_row["epoch"])],
                y=[float(best_row["val_acc1"])],
                mode="markers",
                name=f"Лучший checkpoint · {label}",
                legendgroup=group,
                showlegend=False,
                marker={
                    "symbol": "star",
                    "size": 15,
                    "color": color,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    f"<b>Лучший checkpoint · {html.escape(label)}</b><br>"
                    "Эпоха: %{x}<br>Validation accuracy: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        add_line(
            figure,
            row=1,
            col=2,
            x=curve["epoch"],
            y=train_acc_smooth,
            name=f"Train · {label}",
            legendgroup=group,
            color=color,
            hover_value="Train accuracy, %",
        )
        add_line(
            figure,
            row=1,
            col=2,
            x=curve["epoch"],
            y=val_smooth,
            name=f"Validation · {label}",
            legendgroup=group,
            color=color,
            dash="dash",
            hover_value="Validation accuracy, %",
        )
        add_line(
            figure,
            row=2,
            col=1,
            x=curve["epoch"],
            y=train_loss_smooth,
            name=f"Train loss · {label}",
            legendgroup=group,
            color=color,
            hover_value="Train loss",
        )
        add_line(
            figure,
            row=2,
            col=1,
            x=curve["epoch"],
            y=val_loss_smooth,
            name=f"Validation loss · {label}",
            legendgroup=group,
            color=color,
            dash="dash",
            hover_value="Validation loss",
        )
        add_line(
            figure,
            row=2,
            col=2,
            x=curve["epoch"],
            y=curve["learning_rate"],
            name=f"Learning rate · {label}",
            legendgroup=group,
            color=color,
            hover_value="Learning rate",
        )

    summaries = []
    for run in architecture_runs.sort_values("target_epochs").itertuples(index=False):
        summaries.append(
            f"{int(run.target_epochs)} эпох: best val {float(run.best_val_acc1):.2f}% "
            f"@{int(run.best_epoch)}, final test {float(run.test_acc1):.2f}%"
        )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"<b>Архитектура {arch_index} · независимые plain-training trials</b>"
                f"<br><sup>{' · '.join(summaries)}</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        height=930,
        margin={"l": 75, "r": 245, "t": 120, "b": 125},
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="#F7F9FC",
        font={"family": "Arial, sans-serif", "color": "#172033", "size": 13},
        legend={
            "orientation": "v",
            "x": 1.01,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "groupclick": "togglegroup",
            "title": {"text": "Нажмите на trial, чтобы скрыть/показать"},
        },
        annotations=[
            *figure.layout.annotations,
            {
                "x": 0,
                "y": -0.14,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "showarrow": False,
                "align": "left",
                "text": (
                    f"Толстая линия: rolling mean ({rolling_window} эпох); "
                    "пунктир: validation; звезда: лучший raw validation checkpoint.<br>"
                    "Важно: trials используют разные seed и разные CosineAnnealingLR "
                    "T_max. Test accuracy рассчитана на final checkpoint."
                ),
                "font": {"size": 12, "color": "#64748B"},
            },
        ],
    )
    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        figure.update_xaxes(
            title="Эпоха",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            row=row,
            col=col,
        )
    figure.update_yaxes(
        title="Accuracy, %",
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="Accuracy, %",
        ticksuffix="%",
        showgrid=True,
        gridcolor=GRID_COLOR,
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title="Loss",
        showgrid=True,
        gridcolor=GRID_COLOR,
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title="Learning rate",
        tickformat=".2e",
        showgrid=True,
        gridcolor=GRID_COLOR,
        row=2,
        col=2,
    )
    return figure


def summary_figure(runs: pd.DataFrame, colors: dict[int, str]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.09,
        subplot_titles=(
            "Final test accuracy",
            "Best validation accuracy",
            "Время обучения и test accuracy",
        ),
    )
    for target_epochs in sorted(runs["target_epochs"].astype(int).unique()):
        selected = runs.loc[runs["target_epochs"] == target_epochs].sort_values(
            "arch_index"
        )
        color = colors[target_epochs]
        label = f"{target_epochs} эпох"
        x = selected["arch_index"].astype(int).astype(str)
        custom = selected[["trial_seed", "best_epoch", "elapsed_seconds"]].to_numpy()
        figure.add_trace(
            go.Bar(
                x=x,
                y=selected["test_acc1"],
                name=label,
                legendgroup=str(target_epochs),
                marker_color=color,
                text=selected["test_acc1"].map(lambda value: f"{value:.2f}%"),
                textposition="outside",
                customdata=custom,
                hovertemplate=(
                    f"<b>{label}</b><br>Архитектура: %{{x}}<br>"
                    "Final test accuracy: %{y:.2f}%<br>"
                    "Seed: %{customdata[0]:.0f}<br>"
                    "Лучший epoch: %{customdata[1]:.0f}<br>"
                    "Время: %{customdata[2]:.1f} сек.<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=x,
                y=selected["best_val_acc1"],
                name=label,
                legendgroup=str(target_epochs),
                showlegend=False,
                marker_color=color,
                text=selected["best_val_acc1"].map(lambda value: f"{value:.2f}%"),
                textposition="outside",
                customdata=custom,
                hovertemplate=(
                    f"<b>{label}</b><br>Архитектура: %{{x}}<br>"
                    "Best validation accuracy: %{y:.2f}%<br>"
                    "Seed: %{customdata[0]:.0f}<br>"
                    "Лучший epoch: %{customdata[1]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=selected["elapsed_seconds"] / 60.0,
                y=selected["test_acc1"],
                mode="markers+text",
                name=label,
                legendgroup=str(target_epochs),
                showlegend=False,
                text=selected["arch_index"].map(lambda value: f"arch {int(value)}"),
                textposition="top center",
                marker={"color": color, "size": 13},
                customdata=selected[["trial_seed", "target_epochs"]].to_numpy(),
                hovertemplate=(
                    "<b>Архитектура %{text}</b><br>"
                    "Время: %{x:.2f} мин.<br>Test accuracy: %{y:.2f}%<br>"
                    "Seed: %{customdata[0]:.0f}<br>"
                    "Target: %{customdata[1]:.0f} эпох<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>Plain training · сравнение 100 и 200 эпох</b><br>"
                "<sup>Test измерен на final checkpoint; best validation — на лучшей эпохе</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        barmode="group",
        height=650,
        margin={"l": 70, "r": 45, "t": 120, "b": 120},
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033", "size": 13},
        legend={"orientation": "h", "x": 0, "y": 1.08},
        annotations=[
            *figure.layout.annotations,
            {
                "x": 0,
                "y": -0.18,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "showarrow": False,
                "align": "left",
                "text": (
                    "100 и 200 эпох — независимые trials: seed 42 против 43, "
                    "CosineAnnealingLR T_max 100 против 200.<br>"
                    "Поэтому разница не является чистым эффектом дополнительных эпох."
                ),
                "font": {"size": 12, "color": "#64748B"},
            },
        ],
    )
    for col in (1, 2):
        figure.update_xaxes(title="Архитектура", row=1, col=col)
        figure.update_yaxes(
            title="Accuracy, %",
            range=[0, 100],
            ticksuffix="%",
            gridcolor=GRID_COLOR,
            row=1,
            col=col,
        )
    figure.update_xaxes(title="Время, минуты", gridcolor=GRID_COLOR, row=1, col=3)
    figure.update_yaxes(
        title="Final test accuracy, %",
        ticksuffix="%",
        gridcolor=GRID_COLOR,
        row=1,
        col=3,
    )
    return figure


def write_figure(
    figure: go.Figure,
    path: Path,
    *,
    include_plotlyjs: bool | str,
) -> None:
    figure.write_html(
        path,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": path.stem,
                "scale": 2,
            },
        },
    )


def index_document(runs: pd.DataFrame, architecture_files: dict[int, str]) -> str:
    cards = [
        (
            "summary.html",
            "Общее сравнение",
            "Final test, best validation и компромисс между временем и точностью.",
        )
    ]
    for arch_index, filename in sorted(architecture_files.items()):
        arch_runs = runs.loc[runs["arch_index"] == arch_index].sort_values(
            "target_epochs"
        )
        first = arch_runs.iloc[0]
        last = arch_runs.iloc[-1]
        delta_test = float(last["test_acc1"] - first["test_acc1"])
        time_ratio = float(last["elapsed_seconds"] / first["elapsed_seconds"])
        cards.append(
            (
                filename,
                f"Архитектура {arch_index}",
                (
                    f"100 → 200 эпох: test {delta_test:+.2f} п.п., "
                    f"время ×{time_ratio:.2f}. Learning curves, loss и LR."
                ),
            )
        )
    card_html = "\n".join(
        "<a class='card' href='"
        + html.escape(filename)
        + "'><strong>"
        + html.escape(title)
        + "</strong><span>"
        + html.escape(description)
        + "</span></a>"
        for filename, title, description in cards
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plain training — графики</title>
  <style>
    body {{ margin: 0; background: #f3f6fa; color: #172033;
           font: 16px Arial, sans-serif; }}
    main {{ max-width: 1050px; margin: 48px auto; padding: 0 24px; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #64748b; line-height: 1.55; }}
    .notice {{ padding: 16px 18px; background: #fff8e7; border: 1px solid #f4d58d;
               border-radius: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
             gap: 16px; margin-top: 24px; }}
    .card {{ display: flex; flex-direction: column; gap: 10px; padding: 22px;
             color: inherit; text-decoration: none; background: white;
             border: 1px solid #dce4ef; border-radius: 14px; }}
    .card:hover {{ border-color: #3978d4; box-shadow: 0 8px 24px #1e293b14; }}
    .card strong {{ color: #3978d4; font-size: 18px; }}
    .card span {{ color: #64748b; line-height: 1.45; }}
  </style>
</head>
<body>
<main>
  <h1>Plain training — интерактивные графики</h1>
  <p>Наведите курсор для точных значений, используйте легенду для скрытия линий,
     колесо мыши для zoom и кнопку камеры для сохранения PNG.</p>
  <p class="notice"><strong>Важно:</strong> 100- и 200-epoch trials используют
     разные seed и разные cosine schedules. Это отдельные обучения, а не один запуск,
     продолженный ещё на 100 эпох. Test accuracy относится к final checkpoint.</p>
  <div class="grid">{card_html}</div>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.rolling_window < 1:
        raise ValueError("--rolling-window must be positive")
    input_dir = project_path(args.input_dir)
    output_dir = project_path(args.output_dir)
    epochs, runs = load_results(input_dir)

    horizons = sorted(runs["target_epochs"].astype(int).unique())
    colors = {
        target_epochs: PALETTE[index % len(PALETTE)]
        for index, target_epochs in enumerate(horizons)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    include_plotlyjs: bool | str = True if args.standalone else "directory"
    if not args.standalone:
        (output_dir / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

    write_figure(
        summary_figure(runs, colors),
        output_dir / "summary.html",
        include_plotlyjs=include_plotlyjs,
    )
    architecture_files: dict[int, str] = {}
    for arch_index, architecture_runs in runs.groupby("arch_index", sort=True):
        filename = safe_filename(f"architecture_{int(arch_index)}") + ".html"
        write_figure(
            architecture_figure(
                architecture_runs,
                epochs,
                rolling_window=args.rolling_window,
                colors=colors,
            ),
            output_dir / filename,
            include_plotlyjs=include_plotlyjs,
        )
        architecture_files[int(arch_index)] = filename

    (output_dir / "index.html").write_text(
        index_document(runs, architecture_files),
        encoding="utf-8",
    )
    print(f"Generated {len(architecture_files) + 1} graphs in {output_dir}")
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
