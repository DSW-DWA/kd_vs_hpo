from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRUNED_COLOR = "#D98E04"
COMPLETE_COLOR = "#3978D4"
BEST_COLOR = "#169B62"
GRID_COLOR = "#E2E8F0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one interactive validation-learning graph for every HPO study."
        )
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("hpo_output/tables"),
        help="Directory with epoch_metrics.csv, trials.csv, and studies.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo_output/figures/trial_training_interactive"),
        help="Directory for the generated HTML files.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Embed Plotly into every HTML file. Files become fully standalone, "
            "but the output is much larger."
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def load_tables(
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "epochs": tables_dir / "epoch_metrics.csv",
        "trials": tables_dir / "trials.csv",
        "studies": tables_dir / "studies.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing input tables: {missing_files}")

    epochs = pd.read_csv(paths["epochs"])
    trials = pd.read_csv(paths["trials"])
    studies = pd.read_csv(paths["studies"])

    require_columns(
        epochs,
        {
            "study_name",
            "trial_id",
            "epoch",
            "val_acc1",
            "best_val_acc1",
            "train_loss",
            "learning_rate",
            "cumulative_trial_flops",
        },
        paths["epochs"],
    )
    require_columns(
        trials,
        {
            "study_name",
            "trial_id",
            "state",
            "completed_epochs",
            "lr",
            "weight_decay",
        },
        paths["trials"],
    )
    require_columns(
        studies,
        {
            "study_name",
            "sampler",
            "pruner",
            "arch_index",
            "complete_trials",
            "pruned_trials",
            "best_trial_id",
            "best_val_acc1",
        },
        paths["studies"],
    )

    if trials.duplicated(["study_name", "trial_id"]).any():
        raise ValueError("trials.csv contains duplicate (study_name, trial_id) keys")
    if studies["study_name"].duplicated().any():
        raise ValueError("studies.csv contains duplicate study_name keys")
    if epochs.duplicated(["study_name", "trial_id", "epoch"]).any():
        raise ValueError(
            "epoch_metrics.csv contains duplicate (study_name, trial_id, epoch) keys"
        )

    expected_studies = set(studies["study_name"])
    if set(trials["study_name"]) != expected_studies:
        raise ValueError("trials.csv and studies.csv contain different studies")
    if set(epochs["study_name"]) != expected_studies:
        raise ValueError("epoch_metrics.csv and studies.csv contain different studies")

    return epochs, trials, studies


def display_pruner(pruner: str) -> str:
    if pruner == "hyperband":
        return "Hyperband"
    if pruner == "successive_halving":
        return "Successive Halving"
    return pruner.replace("_", " ").title()


def safe_filename(study_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name).strip("._")


def y_range(curves: pd.DataFrame) -> list[float]:
    minimum = math.floor((float(curves["val_acc1"].min()) - 2.0) / 5.0) * 5.0
    maximum = math.ceil((float(curves["val_acc1"].max()) + 2.0) / 5.0) * 5.0
    return [max(0.0, minimum), min(100.0, maximum)]


def trial_style(state: str, is_best: bool) -> tuple[str, float, int, str]:
    if is_best:
        return BEST_COLOR, 1.0, 4, "solid"
    if state == "COMPLETE":
        return COMPLETE_COLOR, 0.65, 2, "solid"
    return PRUNED_COLOR, 0.45, 2, "dash"


def add_trial(
    figure: go.Figure,
    curve: pd.DataFrame,
    trial: pd.Series,
    *,
    best_trial_id: int,
) -> None:
    trial_id = int(trial["trial_id"])
    state = str(trial["state"])
    is_best = trial_id == best_trial_id
    color, opacity, width, dash = trial_style(state, is_best)
    completed_epochs = int(trial["completed_epochs"])
    state_label = (
        "лучший"
        if is_best
        else "завершён"
        if state == "COMPLETE"
        else "остановлен"
    )
    legend_group = f"trial-{trial_id}"

    ordered = curve.sort_values("epoch")
    customdata = ordered[
        [
            "best_val_acc1",
            "train_loss",
            "learning_rate",
            "cumulative_trial_flops",
        ]
    ].copy()
    customdata["cumulative_trial_flops"] /= 1e12

    figure.add_trace(
        go.Scatter(
            x=ordered["epoch"],
            y=ordered["val_acc1"],
            mode="lines",
            name=f"Trial #{trial_id} · {state_label} · {completed_epochs} эп.",
            legendgroup=legend_group,
            legendrank=(
                trial_id
                if is_best
                else 100 + trial_id
                if state == "COMPLETE"
                else 200 + trial_id
            ),
            line={"color": color, "width": width, "dash": dash},
            opacity=opacity,
            customdata=customdata.to_numpy(),
            hovertemplate=(
                f"<b>Trial #{trial_id}</b> · {state_label}<br>"
                "Эпоха: %{x}<br>"
                "Validation accuracy: %{y:.2f}%<br>"
                "Лучший результат к этой эпохе: %{customdata[0]:.2f}%<br>"
                "Train loss: %{customdata[1]:.4f}<br>"
                "Learning rate: %{customdata[2]:.3e}<br>"
                "Вычисления: %{customdata[3]:.2f} TFLOPs<br>"
                f"Начальный lr: {float(trial['lr']):.3e}<br>"
                f"Weight decay: {float(trial['weight_decay']):.3e}"
                "<extra></extra>"
            ),
        )
    )

    if state == "PRUNED":
        last = ordered.iloc[-1]
        figure.add_trace(
            go.Scatter(
                x=[int(last["epoch"])],
                y=[float(last["val_acc1"])],
                mode="markers",
                name=f"Остановка trial #{trial_id}",
                legendgroup=legend_group,
                showlegend=False,
                marker={
                    "symbol": "x",
                    "size": 9,
                    "color": PRUNED_COLOR,
                    "line": {"width": 2},
                },
                hovertemplate=(
                    f"<b>Trial #{trial_id} остановлен pruner’ом</b><br>"
                    "Последняя эпоха: %{x}<br>"
                    "Validation accuracy: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )


def build_study_figure(
    study: pd.Series,
    epochs: pd.DataFrame,
    trials: pd.DataFrame,
) -> go.Figure:
    study_name = str(study["study_name"])
    study_epochs = epochs.loc[epochs["study_name"] == study_name]
    study_trials = trials.loc[trials["study_name"] == study_name].copy()
    if study_epochs.empty or study_trials.empty:
        raise ValueError(f"{study_name} has no trial or epoch rows")

    best_trial_id = int(study["best_trial_id"])
    best_trial = study_trials.loc[study_trials["trial_id"] == best_trial_id]
    if len(best_trial) != 1:
        raise ValueError(f"{study_name} has invalid best_trial_id={best_trial_id}")

    figure = go.Figure()
    order = study_trials.assign(
        plot_order=study_trials.apply(
            lambda row: (
                2
                if int(row["trial_id"]) == best_trial_id
                else 1
                if row["state"] == "COMPLETE"
                else 0
            ),
            axis=1,
        )
    ).sort_values(["plot_order", "trial_id"])
    for trial in order.itertuples(index=False):
        trial_series = pd.Series(trial._asdict())
        curve = study_epochs.loc[study_epochs["trial_id"] == int(trial.trial_id)]
        if curve.empty:
            raise ValueError(f"{study_name} trial {trial.trial_id} has no epoch rows")
        add_trial(
            figure,
            curve,
            trial_series,
            best_trial_id=best_trial_id,
        )

    best_curve = study_epochs.loc[study_epochs["trial_id"] == best_trial_id]
    best_point = best_curve.loc[best_curve["val_acc1"].idxmax()]
    best_epoch = int(best_point["epoch"])
    best_accuracy = float(best_point["val_acc1"])
    figure.add_trace(
        go.Scatter(
            x=[best_epoch],
            y=[best_accuracy],
            mode="markers",
            name="Лучший checkpoint",
            legendgroup=f"trial-{best_trial_id}",
            showlegend=False,
            marker={
                "symbol": "star",
                "size": 16,
                "color": BEST_COLOR,
                "line": {"color": "white", "width": 1},
            },
            hovertemplate=(
                f"<b>Лучший checkpoint · trial #{best_trial_id}</b><br>"
                "Эпоха: %{x}<br>"
                "Validation accuracy: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    for epoch in (10, 30, 90):
        figure.add_vline(
            x=epoch,
            line={"color": "#C7CDD7", "width": 1, "dash": "dot"},
            layer="below",
        )

    sampler = str(study["sampler"]).upper()
    pruner = display_pruner(str(study["pruner"]))
    complete = int(study["complete_trials"])
    pruned = int(study["pruned_trials"])
    total = complete + pruned
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"<b>Архитектура {int(study['arch_index'])} · "
                f"{sampler} + {pruner}</b><br>"
                f"<sup>{total} trials: {complete} завершены, {pruned} остановлены · "
                f"лучший trial #{best_trial_id}: {best_accuracy:.2f}% "
                f"на эпохе {best_epoch}</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        autosize=True,
        height=850,
        margin={"l": 85, "r": 330, "t": 110, "b": 80},
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="#F7F9FC",
        font={"family": "Arial, sans-serif", "color": "#172033", "size": 14},
        legend={
            "title": {"text": "Trials — нажмите, чтобы скрыть/показать"},
            "x": 1.01,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#D9E1EC",
            "borderwidth": 1,
            "groupclick": "togglegroup",
        },
        annotations=[
            {
                "x": 0.01,
                "y": 0.99,
                "xref": "paper",
                "yref": "paper",
                "text": (
                    "Оранжевый пунктир: остановлен · синий: завершён · "
                    "зелёный: лучший trial · вертикальные точки: рубежи pruning"
                ),
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 13, "color": "#64748B"},
                "bgcolor": "rgba(255,255,255,0.78)",
                "borderpad": 5,
            }
        ],
    )
    figure.update_xaxes(
        title="Эпоха",
        range=[0, 202],
        dtick=20,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        fixedrange=False,
    )
    figure.update_yaxes(
        title="Validation accuracy, %",
        range=y_range(study_epochs),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        fixedrange=False,
        ticksuffix="%",
    )
    return figure


def write_index(output_dir: Path, generated: list[tuple[pd.Series, str]]) -> None:
    cards: list[str] = []
    for study, filename in generated:
        sampler = html.escape(str(study["sampler"]).upper())
        pruner = html.escape(display_pruner(str(study["pruner"])))
        cards.append(
            "<a class='card' href='"
            + html.escape(filename)
            + "'><strong>Архитектура "
            + str(int(study["arch_index"]))
            + "</strong><span>"
            + sampler
            + " + "
            + pruner
            + "</span><small>"
            + str(int(study["complete_trials"]))
            + " завершены · "
            + str(int(study["pruned_trials"]))
            + " остановлены · лучший результат "
            + f"{float(study['best_val_acc1']):.2f}%"
            + "</small></a>"
        )
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Графики обучения HPO trials</title>
  <style>
    body { margin: 0; background: #f3f6fa; color: #172033;
           font: 16px Arial, sans-serif; }
    main { max-width: 1120px; margin: 48px auto; padding: 0 24px; }
    h1 { margin-bottom: 8px; }
    p { color: #64748b; margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px; margin-top: 28px; }
    .card { display: flex; flex-direction: column; gap: 8px; padding: 22px;
            color: inherit; text-decoration: none; background: white;
            border: 1px solid #dce4ef; border-radius: 14px; }
    .card:hover { border-color: #3978d4; box-shadow: 0 8px 24px #1e293b14; }
    .card span { color: #3978d4; font-weight: 700; }
    .card small { color: #64748b; }
  </style>
</head>
<body>
<main>
  <h1>Графики обучения HPO trials</h1>
  <p>Отдельный интерактивный график validation accuracy для каждого Optuna study.</p>
  <div class="grid">""" + "\n".join(cards) + """</div>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = project_path(args.tables_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs, trials, studies = load_tables(tables_dir)

    generated: list[tuple[pd.Series, str]] = []
    ordered_studies = studies.sort_values(
        ["arch_index", "sampler", "pruner"],
        ignore_index=True,
    )
    include_plotlyjs: bool | str = True if args.standalone else "directory"
    if not args.standalone:
        (output_dir / "plotly.min.js").write_text(
            get_plotlyjs(),
            encoding="utf-8",
        )
    for _, study in ordered_studies.iterrows():
        study_name = str(study["study_name"])
        filename = safe_filename(study_name) + ".html"
        figure = build_study_figure(study, epochs, trials)
        config = {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": safe_filename(study_name),
                "scale": 2,
            },
        }
        figure.write_html(
            output_dir / filename,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
            config=config,
        )
        generated.append((study, filename))

    write_index(output_dir, generated)
    print(f"Generated {len(generated)} study graphs in {output_dir}")
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
