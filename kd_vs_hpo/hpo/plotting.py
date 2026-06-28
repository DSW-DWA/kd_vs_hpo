from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _write_figure(figure: go.Figure, path: Path) -> Path:
    figure.update_layout(
        template="plotly_white",
        legend_title_text="Architecture",
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def save_hpo_plots(
    stages: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Build and save interactive plots for a completed HPO experiment."""
    if stages.empty:
        return ()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    completed = stages.loc[stages["status"] == "completed"].copy()
    if completed.empty:
        return ()

    completed["arch_index"] = completed["arch_index"].astype(str)
    completed["budget_used_pct"] = (
        100 * completed["cumulative_flops"] / completed["budget_flops"]
    )

    history = px.line(
        completed,
        x="budget_used_pct",
        y="best_val_acc1_so_far",
        color="arch_index",
        markers=True,
        hover_data=[
            "trial_id",
            "rung",
            "target_epochs",
            "lr",
            "weight_decay",
            "val_acc1",
        ],
        labels={
            "budget_used_pct": "FLOPs budget used, %",
            "best_val_acc1_so_far": "Best validation accuracy, %",
            "arch_index": "Architecture",
        },
        title="HPO progress under the FLOPs budget",
    )

    parameters = px.scatter(
        completed,
        x="lr",
        y="weight_decay",
        color="val_acc1",
        symbol="arch_index",
        size="target_epochs",
        hover_data=["trial_id", "rung", "target_epochs", "budget_used_pct"],
        log_x=True,
        log_y=True,
        color_continuous_scale="Viridis",
        labels={
            "lr": "Learning rate",
            "weight_decay": "Weight decay",
            "val_acc1": "Validation accuracy, %",
            "arch_index": "Architecture",
            "target_epochs": "Epochs",
        },
        title="Hyperparameter performance",
    )

    plot_paths = [
        _write_figure(history, plots_dir / "optimization_history.html"),
        _write_figure(parameters, plots_dir / "hyperparameter_performance.html"),
    ]

    if not summary.empty:
        comparison = summary.copy()
        comparison["arch_index"] = comparison["arch_index"].astype(str)
        comparison = comparison.melt(
            id_vars=["arch_index"],
            value_vars=["val_acc1", "test_acc1"],
            var_name="metric",
            value_name="accuracy",
        )
        comparison["metric"] = comparison["metric"].map(
            {
                "val_acc1": "Validation",
                "test_acc1": "Test",
            }
        )
        architecture_comparison = px.bar(
            comparison,
            x="arch_index",
            y="accuracy",
            color="metric",
            barmode="group",
            labels={
                "arch_index": "Architecture",
                "accuracy": "Accuracy, %",
                "metric": "Dataset",
            },
            title="Best HPO result by architecture",
        )
        plot_paths.append(
            _write_figure(
                architecture_comparison,
                plots_dir / "architecture_comparison.html",
            )
        )

    return tuple(plot_paths)
