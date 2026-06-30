from pathlib import Path

import pandas as pd
import plotly.express as px


def save_plots(
    epochs: pd.DataFrame,
    studies: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    if epochs.empty:
        return ()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    history = px.line(
        epochs,
        x="epoch",
        y="best_val_acc1",
        color="strategy",
        facet_col="arch_index",
        line_group="trial_id",
        title="Validation accuracy by Optuna strategy",
    )
    history_path = plots_dir / "training_history.html"
    history.write_html(history_path, include_plotlyjs=True)

    comparison = px.bar(
        studies,
        x="strategy",
        y="best_val_acc1",
        color="pruner",
        facet_col="arch_index",
        title="Best validation accuracy by sampler and pruner",
    )
    comparison_path = plots_dir / "strategy_comparison.html"
    comparison.write_html(comparison_path, include_plotlyjs=True)
    return history_path, comparison_path
