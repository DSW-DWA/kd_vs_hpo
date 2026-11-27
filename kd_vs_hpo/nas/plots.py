import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable


def _collect_curves(
    run_fn: Callable,
    flops: np.ndarray,
    compute_time: np.ndarray,
    n_seeds: int,
    n_points: int = 500,
    dataset_n_samples: int = 54000,
    epochs: int = 12,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "evals": [],
        "flops": [],
        "time": [],
    }

    for seed in range(n_seeds):
        df = run_fn(seed)
        arch_ids = df["archid"].astype(int).values
        best_acc = df["accuracy"].cummax().values
        n = len(df)

        # flops_mul = 3 * dataset_n_samples * epochs
        flops_mul = 1 # already applied in flops creation
        raw["evals"].append((np.arange(1, n + 1, dtype=np.float64), best_acc))
        raw["flops"].append((np.cumsum(flops_mul * flops[arch_ids]), best_acc))
        raw["time"].append((np.cumsum(compute_time[arch_ids]) / 3600, best_acc))

    results = {}
    for key, curves in raw.items():
        x_max = min(x.max() for x, _ in curves)
        x_min = max(x.min() for x, _ in curves)
        x_grid = np.linspace(x_min, x_max, n_points)
        matrix = np.array([np.interp(x_grid, x, y) for x, y in curves])
        results[key] = (x_grid, matrix.mean(axis=0), matrix.std(axis=0))

    return results


def _add_band(fig, x, mean, std, name, color, row, col):
    fill_color = color.replace(",1)", ",0.2)")
    kw = dict(row=row, col=col)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean + std,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        **kw,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean - std,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fill_color,
            showlegend=False,
            hoverinfo="skip",
        ),
        **kw,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        ),
        **kw,
    )


def plot_search_comparison(
    algorithms: list[tuple[str, Callable, str]],
    acc: np.ndarray,
    flops: np.ndarray,
    compute_time: np.ndarray,
    n_seeds: int = 10,
    n_points: int = 500,
) -> go.Figure:
    """
    Plot 1×3 grid: Best Accuracy vs. {# Evaluations, Cumulative FLOPs, Training Time}.

    Parameters
    ----------
    algorithms : list of (label, run_fn, rgba_color)
        Each `run_fn(seed)` returns a DataFrame with columns ["archid", "accuracy"].
    acc : precomputed accuracies indexed by arch id.
    flops : precomputed FLOPs indexed by arch id.
    compute_time : precomputed training times (seconds) indexed by arch id.
    n_seeds : number of random seeds for mean±std.
    n_points : interpolation resolution per curve.
    """
    panels = [
        ("evals", "|Architectures Evaluated|"),
        ("flops", "Total cumulative estimate, 1e15 FLOPs"),
        ("time", "Training Time (hours)"),
    ]

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[f"Best Accuracy vs. {title}" for _, title in panels],
        horizontal_spacing=0.07,
    )

    y_min, y_max = float("inf"), float("-inf")

    for label, run_fn, color in algorithms:
        all_curves = _collect_curves(run_fn, flops, compute_time, n_seeds, n_points)

        for col_idx, (key, _) in enumerate(panels, start=1):
            x, mean, std = all_curves[key]
            _add_band(fig, x, mean, std, label, color, row=1, col=col_idx)
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())

    # Optimum line & x-axis labels
    for col_idx, (_, x_label) in enumerate(panels, start=1):
        fig.add_hline(
            y=acc.max(), line_dash="dash", line_color="green", row=1, col=col_idx
        )
        fig.update_xaxes(title_text=x_label, row=1, col=col_idx)

    fig.update_yaxes(title_text="Best Accuracy (%)", row=1, col=1)

    y_max = max(y_max, acc.max())
    padding = (y_max - y_min) * 0.05
    y_range = [y_min - padding, y_max + padding]
    for col_idx in range(1, len(panels) + 1):
        fig.update_yaxes(range=y_range, row=1, col=col_idx)

    # Deduplicate legend
    seen = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        elif trace.name:
            seen.add(trace.name)

    fig.update_layout(
        title="NAS Algorithm Comparison",
        template="plotly_white",
        width=1600,
        height=480,
        legend=dict(x=0.42, y=0.05),
    )
    return fig

def calculate_evolution_params(total_iterations: int) -> dict:
    """
    Calculate parameters for EvolutionParetoSearch based on TOTAL_ITERATIONS.

    The total number of model evaluations is approximately:
        init_num_models + num_iters * (num_random_mix + pareto_size * mutations_per_parent + num_crossovers)

    Args:
        total_iterations: Approximate total number of model evaluations budget.

    Returns:
        Dictionary with parameters for EvolutionParetoSearch.
    """
    x = int(np.sqrt(12 * total_iterations) / 7)
    x_sqrt = int(np.sqrt(x))

    return {
        "num_iters": x,
        "init_num_models": int(2 * x),
        "num_random_mix": 3 * x_sqrt,
        "mutations_per_parent": 3 * x_sqrt,
        "num_crossovers": 3 * x_sqrt,
    }

