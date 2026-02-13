"""Unified BLiMP/GLUE learning-curve loading and plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Optional, Tuple, Literal
import os

from src.visualization.utils import adjust_lightness
from src.visualization.config import DATASET_COLOR_SCHEME


# Task-specific metrics for GLUE/SuperGLUE
GLUE_TASK_METRICS = {
    'wsc': 'accuracy',
    'rte': 'accuracy',
    'mrpc': 'f1',
    'boolq': 'accuracy',
    'multirc': 'accuracy',
    'mnli': 'accuracy',
    'qqp': 'f1',
}


def load_learning_curve_data(
    datasets_config: Dict[str, Dict],
    benchmark: Literal["blimp", "glue"] = "blimp",
    data_dir: str = "./results/data/",
    split: str = "test",
) -> Dict:
    """Load curve-ready data for BLiMP or GLUE from consolidated CSVs."""
    if benchmark == "blimp":
        return _load_blimp_data(datasets_config, data_dir)
    elif benchmark == "glue":
        return _load_glue_data(datasets_config, data_dir, split)
    else:
        raise ValueError(f"benchmark must be 'blimp' or 'glue', got '{benchmark}'")


def _load_blimp_data(datasets_config: Dict, data_dir: str) -> Dict:
    """Load BLiMP data from consolidated CSV file."""
    data_dict = {}

    # Load consolidated BLiMP file
    csv_path = os.path.join(data_dir, "blimp_all_overall.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BLiMP results file not found: {csv_path}")

    full_df = pd.read_csv(csv_path)

    for dataset, info in datasets_config.items():
        data_dict[dataset] = {}
        vocab_size = info['vocab_size']

        for model_size in info['model_sizes']:
            try:
                # Filter for this dataset, model size, and vocab size
                overall_df = full_df[
                    (full_df['dataset'] == dataset) &
                    (full_df['model_size'] == model_size) &
                    (full_df['vocab_size'] == vocab_size)
                ].copy()

                if overall_df.empty:
                    print(f"✗ No data for {dataset} {model_size} with vocab={vocab_size}")
                    continue

                # Get numeric milestones (exclude 'final', limit to <=2000)
                all_milestones = overall_df['milestone'].unique()
                numeric_milestones = []
                for m in all_milestones:
                    try:
                        m_val = float(m)
                        if m_val <= 2000:
                            numeric_milestones.append(m_val)
                    except (ValueError, TypeError):
                        pass  # Skip 'final' or other non-numeric
                numeric_milestones = sorted(numeric_milestones)

                # Calculate mean/std curves
                curves_mean = []
                curves_std = []
                for milestone in numeric_milestones:
                    mask = overall_df['milestone'].astype(str) == str(int(milestone))
                    vals = overall_df.loc[mask, 'acc'].values
                    if len(vals) > 0:
                        curves_mean.append(np.mean(vals))
                        curves_std.append(np.std(vals))
                    else:
                        curves_mean.append(np.nan)
                        curves_std.append(np.nan)

                # Add 'anchor' column for compatibility with plotting
                if 'anchor' not in overall_df.columns:
                    overall_df['anchor'] = overall_df['milestone']

                data_dict[dataset][model_size] = {
                    'overall_df': overall_df,
                    'curves_mean': np.array(curves_mean),
                    'curves_std': np.array(curves_std),
                    'milestones': numeric_milestones,
                    'vocab_size': vocab_size
                }

                print(f"✓ Loaded {dataset} {model_size} (vocab={vocab_size}, {len(numeric_milestones)} milestones)")

            except Exception as e:
                print(f"✗ Error loading {dataset} {model_size}: {e}")
                continue

    return data_dict


def _load_glue_data(datasets_config: Dict, data_dir: str, split: str) -> Dict:
    """Load GLUE/SuperGLUE data from CSV files."""
    data_dict = {}

    csv_path = os.path.join(data_dir, "glue_overall_by_task.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GLUE results file not found: {csv_path}")

    full_df = pd.read_csv(csv_path)
    full_df = full_df[full_df['split'] == split].copy()

    for dataset, info in datasets_config.items():
        data_dict[dataset] = {}
        vocab_size = info['vocab_size']

        for model_size in info['model_sizes']:
            try:
                df = full_df[
                    (full_df['dataset'] == dataset) &
                    (full_df['model_size'] == model_size) &
                    (full_df['vocab_size'] == vocab_size)
                ].copy()

                if df.empty:
                    print(f"✗ No data for {dataset} {model_size} with vocab={vocab_size}")
                    continue

                # Get numeric milestones (handle "250M" format, exclude > 2000)
                all_milestones = df['milestone'].unique()
                numeric_milestones = []
                for m in all_milestones:
                    try:
                        if m == 'final':
                            continue
                        # Handle "250M" format
                        m_str = str(m)
                        if m_str.endswith('M'):
                            m_val = int(m_str.replace('M', ''))
                        else:
                            m_val = int(float(m_str))
                        if m_val <= 2000:
                            numeric_milestones.append(m_val)
                    except (ValueError, TypeError, AttributeError):
                        pass
                numeric_milestones = sorted(numeric_milestones)

                # Calculate average across tasks for each milestone
                curves_mean = []
                curves_std = []

                for milestone in numeric_milestones:
                    milestone_str = f"{milestone}M"
                    milestone_df = df[df['milestone'] == milestone_str]

                    if milestone_df.empty:
                        curves_mean.append(np.nan)
                        curves_std.append(np.nan)
                        continue

                    # Calculate average across tasks for each seed
                    seed_averages = []
                    for seed in milestone_df['seed'].unique():
                        seed_df = milestone_df[milestone_df['seed'] == seed]
                        task_scores = []
                        for task, metric in GLUE_TASK_METRICS.items():
                            task_data = seed_df[seed_df['task'] == task]
                            if not task_data.empty and metric in task_data.columns:
                                task_scores.append(task_data[metric].values[0])
                        if task_scores:
                            seed_averages.append(np.mean(task_scores))

                    if seed_averages:
                        curves_mean.append(np.mean(seed_averages))
                        curves_std.append(np.std(seed_averages))
                    else:
                        curves_mean.append(np.nan)
                        curves_std.append(np.nan)

                data_dict[dataset][model_size] = {
                    'overall_df': df,
                    'curves_mean': np.array(curves_mean),
                    'curves_std': np.array(curves_std),
                    'milestones': numeric_milestones,
                    'vocab_size': vocab_size
                }

                print(f"✓ Loaded {dataset} {model_size} (vocab={vocab_size}, {len(numeric_milestones)} milestones)")

            except Exception as e:
                print(f"✗ Error loading {dataset} {model_size}: {e}")
                continue

    return data_dict


def plot_learning_curves(
    data_dict: Dict,
    benchmark: Literal["blimp", "glue"] = "blimp",
    model_size_order: List[str] = ['20m', '60m', '180m'],
    show_shading: bool = True,
    spread_mode: str = "minmax",
    save_path: Optional[str] = None,
    color_scheme: Optional[Dict] = None,
    align_right_axis: bool = True,
    right_axis_padding: float = 0.03,
    show_title: bool = False,
    split: str = "test",
    # Layout / sizing
    figsize: Tuple[float, float] = (16.5, 8.2),
    main_axes_rect: Tuple[float, float, float, float] = (0.06, 0.1, 0.68, 0.8),
    final_axes_rect: Tuple[float, float, float, float] = (0.77, 0.1, 0.19, 0.8),
    # Y-axis control
    ylims: Optional[Tuple[float, float]] = None,
    # Font sizes
    label_fontsize: int = 17,
    tick_fontsize_main: int = 17,
    tick_fontsize_final: int = 16,
    legend_fontsize: int = 13,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot two-panel learning curves (main + final-summary) for BLiMP/GLUE."""

    # replace dataset names "hybrid_3.7B" with "composite"
    data_dict = {k.replace("hybrid_3.7B", "composite"): v for k, v in data_dict.items()}
    
    if color_scheme is None:
        color_scheme = DATASET_COLOR_SCHEME

    # Benchmark-specific settings
    if benchmark == "blimp":
        milestone_col = "anchor"
        score_col = "acc"
        x_label = "Anchor (in million tokens)"
        y_label = "BLiMP Macro-Average"
        final_label = "Final Macro-Average"
        title = "BLiMP Learning Curves — Multi-Dataset Comparison"
    elif benchmark == "glue":
        milestone_col = "milestone"
        score_col = None  # GLUE uses task-specific metrics
        x_label = "Milestone (in million tokens)"
        y_label = "GLUE Macro-Average"
        final_label = "Final Macro-Average"
        title = f"GLUE/SuperGLUE Learning Curves ({split.capitalize()} Set)"
    else:
        raise ValueError(f"benchmark must be 'blimp' or 'glue', got '{benchmark}'")

    # Figure / axes
    fig = plt.figure(figsize=figsize)
    ax_main = fig.add_axes(list(main_axes_rect))
    ax_final = fig.add_axes(list(final_axes_rect))
    ax_final.grid(False)

    # Dataset ordering
    dataset_order = ['tinystories', 'babylm3', 'hybrid_3.7B']
    ordered_datasets = [d for d in dataset_order if d in data_dict]
    ordered_datasets.extend([d for d in sorted(data_dict.keys()) if d not in dataset_order])

    # Left panel: learning curves
    for dataset_name in ordered_datasets:
        models = data_dict[dataset_name]

        for model_size in model_size_order:
            if model_size not in models:
                continue

            data = models[model_size]
            df = data['overall_df']
            mean_vals = np.asarray(data['curves_mean'])
            std_vals = np.asarray(data['curves_std'])
            milestones = np.asarray(data['milestones'][:len(mean_vals)])
            vocab_size = data['vocab_size']

            model_color = color_scheme.get(dataset_name, {}).get(model_size, '#666666')
            label = f"{dataset_name} {model_size} (V={vocab_size//1000}k)"

            ax_main.plot(
                milestones,
                mean_vals,
                marker='o',
                linewidth=3.0,
                color=model_color,
                label=label,
                zorder=3,
            )

            if show_shading:
                lower, upper = _compute_shading(
                    df, milestones, mean_vals, std_vals,
                    spread_mode, benchmark, milestone_col
                )

                mask = np.isfinite(lower) & np.isfinite(upper)
                if np.any(mask):
                    ax_main.fill_between(
                        milestones[mask],
                        lower[mask],
                        upper[mask],
                        color=adjust_lightness(model_color, 1.1),
                        alpha=0.35,
                        edgecolor=model_color,
                        linewidth=1.2,
                        zorder=2,
                    )

    # Right panel: final summary
    summary_positions = []
    y_mins, y_maxs = [], []
    model_labels = []
    idx = 0

    for dataset_name in ordered_datasets:
        models = data_dict[dataset_name]

        for model_size in model_size_order:
            if model_size not in models:
                continue

            data = models[model_size]
            df = data['overall_df']
            model_color = color_scheme.get(dataset_name, {}).get(model_size, '#666666')

            final_mean, final_min, final_max = _compute_final_scores(
                df, benchmark, milestone_col
            )

            if final_mean is None:
                continue

            summary_positions.append(idx)
            y_mins.append(final_min)
            y_maxs.append(final_max)
            model_labels.append(f"{dataset_name}-{model_size}")

            ax_final.vlines(idx, final_min, final_max,
                            color=model_color, linewidth=2.2)
            ax_final.scatter(idx, final_mean,
                             color=model_color, s=95,
                             marker='x', linewidth=2.6, zorder=6)
            idx += 1

    ax_final.set_xlim(-0.5, len(summary_positions) - 0.5)
    ax_final.set_xticks(summary_positions)
    ax_final.set_xticklabels(
        model_labels,
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )
    ax_final.yaxis.tick_right()
    ax_final.yaxis.set_label_position("right")
    ax_final.set_ylabel(final_label, fontsize=label_fontsize, fontweight="bold")

    # Y-axis limits
    if ylims is not None:
        ax_main.set_ylim(*ylims)
        ax_final.set_ylim(*ylims)

    elif align_right_axis and y_mins:
        ymin = min(y_mins)
        ymax = max(y_maxs)
        pad = right_axis_padding * (ymax - ymin if ymax > ymin else 1.0)

        padded_min = ymin - pad
        padded_max = min(ymax + pad, 1.0)

        left_ymin, left_ymax = ax_main.get_ylim()
        new_ymin = min(left_ymin, padded_min)
        new_ymax = min(max(left_ymax, padded_max), 1.0)

        ax_main.set_ylim(new_ymin, new_ymax)
        ax_final.set_ylim(new_ymin, new_ymax)

    # Formatting / styling
    formatter = FuncFormatter(lambda y, _: f"{y:.3f}")
    ax_main.yaxis.set_major_formatter(formatter)
    ax_final.yaxis.set_major_formatter(formatter)

    if show_title:
        ax_main.set_title(title, fontsize=18, pad=15)

    ax_main.set_xlabel(x_label, fontsize=label_fontsize, fontweight="bold")
    ax_main.set_ylabel(y_label, fontsize=label_fontsize, fontweight="bold")

    ax_main.grid(True, linestyle="--", alpha=0.3)
    ax_main.legend(fontsize=legend_fontsize, loc="lower right")

    for spine in ax_main.spines.values():
        spine.set_linewidth(1.4)

    # Tick sizing (single authority)
    ax_main.tick_params(axis='both', labelsize=tick_fontsize_main)
    ax_final.tick_params(axis='both', labelsize=tick_fontsize_final)
    ax_final.tick_params(axis='x', pad=7)

    # Save
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if save_path.lower().endswith(".pdf"):
            fig.savefig(save_path[:-4] + ".png", dpi=300, bbox_inches="tight")

    plt.show()
    return fig, (ax_main, ax_final)


def _compute_shading(
    df: pd.DataFrame,
    milestones: np.ndarray,
    mean_vals: np.ndarray,
    std_vals: np.ndarray,
    spread_mode: str,
    benchmark: str,
    milestone_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper bounds for shading."""

    if spread_mode == "std":
        return mean_vals - std_vals, mean_vals + std_vals

    elif spread_mode == "minmax":
        if benchmark == "blimp":
            return _compute_minmax_blimp(df, milestones, milestone_col)
        else:  # glue
            return _compute_minmax_glue(df, milestones)

    else:
        raise ValueError("spread_mode must be 'minmax' or 'std'")


def _compute_minmax_blimp(
    df: pd.DataFrame,
    milestones: np.ndarray,
    milestone_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min/max bounds for BLiMP."""
    lower, upper = [], []

    for milestone in milestones:
        # Match milestone as string (consistent with loader)
        mask = df[milestone_col].astype(str) == str(int(milestone))
        vals = df.loc[mask, "acc"].to_numpy(dtype=float)
        lower.append(np.nanmin(vals) if vals.size else np.nan)
        upper.append(np.nanmax(vals) if vals.size else np.nan)

    return np.asarray(lower), np.asarray(upper)


def _compute_minmax_glue(
    df: pd.DataFrame,
    milestones: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min/max bounds for GLUE (average across tasks per seed)."""
    lower, upper = [], []

    for milestone in milestones:
        milestone_df = df[df['milestone'] == f"{milestone}M"]
        if milestone_df.empty:
            lower.append(np.nan)
            upper.append(np.nan)
            continue

        seed_averages = []
        for seed in milestone_df['seed'].unique():
            seed_df = milestone_df[milestone_df['seed'] == seed]
            task_scores = []
            for task, metric in GLUE_TASK_METRICS.items():
                task_data = seed_df[seed_df['task'] == task]
                if not task_data.empty and metric in task_data.columns:
                    task_scores.append(task_data[metric].values[0])
            if task_scores:
                seed_averages.append(float(np.mean(task_scores)))

        lower.append(np.min(seed_averages) if seed_averages else np.nan)
        upper.append(np.max(seed_averages) if seed_averages else np.nan)

    return np.asarray(lower), np.asarray(upper)


def _compute_final_scores(
    df: pd.DataFrame,
    benchmark: str,
    milestone_col: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute final mean, min, max scores."""

    if benchmark == "blimp":
        final_rows = df[df[milestone_col] == "final"]
        if final_rows.empty:
            return None, None, None

        vals = final_rows["acc"].to_numpy(dtype=float)
        return float(np.nanmean(vals)), float(np.nanmin(vals)), float(np.nanmax(vals))

    else:  # glue
        final_df = df[df['milestone'] == 'final']
        if final_df.empty:
            return None, None, None

        seed_averages = []
        for seed in final_df['seed'].unique():
            seed_df = final_df[final_df['seed'] == seed]
            task_scores = []
            for task, metric in GLUE_TASK_METRICS.items():
                task_data = seed_df[seed_df['task'] == task]
                if not task_data.empty and metric in task_data.columns:
                    task_scores.append(task_data[metric].values[0])
            if task_scores:
                seed_averages.append(float(np.mean(task_scores)))

        if not seed_averages:
            return None, None, None

        return (
            float(np.mean(seed_averages)),
            float(np.min(seed_averages)),
            float(np.max(seed_averages))
        )
