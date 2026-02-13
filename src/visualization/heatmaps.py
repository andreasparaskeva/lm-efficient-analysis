"""Unified BLiMP/GLUE heatmap and summary-plot utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Literal, Tuple
import os

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


def load_heatmap_data(
    model_configs: List[Dict[str, str]],
    benchmark: Literal["blimp", "glue"] = "blimp",
    data_dir: str = "results/data/",
    split: str = "test",
) -> pd.DataFrame:
    """Load task-level rows used by `plot_heatmap`."""
    if benchmark == "blimp":
        return _load_blimp_heatmap_data(model_configs, data_dir)
    elif benchmark == "glue":
        return _load_glue_heatmap_data(model_configs, data_dir, split)
    else:
        raise ValueError(f"benchmark must be 'blimp' or 'glue', got '{benchmark}'")


def _load_blimp_heatmap_data(
    model_configs: List[Dict[str, str]],
    data_dir: str,
) -> pd.DataFrame:
    """Load BLiMP task-level data for heatmap."""
    # Load consolidated BLiMP tasks file
    tasks_file = os.path.join(data_dir, "blimp_all_tasks.csv")
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"BLiMP tasks file not found: {tasks_file}")

    full_df = pd.read_csv(tasks_file)
    all_rows = []
    loaded_configs = 0

    for config in model_configs:
        dataset = config['dataset']
        model_size = config['model_size']
        milestone = config['milestone']
        vocab_size = config['vocab_size']

        model_name = f"{dataset}-{model_size}-{milestone}"

        # Handle milestone format: strip "M" suffix if present (e.g., "2000M" -> "2000")
        milestone_str = str(milestone)
        if milestone_str.endswith('M') and milestone_str[:-1].isdigit():
            milestone_str = milestone_str[:-1]

        # Filter for this configuration
        filtered = full_df[
            (full_df['dataset'] == dataset) &
            (full_df['model_size'] == model_size) &
            (full_df['vocab_size'] == vocab_size) &
            (full_df['milestone'].astype(str) == milestone_str)
        ].copy()

        if filtered.empty:
            print(f"⚠️  No data for {model_name} (vocab={vocab_size}, milestone={milestone})")
            continue

        # Standardize columns
        filtered['model_name'] = model_name
        filtered['score'] = filtered['acc']
        filtered = filtered[['model_name', 'task', 'score', 'seed']]

        n_tasks = filtered['task'].nunique()
        n_seeds = filtered['seed'].nunique()
        all_rows.append(filtered)
        loaded_configs += 1
        print(f"✓ Loaded {model_name}: {len(filtered)} results ({n_tasks} tasks × {n_seeds} seeds)")

    if not all_rows:
        raise ValueError("No data could be loaded for any model configuration")

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\n✓ Total: {loaded_configs}/{len(model_configs)} configs, {len(combined)} results")
    return combined


def _load_glue_heatmap_data(
    model_configs: List[Dict[str, str]],
    data_dir: str,
    split: str,
) -> pd.DataFrame:
    """Load GLUE/SuperGLUE task-level data for heatmap."""
    csv_path = os.path.join(data_dir, "glue_overall_by_task.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GLUE results file not found: {csv_path}")

    full_df = pd.read_csv(csv_path)
    full_df = full_df[full_df['split'] == split].copy()

    all_rows = []
    loaded_configs = 0

    for config in model_configs:
        dataset = config['dataset']
        model_size = config['model_size']
        milestone = config['milestone']
        vocab_size = config['vocab_size']

        model_name = f"{dataset}-{model_size}-{milestone}"

        # Filter for this configuration
        filtered = full_df[
            (full_df['dataset'] == dataset) &
            (full_df['model_size'] == model_size) &
            (full_df['milestone'] == milestone) &
            (full_df['vocab_size'] == vocab_size)
        ].copy()

        if filtered.empty:
            print(f"⚠️  No data for {model_name} (vocab={vocab_size})")
            continue

        config_rows = 0
        # Extract the correct metric for each task
        for task, metric_name in GLUE_TASK_METRICS.items():
            task_data = filtered[filtered['task'] == task]

            if task_data.empty:
                continue

            if metric_name not in task_data.columns:
                print(f"⚠️  Missing {metric_name} for {model_name} task {task}")
                continue

            for _, row in task_data.iterrows():
                all_rows.append({
                    'model_name': model_name,
                    'task': task,
                    'score': row[metric_name],
                    'seed': row['seed'],
                })
                config_rows += 1

        if config_rows > 0:
            loaded_configs += 1
            print(f"✓ Loaded {model_name}: {config_rows} results ({config_rows // 3} tasks × 3 seeds)")

    if not all_rows:
        raise ValueError("No data could be loaded for any model configuration")

    print(f"\n✓ Total: {loaded_configs}/{len(model_configs)} configs, {len(all_rows)} results")
    return pd.DataFrame(all_rows)


def plot_heatmap(
    model_configs: List[Dict[str, str]],
    benchmark: Literal["blimp", "glue"] = "blimp",
    data_dir: str = "results/data/",
    split: str = "test",
    save_path: Optional[str] = None,
    color_scheme: Optional[Dict[str, Dict[str, str]]] = None,
    heatmap_cmap: str = 'viridis',
    figsize: Optional[Tuple[float, float]] = None,
    show_title: bool = False,
    show_std: Optional[bool] = None,
    # Font sizes
    label_fontsize: int = 20,
    tick_fontsize_x: int = 16,
    tick_fontsize_y: Optional[int] = None,  # Auto-adaptive if None
    cell_fontsize: Optional[int] = None,  # Auto-adaptive if None
    boxplot_label_fontsize: int = 18,
) -> plt.Figure:
    """Create a two-panel task-performance figure (heatmap + boxplot)."""
    if color_scheme is None:
        color_scheme = DATASET_COLOR_SCHEME

    # Benchmark-specific settings
    if benchmark == "blimp":
        y_axis_label = "BLiMP Tasks"
        boxplot_y_label = "Accuracy"
        title = "BLiMP Task Performance Heatmap"
        default_figsize = (24, 18)
    elif benchmark == "glue":
        y_axis_label = "GLUE/SuperGLUE Tasks"
        boxplot_y_label = "Score"
        title = f"GLUE/SuperGLUE Task Performance Heatmap ({split.capitalize()} Set)"
        default_figsize = (20, 12)
    else:
        raise ValueError(f"benchmark must be 'blimp' or 'glue', got '{benchmark}'")

    # Load data
    df = load_heatmap_data(model_configs, benchmark=benchmark, data_dir=data_dir, split=split)

    # Get unique models and tasks in order 
    # replace dataset name "hybrid_3.7B" with "composite"
    model_names = [f"{c['dataset']}-{c['model_size']}-{c['milestone']}" for c in model_configs]

    # Task ordering
    if benchmark == "glue":
        # Order GLUE tasks by metric type (accuracy first, then f1)
        all_tasks = sorted(df['task'].unique(), key=lambda x: (GLUE_TASK_METRICS.get(x, 'z'), x))
    else:
        all_tasks = sorted(df['task'].unique())

    n_tasks = len(all_tasks)
    n_models = len(model_names)

    # Figure size
    if figsize is None:
        figsize = default_figsize

    # Auto-determine whether to show std based on density
    if show_std is None:
        show_std = n_tasks <= 30

    # Auto-adaptive font sizes based on number of tasks
    if cell_fontsize is None:
        if n_tasks > 50:
            cell_fontsize = 11
        elif n_tasks > 30:
            cell_fontsize = 12
        elif n_tasks > 15:
            cell_fontsize = 13
        else:
            cell_fontsize = 14

    if tick_fontsize_y is None:
        if n_tasks > 50:
            tick_fontsize_y = 12
        elif n_tasks > 30:
            tick_fontsize_y = 14
        elif n_tasks > 15:
            tick_fontsize_y = 16
        else:
            tick_fontsize_y = 18

    # Calculate mean and std score per model-task combination
    score_matrix = []
    std_matrix = []
    data_for_boxplot = []

    for model_name in model_names:
        model_data = df[df['model_name'] == model_name]

        task_means = []
        task_stds = []

        for task in all_tasks:
            task_data = model_data[model_data['task'] == task]['score']
            if len(task_data) > 0:
                task_means.append(task_data.mean())
                task_stds.append(task_data.std())
            else:
                task_means.append(np.nan)
                task_stds.append(np.nan)

        score_matrix.append(task_means)
        std_matrix.append(task_stds)
        data_for_boxplot.append([x for x in task_means if not np.isnan(x)])

    # Convert to numpy arrays (tasks x models)
    score_matrix = np.array(score_matrix).T
    std_matrix = np.array(std_matrix).T

    # Task-wise normalization (normalize each row)
    row_mins = np.nanmin(score_matrix, axis=1, keepdims=True)
    row_maxs = np.nanmax(score_matrix, axis=1, keepdims=True)
    score_matrix_normalized = (score_matrix - row_mins) / (row_maxs - row_mins + 1e-8)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [13, 4]}
    )

    # Left panel: heatmap
    ax1.imshow(score_matrix_normalized, cmap=heatmap_cmap, aspect='auto')

    # Add values inside each cell
    for i in range(score_matrix.shape[0]):  # Tasks (rows)
        for j in range(score_matrix.shape[1]):  # Models (columns)
            value = score_matrix[i, j]
            std_value = std_matrix[i, j]
            normalized_value = score_matrix_normalized[i, j]

            if np.isnan(value):
                continue

            # Text color for visibility
            text_color = "white" if normalized_value > 0.5 else "black"

            # Format cell text
            if show_std and not np.isnan(std_value):
                cell_text = f"{value:.3f}\n±{std_value:.3f}"
            else:
                cell_text = f"{value:.3f}"

            ax1.text(
                j, i, cell_text,
                ha="center", va="center",
                color=text_color,
                fontsize=cell_fontsize,
                fontweight="bold"
            )

    # Task labels
    if benchmark == "glue":
        task_labels = [f"{task.upper()} ({GLUE_TASK_METRICS[task]})" for task in all_tasks]
    else:
        task_labels = all_tasks
        
        
    # replace datset name "hybrid_3.7B" with "composite" in model_names
    model_names = [label.replace("hybrid_3.7B", "composite") for label in model_names]

    ax1.set_xticks(np.arange(len(model_names)))
    ax1.set_yticks(np.arange(len(all_tasks)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_yticklabels(task_labels)
    ax1.tick_params(axis='x', labelsize=tick_fontsize_x)
    ax1.tick_params(axis='y', labelsize=tick_fontsize_y)
    ax1.set_xlabel('Models', fontsize=label_fontsize, fontweight='bold')
    ax1.set_ylabel(y_axis_label, fontsize=label_fontsize, fontweight='bold')

    # Right panel: boxplot
    colors = []
    
    for model_name in model_names:
        parts = model_name.split("-")
        dataset_name = parts[0]
        dataset_name = dataset_name.replace("hybrid_3.7B", "composite")
        model_size = parts[1]
        model_color = color_scheme.get(dataset_name, {}).get(model_size, '#666666')
        colors.append(model_color)

    box = ax2.boxplot(
        data_for_boxplot,
        labels=model_names,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        showfliers=True
    )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_ylabel(boxplot_y_label, fontsize=boxplot_label_fontsize, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Title (optional)
    if show_title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)

    # Save and show
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.abspath(save_path)}")
        print(f"Saved: {os.path.abspath(png_path)}")

    plt.show()
    return fig
