"""Grouped BLiMP task visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

from src.visualization.blimp_tasks import BLIMP_GROUPS, BLIMP_TASKS
from src.visualization.data_simple import load_blimp
from src.visualization.config import DATASET_COLOR_SCHEME


def load_grouped_blimp_data_multi(
    model_configs: List[Dict[str, any]],
    data_dir: str = "results/data/",
    return_std: bool = False
) -> pd.DataFrame:
    """Load grouped BLiMP task means (and optional std) for multiple configs."""
    all_data_mean = []
    all_data_std = []

    for config in model_configs:
        dataset = config['dataset']
        model_size = config['model_size']
        milestone = config['milestone']
        vocab_size = config['vocab_size']

        # Generate column label
        if 'label' in config:
            column_label = config['label']
        else:
            column_label = f"{dataset}-{model_size}-V{vocab_size//1000}k"

        try:
            # Load task-level data
            _, task_df = load_blimp(dataset, model_size, data_dir)

            # Handle milestone format: strip "M" suffix if present (e.g., "2000M" -> "2000")
            milestone_str = str(milestone)
            if milestone_str.endswith('M') and milestone_str[:-1].isdigit():
                milestone_str = milestone_str[:-1]

            # Filter for specific milestone and vocab size
            filtered = task_df[
                (task_df['milestone'].astype(str) == milestone_str) &
                (task_df['vocab_size'] == vocab_size)
            ]

            if filtered.empty:
                print(f"⚠️  No data for {column_label} at milestone {milestone}")
                continue

            # Calculate mean and std across seeds for each task
            task_means = filtered.groupby('task')['acc'].mean()
            task_means.name = column_label
            all_data_mean.append(task_means)

            if return_std:
                task_stds = filtered.groupby('task')['acc'].std()
                task_stds.name = column_label
                all_data_std.append(task_stds)

        except FileNotFoundError:
            print(f"✗ Data file not found for {dataset} {model_size}")
            continue
        except Exception as e:
            print(f"✗ Error loading {column_label}: {e}")
            continue

    if not all_data_mean:
        raise ValueError("No data could be loaded for any configuration")

    # Combine all columns
    df_mean = pd.concat(all_data_mean, axis=1)

    # Reorder rows to match BLIMP_TASKS order
    task_order = [t for t in BLIMP_TASKS if t in df_mean.index]
    df_mean = df_mean.loc[task_order]

    if return_std:
        df_std = pd.concat(all_data_std, axis=1)
        df_std = df_std.loc[task_order]
        return df_mean, df_std

    return df_mean


def load_grouped_blimp_data(
    dataset: str,
    model_size: str,
    milestone: str,
    vocab_sizes: List[int],
    data_dir: str = "results/data/"
) -> pd.DataFrame:
    """Legacy wrapper: load grouped BLiMP data indexed by vocab size."""
    # Convert to new format
    configs = [
        {
            'dataset': dataset,
            'model_size': model_size,
            'milestone': milestone,
            'vocab_size': vocab_size,
            'label': f"{vocab_size}"
        }
        for vocab_size in vocab_sizes
    ]

    df = load_grouped_blimp_data_multi(configs, data_dir=data_dir)

    # Convert column labels back to int for backward compatibility
    df.columns = [int(col) for col in df.columns]
    df = df[sorted(df.columns)]

    return df


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise min-max normalization with neutral 0.5 for constant rows."""
    df_scaled = df.copy()
    for row in df_scaled.index:
        vals = df_scaled.loc[row]
        val_min, val_max = vals.min(), vals.max()
        if val_max == val_min:
            df_scaled.loc[row] = 0.5  # Neutral if all values are same
        else:
            df_scaled.loc[row] = (vals - val_min) / (val_max - val_min)
    return df_scaled


def plot_grouped_blimp_heatmaps(
    model_configs: List[Dict[str, any]] = None,
    dataset: str = None,
    model_size: str = None,
    milestone: str = None,
    vocab_sizes: List[int] = None,
    data_dir: str = "results/data/",
    save_path: Optional[str] = None,
    groups_per_row: int = 4,
    include_misc: bool = False,
    figsize_per_subplot: Tuple[float, float] = (8, 6),
    cmap: str = 'viridis',
    show_title: bool = False,
) -> plt.Figure:
    """Plot per-group BLiMP heatmaps for one or more model configurations."""
    # Handle legacy interface
    if model_configs is None:
        if dataset is None or model_size is None or milestone is None or vocab_sizes is None:
            raise ValueError("Either provide model_configs or all of (dataset, model_size, milestone, vocab_sizes)")

        # Convert legacy parameters to new format
        model_configs = [
            {
                'dataset': dataset,
                'model_size': model_size,
                'milestone': milestone,
                'vocab_size': vocab_size,
                'label': f"{vocab_size}"
            }
            for vocab_size in vocab_sizes
        ]
        plot_title = f"BLiMP Task Groups - {dataset} {model_size} (Milestone: {milestone})"
    else:
        # Generate title from configs
        unique_datasets = list(set(c['dataset'] for c in model_configs))
        unique_models = list(set(c['model_size'] for c in model_configs))
        if len(unique_datasets) == 1 and len(unique_models) == 1:
            plot_title = f"BLiMP Task Groups - {unique_datasets[0]} {unique_models[0]}"
        else:
            plot_title = "BLiMP Task Groups - Multi-Model Comparison"

    # Load data with std
    df, df_std = load_grouped_blimp_data_multi(model_configs, data_dir, return_std=True)

    # Determine groups to plot
    groups_to_plot = dict(BLIMP_GROUPS)

    if include_misc:
        included_tasks = set(t for tasks in BLIMP_GROUPS.values() for t in tasks)
        missing_tasks = [t for t in df.index if t not in included_tasks]
        if missing_tasks:
            groups_to_plot["Miscellaneous"] = missing_tasks

    n_groups = len(groups_to_plot)
    n_rows = (n_groups + groups_per_row - 1) // groups_per_row

    # Create figure
    fig_width = figsize_per_subplot[0] * groups_per_row
    fig_height = figsize_per_subplot[1] * n_rows
    fig, axes = plt.subplots(n_rows, groups_per_row, figsize=(fig_width, fig_height))
    axes = axes.flatten() if n_groups > 1 else [axes]

    # Plot each group
    for i, (group_name, tasks) in enumerate(groups_to_plot.items()):
        # Filter to tasks present in data
        tasks_present = [t for t in tasks if t in df.index]
        if not tasks_present:
            fig.delaxes(axes[i])
            continue

        df_group = df.loc[tasks_present]
        df_group_std = df_std.loc[tasks_present]
        df_scaled = normalize_df(df_group)

        # Create custom annotations with mean±std
        annot_labels = []
        for row_idx in range(len(df_group)):
            row = []
            for col_idx in range(len(df_group.columns)):
                mean_val = df_group.iloc[row_idx, col_idx]
                std_val = df_group_std.iloc[row_idx, col_idx]
                if pd.isna(std_val):
                    row.append(f"{mean_val:.3f}")
                else:
                    row.append(f"{mean_val:.3f}±{std_val:.3f}")
            annot_labels.append(row)
        annot_labels = np.array(annot_labels)

        # Create heatmap
        sns.heatmap(
            df_scaled,
            annot=annot_labels,
            fmt="",
            cmap=cmap,
            cbar=False,
            annot_kws={"fontsize": 9},
            ax=axes[i],
            yticklabels=True
        )

        axes[i].set_title(group_name, fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Vocab Size", fontsize=12)
        axes[i].set_ylabel("")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=11)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=10)

    # Hide unused subplots
    for j in range(n_groups, len(axes)):
        fig.delaxes(axes[j])

    # Add overall title (optional)
    if show_title:
        fig.suptitle(
            plot_title,
            fontsize=18,
            fontweight='bold',
            y=0.995
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    # Save
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_grouped_summary_heatmap(
    model_configs: List[Dict[str, any]] = None,
    dataset: str = None,
    model_size: str = None,
    milestone: str = None,
    vocab_sizes: List[int] = None,
    data_dir: str = "results/data/",
    save_path: Optional[str] = None,
    include_misc: bool = False,
    figsize: Tuple[float, float] = None,
    cmap: str = 'viridis',
    show_title: bool = False,
) -> plt.Figure:
    """Plot group-level BLiMP summary heatmap with boxplot side panel."""
    
    
    # Handle legacy interface
    if model_configs is None:
        if dataset is None or model_size is None or milestone is None or vocab_sizes is None:
            raise ValueError("Either provide model_configs or all of (dataset, model_size, milestone, vocab_sizes)")

        model_configs = [
            {
                'dataset': dataset,
                'model_size': model_size,
                'milestone': milestone,
                'vocab_size': vocab_size,
                'label': f"{vocab_size}"
            }
            for vocab_size in vocab_sizes
        ]
        plot_title = f"Group-wise Summary - {dataset} {model_size}\n(Milestone: {milestone})"
    else:
        unique_datasets = list(set(c['dataset'] for c in model_configs))
        unique_models = list(set(c['model_size'] for c in model_configs))
        if len(unique_datasets) == 1 and len(unique_models) == 1:
            plot_title = f"Group-wise Summary - {unique_datasets[0]} {unique_models[0]}"
        else:
            plot_title = "Group-wise Summary - Multi-Model Comparison"

    # Load data with task-level means and stds (std across seeds for each task)
    df, df_task_std = load_grouped_blimp_data_multi(model_configs, data_dir, return_std=True)

    # Determine groups
    groups_to_plot = dict(BLIMP_GROUPS)

    if include_misc:
        included_tasks = set(t for tasks in BLIMP_GROUPS.values() for t in tasks)
        missing_tasks = [t for t in df.index if t not in included_tasks]
        if missing_tasks:
            groups_to_plot["Miscellaneous"] = missing_tasks

    # Calculate group means and average stds across seeds
    # The std shown represents the average std across repetitions for tasks in each group
    group_means = {}
    group_stds = {}
    for group_name, tasks in groups_to_plot.items():
        tasks_present = [t for t in tasks if t in df.index]
        if tasks_present:
            # Mean of task means within group
            group_means[group_name] = df.loc[tasks_present].mean()
            # Mean of task stds within group (average variability across seeds)
            group_stds[group_name] = df_task_std.loc[tasks_present].mean()

    # Add overall BLiMP average
    group_means["BLiMP Overall"] = df.mean()
    group_stds["BLiMP Overall"] = df_task_std.mean()

    # Create summary DataFrames
    df_summary = pd.DataFrame(group_means).T
    df_summary_std = pd.DataFrame(group_stds).T

    n_groups = len(df_summary)
    n_configs = len(df_summary.columns)

    # Default figsize
    if figsize is None:
        figsize = (16, 10)

    # Normalize row-wise
    df_scaled = normalize_df(df_summary)

    # Create custom annotations with mean ± std
    annot_labels = []
    for i in range(len(df_summary)):
        row = []
        for j in range(len(df_summary.columns)):
            mean_val = df_summary.iloc[i, j]
            std_val = df_summary_std.iloc[i, j]
            row.append(f"{mean_val:.3f}\n±{std_val:.3f}")
        annot_labels.append(row)
    annot_labels = np.array(annot_labels)

    # Prepare data for boxplot (all task values per model config)
    data_for_boxplot = []
    for col in df.columns:
        data_for_boxplot.append(df[col].dropna().values)

    # Create figure with 2 subplots: heatmap and boxplot
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [13, 4]}
    )

    column_labels = df.columns.tolist()
    column_labels_display = [
        label.replace('hybrid_3.7B', 'composite')
        for label in column_labels
    ]

    # Left panel: heatmap
    sns.heatmap(
        df_scaled,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        cbar=False,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax1
    )

    ax1.set_xlabel("Model Configuration", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Linguistic Group", fontsize=16, fontweight='bold')
    ax1.set_xticklabels(column_labels_display, rotation=45, ha='right', fontsize=13)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=13)

    # Right panel: boxplot
    # Generate colors based on dataset and model size
    colors = []
    

    for col_label in column_labels:
        # Try to extract dataset and model_size from the label or config
        # Find matching config
        color = '#666666'  # default gray
        for config in model_configs:
            config_label = config.get('label', f"{config['dataset']}-{config['model_size']}-V{config['vocab_size']//1000}k")
            if config_label == col_label:
                dataset_name = config['dataset']
                dataset_name = dataset_name.replace("hybrid_3.7B", "composite")
                model_size = config['model_size']
                color = DATASET_COLOR_SCHEME.get(dataset_name, {}).get(model_size, '#666666')
                break
        colors.append(color)

    # Create boxplot
    box = ax2.boxplot(
        data_for_boxplot,
        labels=column_labels_display,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        showfliers=True
    )

    # Apply colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Style boxplot
    ax2.set_xticklabels(column_labels_display, rotation=45, ha='right')
    ax2.tick_params(axis='x', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add title (optional)
    if show_title:
        fig.suptitle(plot_title, fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_both_grouped_plots(
    dataset: str,
    model_size: str,
    milestone: str,
    vocab_sizes: List[int],
    data_dir: str = "results/data/",
    output_dir: str = "results/plots/",
    **kwargs
) -> Tuple[plt.Figure, plt.Figure]:
    """Convenience wrapper to produce both grouped BLiMP plot variants."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames
    base_name = f"blimp_groups_{dataset}_{model_size}_m{milestone}"
    grouped_path = os.path.join(output_dir, f"{base_name}_detailed.pdf")
    summary_path = os.path.join(output_dir, f"{base_name}_summary.pdf")

    # Create plots
    fig_grouped = plot_grouped_blimp_heatmaps(
        dataset, model_size, milestone, vocab_sizes,
        data_dir=data_dir,
        save_path=grouped_path,
        **kwargs
    )

    fig_summary = plot_grouped_summary_heatmap(
        dataset, model_size, milestone, vocab_sizes,
        data_dir=data_dir,
        save_path=summary_path,
        **kwargs
    )

    return fig_grouped, fig_summary
