"""Tokenizer-size comparison plots for BLiMP learning curves."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from matplotlib.ticker import FuncFormatter

from src.visualization.data_simple import load_blimp
from src.visualization.utils import adjust_lightness


# Default color palette for vocab sizes
VOCAB_COLOR_SCHEME = {
    8000:  "#0f4e77",
    16000: "#372516",
    32000: "#89a377",
    50257: "#f3af5c",
}


def load_tokenizer_data(
    dataset: str,
    model_size: str,
    vocab_sizes: List[int],
    milestone: str = "final",
    data_dir: str = "results/data/"
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray], List]:
    """Load BLiMP overall data and aggregate mean/std curves by vocab size."""
    # Load data for this dataset/model
    overall_df, _ = load_blimp(dataset, model_size, data_dir)

    # Filter for requested vocab sizes
    overall_df = overall_df[overall_df['vocab_size'].isin(vocab_sizes)].copy()

    # Add numeric milestone column
    overall_df['milestone_num'] = pd.to_numeric(overall_df['milestone'], errors='coerce')

    # Get unique numeric milestones (excluding 'final')
    numeric_milestones = sorted([
        m for m in overall_df['milestone_num'].dropna().unique()
        if not np.isnan(m)
    ])

    # Calculate mean and std for each vocab size at each milestone
    curves_mean = {}
    curves_std = {}

    for vocab_size in vocab_sizes:
        vocab_data = overall_df[overall_df['vocab_size'] == vocab_size]

        means = []
        stds = []

        for milestone_val in numeric_milestones:
            milestone_data = vocab_data[vocab_data['milestone_num'] == milestone_val]['acc']
            if len(milestone_data) > 0:
                means.append(milestone_data.mean())
                stds.append(milestone_data.std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        curves_mean[vocab_size] = np.array(means)
        curves_std[vocab_size] = np.array(stds)

    return overall_df, curves_mean, curves_std, numeric_milestones


def plot_tokenizer_comparison(
    dataset: str,
    model_size: str,
    vocab_sizes: List[int],
    data_dir: str = "results/data/",
    show_shading: bool = True,
    spread_mode: str = "minmax",
    save_path: Optional[str] = None,
    yscale_mode: str = "linear",
    power_factor: float = 1,
    align_yaxes: bool = False,
    color_scheme: Optional[Dict[int, str]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot two-panel BLiMP curves comparing tokenizer vocabulary sizes."""
    # Load data
    overall_df, curves_mean, curves_std, numeric_milestones = load_tokenizer_data(
        dataset, model_size, vocab_sizes, data_dir=data_dir
    )

    # Use provided color scheme or default
    if color_scheme is None:
        color_scheme = VOCAB_COLOR_SCHEME

    # Styling
    sns.set_context("notebook")
    sns.set_style("white")

    plt.rcParams.update({
        "axes.labelweight": "bold",
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
    })

    # Figure layout
    fig = plt.figure(figsize=(15, 7))

    # Left panel: learning curves
    ax = fig.add_axes([0.06, 0.12, 0.70, 0.76])

    # Right panel: final scores
    summary_ax = fig.add_axes([0.78, 0.12, 0.12, 0.76])
    summary_ax.set_title("Final Score", fontsize=13, weight="bold")

    # Left panel: learning curves
    for vocab_size in vocab_sizes:
        if vocab_size not in curves_mean:
            continue

        color = color_scheme.get(vocab_size, '#555555')
        mean_vals = curves_mean[vocab_size]
        std_vals = curves_std[vocab_size]

        # Plot learning curve
        ax.plot(
            numeric_milestones[:len(mean_vals)],
            mean_vals,
            marker="o",
            linewidth=2.8,
            color=color,
            label=f"Vocab {vocab_size:,}",
            zorder=4,
        )

        # Shading
        if show_shading:
            if spread_mode == "std":
                lower = mean_vals - std_vals
                upper = mean_vals + std_vals

            elif spread_mode == "minmax":
                lower, upper = [], []
                for milestone_val in numeric_milestones[:len(mean_vals)]:
                    vals = overall_df.loc[
                        (overall_df["vocab_size"] == vocab_size) &
                        (overall_df["milestone_num"] == float(milestone_val)),
                        "acc"
                    ].to_numpy(float)
                    lower.append(np.nanmin(vals) if vals.size else np.nan)
                    upper.append(np.nanmax(vals) if vals.size else np.nan)
                lower = np.array(lower)
                upper = np.array(upper)
            else:
                continue

            mask = np.isfinite(lower) & np.isfinite(upper)
            if mask.any():
                ax.fill_between(
                    np.array(numeric_milestones[:len(mean_vals)])[mask],
                    lower[mask],
                    upper[mask],
                    color=adjust_lightness(color, 1.1),
                    alpha=0.35,
                    linewidth=1.2,
                    edgecolor=color,
                    zorder=2
                )

    # Right panel: final scores
    xs = []
    labels = []
    all_final_vals = []

    for i, vocab_size in enumerate(sorted(vocab_sizes)):
        color = color_scheme.get(vocab_size, '#555555')

        # Get final scores
        rows = overall_df[
            (overall_df["vocab_size"] == vocab_size) &
            (overall_df["milestone"].astype(str) == "final")
        ]

        if rows.empty:
            continue

        vals = rows["acc"].to_numpy(float)
        fmin, fmax = vals.min(), vals.max()
        fmean = vals.mean()

        xs.append(i)
        labels.append(f"V={vocab_size//1000}k")
        all_final_vals.extend(vals.tolist())

        # Vertical line for range
        summary_ax.vlines(i, fmin, fmax, color=color, linewidth=2)
        # X marker for mean
        summary_ax.scatter(i, fmean, color=color, marker="x",
                          s=90, linewidth=2.5, zorder=5)

    summary_ax.set_xlim(-0.5, len(xs) - 0.5)
    summary_ax.set_xticks(xs)
    summary_ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)

    # Y-axis alignment
    if align_yaxes:
        # Same y-limits for both panels
        ymin, ymax = ax.get_ylim()
        summary_ax.set_ylim(ymin, ymax)
    else:
        # Independent zoom to final score range
        if all_final_vals:
            lo = min(all_final_vals)
            hi = max(all_final_vals)
            span = hi - lo if hi > lo else 0.01
            pad = span * 0.15
            summary_ax.set_ylim(lo - pad, hi + pad)

    summary_ax.yaxis.tick_right()
    summary_ax.yaxis.set_label_position("right")
    summary_ax.set_ylabel("Final BLiMP Accuracy", labelpad=18)

    # Left-axis y-scaling
    if yscale_mode == "logit":
        forward = lambda y: np.log(y / (1 - y + 1e-10))
        inverse = lambda y: 1 / (1 + np.exp(-y))
        ax.set_yscale("function", functions=(forward, inverse))
    elif yscale_mode == "power" and power_factor != 1:
        forward = lambda y: np.power(y, power_factor)
        inverse = lambda y: np.power(y, 1/power_factor)
        ax.set_yscale("function", functions=(forward, inverse))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    summary_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    # Final styling
    # Format dataset name for title
    if dataset == "hybrid_3.7B":
        dataset_title = "Composite Corpus 3.7B"
    else:
        dataset_title = dataset

    ax.set_title(f"BLiMP Learning Curves — {dataset_title} {model_size.upper()}")
    ax.set_xlabel("Anchor (in million tokens)", fontweight='bold')
    ax.set_ylabel("Average BLiMP Accuracy", fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Save
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        png_path = save_path.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, (ax, summary_ax)
