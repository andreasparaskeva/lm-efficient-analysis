import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr, linregress
from typing import Dict, List, Optional, Tuple, Literal


def load_blimp_overall_data(data_dir: str = "./results/data/") -> pd.DataFrame:
    """Load BLiMP overall results data."""
    csv_path = os.path.join(data_dir, "blimp_all_overall.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BLiMP results file not found: {csv_path}")
    return pd.read_csv(csv_path)

def calculate_correlation_and_r2(
    df: pd.DataFrame,
    milestone_col: str = 'milestone',
    accuracy_col: str = 'acc'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Spearman correlation and R2 score for each anchor (milestone)
    against the final model performance.

    Args:
        df: DataFrame containing BLiMP overall results.
        milestone_col: Name of the column indicating milestones (anchors).
        accuracy_col: Name of the column indicating accuracy.

    Returns:
        Two DataFrames: one for Spearman correlations and one for R2 scores,
        indexed by milestone, with a 'global' column.
    """
    spearman_results = []
    r2_results = []

    # Filter for final performances
    final_performances = df[df[milestone_col] == 'final'].set_index(
        ['dataset', 'model_size', 'vocab_size', 'seed']
    )[accuracy_col].rename('final_acc')

    # Get unique numeric milestones (anchors)
    numeric_milestones = sorted([
        m for m in df[milestone_col].unique() if str(m).isdigit()
    ], key=int)

    for anchor_milestone in numeric_milestones:
        anchor_df = df[df[milestone_col] == str(anchor_milestone)]
        
        # Merge with final performances
        merged_df = pd.merge(
            anchor_df,
            final_performances,
            left_on=['dataset', 'model_size', 'vocab_size', 'seed'],
            right_index=True,
            how='inner'
        )

        if merged_df.empty:
            # print(f"No data for anchor {anchor_milestone}, skipping.") # Removed debug print
            continue

        # Global calculations
        global_spearman = np.nan
        global_r2 = np.nan

        if len(merged_df) >= 2: # Need at least 2 points for correlation/regression
            global_spearman, _ = spearmanr(merged_df[accuracy_col], merged_df['final_acc'])
            
            # Linear Regression for R2
            slope, intercept, r_value, p_value, std_err = linregress(
                merged_df[accuracy_col], merged_df['final_acc']
            )
            global_r2 = r_value**2
        
        spearman_results.append({'milestone': anchor_milestone, 'global': global_spearman})
        r2_results.append({'milestone': anchor_milestone, 'global': global_r2})

    spearman_df = pd.DataFrame(spearman_results).set_index('milestone')
    r2_df = pd.DataFrame(r2_results).set_index('milestone')

    return spearman_df, r2_df


def plot_metric(
    df: pd.DataFrame,
    ylabel: str,
    save_path: Optional[str] = None
):
    """
    Plots a single metric (Spearman or R2) globally, with improved styling,
    and saves it as both PNG and PDF.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['global'], marker='o', linestyle='-', color='blue')
    
    # Axis labels with bold font and increased size
    plt.xlabel("Anchor (in million tokens)", fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Tick labels with increased size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(df.index, rotation=45) # Rotate for better readability if crowded
    plt.ylim(-0.1, 1.1)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=300)
        
        # Save as PDF
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path)

    plt.close() # Close plot to free memory






if __name__ == "__main__":
    # Define paths
    data_dir = "./results/data/"
    plots_dir = "./results/plots/"
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    blimp_df = load_blimp_overall_data(data_dir)

    # Calculate metrics
    spearman_df, r2_df = calculate_correlation_and_r2(blimp_df)

    # Save scores to CSV
    os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists for saving CSVs
    spearman_df.to_csv(os.path.join(data_dir, "blimp_global_spearman.csv"))
    r2_df.to_csv(os.path.join(data_dir, "blimp_global_r2.csv"))

    # Plotting
    # Plot 1: Global Spearman Score
    plot_metric(
        spearman_df,
        "Spearman Correlation",
        os.path.join(plots_dir, "blimp_global_spearman.png")
    )

    # Plot 2: Global R2 Score
    plot_metric(
        r2_df,
        "R2 Score",
        os.path.join(plots_dir, "blimp_global_r2.png")
    )

    print("Analysis complete. Plots saved to ./results/plots/")
