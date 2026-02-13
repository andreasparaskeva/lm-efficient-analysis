"""
Utility functions for BLiMP analysis and plotting
"""
import numpy as np
from matplotlib.colors import to_rgb


def adjust_lightness(color, amount=0.5):
    """
    Adjust the lightness of a color.
    
    Args:
        color: Color in any matplotlib-compatible format
        amount: Multiplier for RGB values (< 1 = darker, > 1 = lighter)
        
    Returns:
        RGB tuple
    """
    c = np.array(to_rgb(color))
    c = np.clip(c * amount, 0, 1)
    return tuple(c)


def compute_group_accuracy(milestone_data, group_tasks, vocab_size):
    """
    Compute average accuracy for a group of tasks.
    
    Args:
        milestone_data: Dictionary mapping tasks to vocab_size-specific accuracies
        group_tasks: List of task names in the group
        vocab_size: Vocabulary size to retrieve
        
    Returns:
        Mean accuracy across tasks in the group
    """
    accs = []
    for task in group_tasks:
        if task in milestone_data and vocab_size in milestone_data[task]:
            acc = milestone_data[task][vocab_size]
            if not np.isnan(acc):
                accs.append(acc)
    
    return np.mean(accs) if accs else np.nan


def extract_numeric_milestones(milestones, max_value=None):
    """
    Extract numeric milestones from a list, filtering out non-numeric values.

    Args:
        milestones: List of milestone values (can be mixed types)
        max_value: Optional maximum value to include

    Returns:
        Sorted list of numeric milestones
    """
    numeric_milestones = []
    for milestone in milestones:
        try:
            milestone_val = float(milestone)
            if max_value is None or milestone_val <= max_value:
                numeric_milestones.append(milestone_val)
        except (ValueError, TypeError):
            pass  # Skip non-numeric milestones like 'final'
    return sorted(numeric_milestones)


def match_milestone_in_df(df, milestone, column='milestone'):
    """
    Create a boolean mask for matching milestones in a DataFrame.

    Handles both string and numeric representations of milestones.

    Args:
        df: DataFrame to filter
        milestone: Milestone value to match
        column: Column name containing milestone values (default: 'milestone')

    Returns:
        Boolean mask for matching rows
    """
    # Try matching as both the original value and as string/int
    try:
        milestone_int = int(float(milestone))
        return (df[column] == milestone) | (df[column] == str(milestone_int)) | (df[column] == float(milestone))
    except (ValueError, TypeError):
        # For non-numeric milestones like 'final'
        return df[column] == milestone
