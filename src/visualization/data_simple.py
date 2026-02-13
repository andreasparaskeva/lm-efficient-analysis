"""Consolidated BLiMP data fetch/load helpers."""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict

from src.visualization.blimp_tasks import BLIMP_TASKS


def fetch_and_save_blimp(
    repo,
    dataset: str,
    model_size: str,
    seeds: List[int],
    vocab_sizes: List[int],
    milestones: List,
    data_dir: str = "results/data/",
    mode: str = "best",
    report_missing: bool = True
) -> Dict:
    """Fetch BLiMP results from a repo client and persist consolidated CSVs."""
    os.makedirs(data_dir, exist_ok=True)

    overall_file = os.path.join(data_dir, "blimp_all_overall.csv")
    tasks_file = os.path.join(data_dir, "blimp_all_tasks.csv")

    # Collect new data
    overall_rows = []
    task_rows = []
    missing_runs = []
    task_set = set(BLIMP_TASKS)

    for vocab_size in vocab_sizes:
        for milestone in milestones:
            for seed in seeds:
                try:
                    results = repo.fetch_blimp_results(
                        dataset, model_size, seed, milestone, vocab_size, mode
                    )["results"]

                    # Overall BLiMP accuracy
                    if "blimp" in results and "acc,none" in results["blimp"]:
                        overall_rows.append({
                            'dataset': dataset,
                            'model_size': model_size,
                            'milestone': milestone,
                            'vocab_size': vocab_size,
                            'seed': seed,
                            'acc': results["blimp"]["acc,none"]
                        })

                    # Per-task accuracies
                    for task_key, vals in results.items():
                        norm_task = task_key.lower().replace("blimp_", "")
                        if norm_task in task_set:
                            task_rows.append({
                                'dataset': dataset,
                                'model_size': model_size,
                                'milestone': milestone,
                                'vocab_size': vocab_size,
                                'seed': seed,
                                'task': norm_task,
                                'acc': vals.get("acc,none", np.nan)
                            })

                except Exception as e:
                    missing_runs.append({
                        'dataset': dataset,
                        'model_size': model_size,
                        'milestone': milestone,
                        'vocab_size': vocab_size,
                        'seed': seed,
                        'error': str(e)
                    })

    if not task_rows:
        print(f"✗ No data fetched for {dataset} {model_size}")
        return {
            'success': False,
            'missing': missing_runs,
            'n_fetched': 0,
            'n_expected': len(vocab_sizes) * len(milestones) * len(seeds)
        }

    # Convert to DataFrames
    new_overall = pd.DataFrame(overall_rows)
    new_tasks = pd.DataFrame(task_rows)

    # Load existing data if it exists
    if os.path.exists(overall_file):
        existing_overall = pd.read_csv(overall_file)
        # Remove old data for this experiment
        existing_overall = existing_overall[
            ~((existing_overall['dataset'] == dataset) &
              (existing_overall['model_size'] == model_size))
        ]
    else:
        existing_overall = pd.DataFrame()

    if os.path.exists(tasks_file):
        existing_tasks = pd.read_csv(tasks_file)
        # Remove old data for this experiment
        existing_tasks = existing_tasks[
            ~((existing_tasks['dataset'] == dataset) &
              (existing_tasks['model_size'] == model_size))
        ]
    else:
        existing_tasks = pd.DataFrame()

    # Append new data
    all_overall = pd.concat([existing_overall, new_overall], ignore_index=True)
    all_tasks = pd.concat([existing_tasks, new_tasks], ignore_index=True)

    # Save
    all_overall.to_csv(overall_file, index=False)
    all_tasks.to_csv(tasks_file, index=False)

    n_expected = len(vocab_sizes) * len(milestones) * len(seeds)
    n_fetched = len(set((r['vocab_size'], r['milestone'], r['seed']) for r in overall_rows))

    return {
        'success': True,
        'missing': missing_runs,
        'n_fetched': n_fetched,
        'n_expected': n_expected,
        'dataset': dataset,
        'model_size': model_size
    }


def load_blimp(
    dataset: str,
    model_size: str,
    data_dir: str = "results/data/"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load consolidated BLiMP overall/task rows for one dataset+model size."""
    overall_file = os.path.join(data_dir, "blimp_all_overall.csv")
    tasks_file = os.path.join(data_dir, "blimp_all_tasks.csv")

    # Load all data
    all_overall = pd.read_csv(overall_file)
    all_tasks = pd.read_csv(tasks_file)

    # Filter for this experiment
    overall_df = all_overall[
        (all_overall['dataset'] == dataset) &
        (all_overall['model_size'] == model_size)
    ].copy()

    task_df = all_tasks[
        (all_tasks['dataset'] == dataset) &
        (all_tasks['model_size'] == model_size)
    ].copy()

    return overall_df, task_df


def load_blimp_multi(
    configs: List[Dict[str, any]],
    data_dir: str = "results/data/"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate BLiMP rows for multiple dataset/model configs."""
    all_overall = []
    all_tasks = []

    for config in configs:
        overall, tasks = load_blimp(config['dataset'], config['model_size'], data_dir)
        all_overall.append(overall)
        all_tasks.append(tasks)

    combined_overall = pd.concat(all_overall, ignore_index=True)
    combined_tasks = pd.concat(all_tasks, ignore_index=True)

    return combined_overall, combined_tasks


def list_experiments(data_dir: str = "results/data/") -> pd.DataFrame:
    """List stored experiments with vocab/milestone/seed coverage."""
    tasks_file = os.path.join(data_dir, "blimp_all_tasks.csv")

    if not os.path.exists(tasks_file):
        print("No data found. Run fetch_and_save_blimp() first.")
        return pd.DataFrame()

    all_tasks = pd.read_csv(tasks_file)

    # Group by experiment
    grouped = all_tasks.groupby(['dataset', 'model_size']).agg({
        'vocab_size': lambda x: sorted(x.unique().tolist()),
        'milestone': lambda x: sorted([str(m) for m in x.unique()]),
        'seed': lambda x: sorted(x.unique().tolist()),
        'task': 'count'
    }).reset_index()

    grouped.columns = ['dataset', 'model_size', 'vocab_sizes', 'milestones', 'seeds', 'n_datapoints']

    return grouped


def get_storage_info(data_dir: str = "results/data/") -> Dict:
    """Return row counts and file sizes for consolidated BLiMP storage."""
    overall_file = os.path.join(data_dir, "blimp_all_overall.csv")
    tasks_file = os.path.join(data_dir, "blimp_all_tasks.csv")

    info = {}

    if os.path.exists(overall_file):
        df = pd.read_csv(overall_file)
        size_kb = os.path.getsize(overall_file) / 1024
        info['overall'] = {
            'file': overall_file,
            'rows': len(df),
            'size_kb': f"{size_kb:.1f}",
            'experiments': len(df.groupby(['dataset', 'model_size']))
        }

    if os.path.exists(tasks_file):
        df = pd.read_csv(tasks_file)
        size_kb = os.path.getsize(tasks_file) / 1024
        info['tasks'] = {
            'file': tasks_file,
            'rows': len(df),
            'size_kb': f"{size_kb:.1f}",
            'experiments': len(df.groupby(['dataset', 'model_size']))
        }

    return info
