import argparse
from datasets import load_dataset, DatasetDict
from pathlib import Path

def main(task_name: str, output_dir: str, split_ratio: float = 0.1, seed: int = 42):
    assert task_name in [
        "boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"
    ], f"Invalid SuperGLUE task: {task_name}"

    # Load full SuperGLUE task
    dataset = load_dataset("super_glue", task_name)

    # Check if 'label' exists and is usable for stratification
    train_dataset = dataset["train"]
    if "label" in train_dataset.column_names:
        labels = train_dataset["label"]
        if all(l is not None for l in labels):
            try:
                split = train_dataset.train_test_split(
                    test_size=split_ratio,
                    seed=seed,
                    stratify_by_column="label"
                )
            except Exception as e:
                print(f"⚠️ Stratified split failed: {e}, falling back to random split")
                split = train_dataset.train_test_split(
                    test_size=split_ratio,
                    seed=seed
                )
        else:
            print("⚠️ Found None in labels — falling back to random split")
            split = train_dataset.train_test_split(
                test_size=split_ratio,
                seed=seed
            )
    else:
        print("⚠️ No 'label' column found — falling back to random split")
        split = train_dataset.train_test_split(
            test_size=split_ratio,
            seed=seed
        )

    dev_dataset = split["test"]
    test_dataset = dataset["validation"]  # official SuperGLUE dev

    # Create output directory
    output_dir = Path(output_dir) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    split["train"].to_json(output_dir / "train.jsonl")
    dev_dataset.to_json(output_dir / "dev.jsonl")
    test_dataset.to_json(output_dir / "test.jsonl")

    print(f"✅ Done. Saved train/dev/test splits to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split SuperGLUE train and save test")
    parser.add_argument("--task", type=str, required=True, help="SuperGLUE task name (e.g., rte, cb, boolq)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save splits")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="Dev split ratio from train (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args.task, args.output_dir, args.split_ratio, args.seed)
