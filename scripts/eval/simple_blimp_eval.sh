#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Modify these variables to control the evaluation runs.
DATASETS=(
  "tinystories"
  # "babylm3"
)

MODEL_SIZES=(
  "20m"
  # "60m"
)

SEEDS=(
  0
  # 1
  # 2
)

TOKENIZER_VOCAB_SIZES=(
  16000
  # 50257
)

ANCHOR_SIZES=(
  2000
  "final"
)

# Base directory for output, relative to the project root
OUTPUT_BASE_DIR="./output"

# Python executable
PYTHON_BIN="python"
# --- End Configuration ---

echo "Starting simplified BLiMP evaluation..."
echo "Datasets: ${DATASETS[*]}"
echo "Model Sizes: ${MODEL_SIZES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Tokenizer Vocab Sizes: ${TOKENIZER_VOCAB_SIZES[*]}"
echo "Anchor Sizes: ${ANCHOR_SIZES[*]}"
echo "----------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
  for model_size in "${MODEL_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for tokenizer_vocab_size in "${TOKENIZER_VOCAB_SIZES[@]}"; do
        for anchor_size in "${ANCHOR_SIZES[@]}"; do
          echo "Running BLiMP for:"
          echo "  Dataset: $dataset"
          echo "  Model Size: $model_size"
          echo "  Seed: $seed"
          echo "  Tokenizer Vocab Size: $tokenizer_vocab_size"
          echo "  Anchor Size: $anchor_size"
          echo "----------------------------------------------------"

          # Construct the command to run blimp_flexible.py
          # Note: We are assuming 'local' source for simplicity as in wrapper_eval.sh example
          # If you need 'hf' source, you might need to adjust this.
          python -m src.models.eval.blimp_flexible -d "$dataset" -m "$model_size" -a "$anchor_size" -s "$seed" -t "$tokenizer_vocab_size" --source local 

          echo "----------------------------------------------------"
          echo "Completed run for $dataset, $model_size, seed $seed, tok $tokenizer_vocab_size, anchor $anchor_size"
          echo "----------------------------------------------------"
        done
      done
    done
  done
done

echo "Simplified BLiMP evaluation finished."
