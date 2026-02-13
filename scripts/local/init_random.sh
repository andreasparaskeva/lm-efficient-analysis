#!/bin/bash
set -euo pipefail

# Seeds for random initialization runs
SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
    echo "Creating random initialization with seed ${SEED}"
    python -m src.models.training.tokenizer_exp --seed "${SEED}" --init --milestone-store local
done
