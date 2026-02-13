#!/bin/bash
set -euo pipefail

# Seeds list (you can pass this as an argument or hardcode it here)
SEEDS=(0)

for SEED in "${SEEDS[@]}"; do
    echo "Running training with seed ${SEED}"
    python -m src.models.training.tokenizer_exp --seed "${SEED}" --milestone-store local
done
