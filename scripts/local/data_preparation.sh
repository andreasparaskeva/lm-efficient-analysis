#!/bin/bash
set -euo pipefail

# One-shot guard: this script is not meant to be rerun on an existing prepared tree.
if [ -d "data/babylm3/train_clean" ]; then
  echo "Error: data/babylm3/train_clean already exists. data_preparation.sh is one-shot." >&2
  exit 1
fi

# Download local datasets used in experiments.
./scripts/get_data.sh babylm3
./scripts/get_data.sh tinystories

# Build the composite dataset after base downloads are available.
python -m src.data.hybrid_dataset

# Clean and tokenize datasets into the format expected by training.
python -m src.data.clean_and_tokenize --dataset babylm3
python -m src.data.clean_and_tokenize --dataset tinystories
python -m src.data.clean_and_tokenize --dataset hybrid_3.7B

# One-shot normalization for downstream code expecting train_clean.
if [ ! -d "data/babylm3/train_100M_clean" ]; then
  echo "Error: expected data/babylm3/train_100M_clean was not created." >&2
  exit 1
fi

mv data/babylm3/train_100M_clean data/babylm3/train_clean
