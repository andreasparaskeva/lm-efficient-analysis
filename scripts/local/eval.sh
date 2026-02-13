#!/bin/bash

# Get dataset, model size, anchor size, seed, and tokenizer size from arguments
dataset="$1"
model_size="$2"
anchor_size="$3"
seed="$4"
tokenizer_vocab_size="${5:-8000}"

python -m src.models.eval.blimp_flexible \
  --source local \
  --dataset_name "$dataset" \
  --model_size "$model_size" \
  --anchor_size "$anchor_size" \
  --seed "$seed" \
  --tokenizer_vocab_size "$tokenizer_vocab_size"
