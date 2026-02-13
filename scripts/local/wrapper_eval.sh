#!/bin/bash

datasets=("hybrid_3.7B")  # Options: babylm3, tinystories, hybrid_3.7B
model_sizes=("180m")      # Options: 20m, 60m, 180m
seeds=(1 2)
tokenizer_vocab_sizes=(50257)
# seeds=(0 1 2)
# tokenizer_vocab_sizes=(8000 16000 32000 50257)
# Leave anchors empty to evaluate all anchors for each combination
# Or specify: anchor_sizes=("25" "50" "100")
anchor_sizes=(2000 final)
# anchor_sizes=(25 50 75 100 250 500 750 1000 1250 1500 1750 2000 final)

# Loop over each dataset, model size, and anchor size
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model_size in "${model_sizes[@]}"; do
            for tokenizer in "${tokenizer_vocab_sizes[@]}"; do
                for anchor_size in "${anchor_sizes[@]}"; do
                    ./scripts/local/eval.sh "$dataset" "$model_size" "$anchor_size" "$seed" "$tokenizer"
                done
            done
        done
    done
done
