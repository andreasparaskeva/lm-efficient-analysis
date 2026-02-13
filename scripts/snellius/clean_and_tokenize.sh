#!/bin/bash
#SBATCH --job-name=clean_tokenize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=03:00:00
#SBATCH --chdir=/projects/0/prjs1537/projects/data-efficient-analysis

source ~/.bashrc
conda activate slm  # Replace with your actual environment

# Set dataset name as an argument
DATASET="hybrid_3.7B"  # Change this to the desired dataset: babylm, tinystories, etc.
# DATASET="hybrid_corpus_1_5B"
export PYTHONPATH=/home/apareskeva/workspace/data-efficient-analysis:$PYTHONPATH

# Define dynamic output paths
LOG_DIR="/home/apareskeva/workspace/data-efficient-analysis/scripts/snellius"
OUT_LOG="${LOG_DIR}/out/clean_tokenize_${DATASET}.out"
ERR_LOG="${LOG_DIR}/err/clean_tokenize_${DATASET}.err"

# Run the Python script via srun, setting output paths
srun \
  --output="$OUT_LOG" \
  --error="$ERR_LOG" \
  python -m src.data.clean_and_tokenize --dataset $DATASET