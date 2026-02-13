#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --chdir=/projects/0/prjs1537/projects/data-efficient-analysis

source ~/.bashrc
conda activate slm

# Convert SEEDS_STR back into an array
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export PYTHONPATH=/home/apareskeva/workspace/data-efficient-analysis:$PYTHONPATH

# Define dynamic output paths
LOG_DIR="/home/apareskeva/workspace/data-efficient-analysis/scripts/snellius"
OUT_LOG="${LOG_DIR}/out/${SLURM_JOB_NAME}_seed${SEED}.out"
ERR_LOG="${LOG_DIR}/err/${SLURM_JOB_NAME}_seed${SEED}.err"

# Run the Python script via srun, setting output paths
srun \
  --output="$OUT_LOG" \
  --error="$ERR_LOG" \
  python -m src.models.training.tokenizer_exp --seed "$SEED" --milestone-store local