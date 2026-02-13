#!/bin/bash
#SBATCH --job-name=hybrid_dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=16:00:00
#SBATCH --chdir=/projects/0/prjs1537/projects/data-efficient-analysis


export PYTHONPATH=/home/apareskeva/workspace/data-efficient-analysis:$PYTHONPATH
export HF_DATASETS_CACHE=/projects/0/prjs1537/cache
source ~/.bashrc
conda activate slm

# Define dynamic output paths
LOG_DIR="/home/apareskeva/workspace/data-efficient-analysis/scripts/snellius"
ERR_LOG="${LOG_DIR}/err/${SLURM_JOB_NAME}.err"
OUT_LOG="${LOG_DIR}/out/${SLURM_JOB_NAME}.out"

# Run the Python script via srun, setting output paths
srun \
  --output="$OUT_LOG" \
  --error="$ERR_LOG" \
  python -m src.data.hybrid_dataset

