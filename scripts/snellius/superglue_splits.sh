#!/bin/bash
#SBATCH --job-name=superglue-splits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=3:00:00
#SBATCH --chdir=/projects/0/prjs1537/projects/data-efficient-analysis

source ~/.bashrc
conda activate slm

# Export paths
export PYTHONPATH=/home/apareskeva/workspace/data-efficient-analysis:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Define logs
LOG_DIR="/home/apareskeva/workspace/data-efficient-analysis/scripts/snellius"
OUT_LOG="${LOG_DIR}/out/${SLURM_JOB_NAME}.out"
ERR_LOG="${LOG_DIR}/err/${SLURM_JOB_NAME}.err"

# Define the list of SuperGLUE tasks
TASKS=("boolq" "cb" "copa" "multirc" "record" "rte" "wic" "wsc")

# Loop through tasks
for TASK in "${TASKS[@]}"; do
  echo "Processing task: $TASK" >> "$OUT_LOG"
  srun \
    --output="$OUT_LOG" \
    --error="$ERR_LOG" \
    python -m src.data.create_superglue_splits \
      --task "$TASK" \
      --output_dir ./data/superglue_splits \
      --split_ratio 0.1
done
