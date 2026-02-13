#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=6:00:00
#SBATCH --chdir=/projects/0/prjs1537/projects/data-efficient-analysis
#SBATCH --job-name=blimp_eval

# BLiMP evaluation script for Snellius
# Self-contained: configure parameters below and submit with: sbatch scripts/snellius/eval.sh

# ============================================================
# CONFIGURATION - Edit these arrays to control what to evaluate
# ============================================================
datasets=("hybrid_3.7B")  # Options: babylm3, tinystories, hybrid_3.7B
model_sizes=("180m")   # Options: 20m, 60m, 180m
seeds=(1 2)
tokenizer_vocab_sizes=(50257)
# seeds=(0 1 2)
# tokenizer_vocab_sizes=(8000 16000 32000 50257)
# tokenizer_vocab_sizes=(16000 50257)
# Leave anchors empty to upload all anchors for each combination
# Or specify: anchor_sizes=("25" "50" "100")
anchor_sizes=(2000 final)
# anchor_sizes=(25 50 75 100 250 500 750 1000 1250 1500 1750 2000 final)

# ============================================================
# SETUP
# ============================================================
export PYTHONPATH=/home/apareskeva/workspace/data-efficient-analysis:$PYTHONPATH

# Activate environment
source ~/.bashrc
conda activate slm

# Directories
RESULTS_BASE_DIR="/projects/0/prjs1537/projects/data-efficient-analysis/results/blimp"
OUTPUT_BASE_DIR="/projects/0/prjs1537/projects/data-efficient-analysis/output"
LOG_DIR="/home/apareskeva/workspace/data-efficient-analysis/scripts/snellius"

# ============================================================
# HELPER FUNCTION
# ============================================================
check_results_exist() {
    local dataset=$1
    local model_size=$2
    local seed=$3
    local tokenizer=$4
    local anchor_size=$5

    local results_file="${RESULTS_BASE_DIR}/${dataset}/${model_size}/seed-${seed}/tok-${tokenizer}/${anchor_size}/blimp_results.json"

    if [ -f "$results_file" ]; then
        return 0  # File exists
    else
        return 1  # File does not exist
    fi
}

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
echo "============================================================"
echo "BLiMP Evaluation Batch"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"
echo ""

total_evaluated=0
total_skipped=0

for dataset in "${datasets[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        for seed in "${seeds[@]}"; do
            for tokenizer in "${tokenizer_vocab_sizes[@]}"; do
                for anchor_size in "${anchor_sizes[@]}"; do

                    # Check if results already exist
                    if check_results_exist "$dataset" "$model_size" "$seed" "$tokenizer" "$anchor_size"; then
                        echo "⏭️  Skipping: $dataset/$model_size/seed-$seed/tok-$tokenizer/anchor-$anchor_size (results exist)"
                        ((total_skipped++))
                        continue
                    fi

                    echo ""
                    echo ">>> Running: $dataset/$model_size/seed-$seed/tok-$tokenizer/anchor-$anchor_size"

                    srun --error="${LOG_DIR}/err/blimp/${dataset}_${model_size}_seed${seed}_tok${tokenizer}_anchor${anchor_size}.err" \
                         --output="${LOG_DIR}/out/blimp/${dataset}_${model_size}_seed${seed}_tok${tokenizer}_anchor${anchor_size}.out" \
                    python -m src.models.eval.blimp_flexible \
                        -d "${dataset}" \
                        -m "${model_size}" \
                        -a "${anchor_size}" \
                        -s "${seed}" \
                        -t "${tokenizer}" \
                        --source local \
                        --output-base-dir "${OUTPUT_BASE_DIR}"

                    if [ $? -eq 0 ]; then
                        echo "✅ Completed: $dataset/$model_size/seed-$seed/tok-$tokenizer/anchor-$anchor_size"
                        ((total_evaluated++))
                    else
                        echo "❌ Failed: $dataset/$model_size/seed-$seed/tok-$tokenizer/anchor-$anchor_size"
                    fi
                    echo ""

                done
            done
        done
    done
done

echo "============================================================"
echo "Summary"
echo "============================================================"
echo "✅ Evaluated: $total_evaluated"
echo "⏭️  Skipped: $total_skipped"
echo "============================================================"
