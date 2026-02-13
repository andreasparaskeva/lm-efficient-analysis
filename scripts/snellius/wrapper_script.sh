#!/bin/bash
process_name=$1
dataset_name=$2
jobname=''$1'_'$2''
time=48:00:00

# Path to your wrapper script
WRAPPER_SCRIPT='scripts/snellius/'$1'.sh'

# Seeds list (you can pass this as an argument or hardcode it here)
SEEDS=(1 2)  # List of seeds for the job array (e.g., 3 seeds)
SEEDS_STR=$(IFS=,; echo "${SEEDS[*]}")
export SEEDS_STR
# Get the number of seeds
NUM_SEEDS=${#SEEDS[@]}

# Submit the job using sbatch
sbatch --job-name=$jobname --array=0-$(($NUM_SEEDS-1)) --export=all --time=$time "$WRAPPER_SCRIPT"