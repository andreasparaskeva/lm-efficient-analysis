"""
DDP Worker - Runs the actual training in a distributed context.
This file is launched by torchrun with multiple processes.
"""
import os
import sys
import json
import argparse
import torch
import torch.distributed as dist

# Import your training logic
from tokenizer_exp import (
    set_seed,
    load_model_and_tokenizer,
    prepare_datasets,
    train_for_anchor,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-config', type=str, required=True, help='Path to job config JSON')
    parser.add_argument('--anchor', type=str, required=True, help='Anchor value')
    args = parser.parse_args()
    
    # Load job configuration
    with open(args.job_config, 'r') as f:
        job = json.load(f)
    
    # Initialize distributed training (torchrun sets these env vars)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    # Only print from rank 0
    if local_rank == 0:
        print(f"[DDP Worker] Initialized with world_size={dist.get_world_size()}, rank={dist.get_rank()}")
    
    # Run the training
    set_seed(int(job["seed"]))
    base_model, tokenizer, output_dir = load_model_and_tokenizer(job)
    train_dataset, eval_dataset = prepare_datasets(job, tokenizer)
    
    base_output_dir = f"{output_dir}/tokenizer-{job['tokenizer_vocab_size']}-seed-{job['seed']}"
    
    train_for_anchor(
        job=job,
        base_model=base_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        anchor=args.anchor,
        seed=job["seed"],
        base_output_dir=base_output_dir
    )
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()