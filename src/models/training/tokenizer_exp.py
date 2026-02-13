import os
import json
import math
import shutil
import logging
import argparse
from pathlib import Path
import types
import copy
import yaml
import random
import numpy as np

import torch
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import Subset
from transformers import Trainer, DataCollatorForLanguageModeling
from deephyper.evaluator import Evaluator, profile

from src.utils.model_utils import get_model
from src.utils.data_utils import get_dataset
from src.utils.training_utils import (
    LossCallback,
    get_training_args,
    return_jobs_done_to_csv,
    HFTokenMilestoneCallback,
    DelayedEarlyStoppingCallback,
    resolve_effective_batch_and_accum,
)
from huggingface_hub import HfApi

# === Constants ===
MODEL_CONFIG_PATHS = {
    "20m": "./models/configs/Llama-20M.yaml",
    "60m": "./models/configs/Llama-60M.yaml",
    "180m": "./models/configs/Llama-180M.yaml",
    "190m": "./models/configs/Llama-190M.yaml",
    "400m": "./models/configs/Llama-400M.yaml",
}
LOG_DIR = "./logs"
ENV_FILE = ".env"

# Load env
load_dotenv(ENV_FILE)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE")
if HF_TOKEN and not HF_NAMESPACE:
    HF_NAMESPACE = HfApi().whoami(token=HF_TOKEN)["name"]

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def load_model_and_tokenizer(job):
    # NOTE: keep original mapping logic
    model_config_file = MODEL_CONFIG_PATHS.get(job["model_config"], "20m")
    with open(model_config_file, "r") as f:
        config = yaml.safe_load(f)
    base_model, tokenizer, output_dir = get_model(
        model_config=config,
        dataset=job["dataset"],
        tokenizer_vocab_size=job["tokenizer_vocab_size"],
        checkpoint=False,
        init=job["init"],
    )
    tokenizer.model_max_length = job["seq_length"]
    base_model.resize_token_embeddings(len(tokenizer))
    return base_model, tokenizer, output_dir


def prepare_datasets(job, tokenizer):
    train_dataset, eval_dataset = get_dataset(job["dataset"], job["seq_length"], tokenizer)
    print(f"Total tokens in dataset: {train_dataset.total_tokens / 1e6:.2f}M")
    return train_dataset, eval_dataset


def train_for_anchor(job, base_model, tokenizer, train_dataset, eval_dataset, anchor, seed, base_output_dir):
    # logger = logging.getLogger(__name__)
    # set_seed(seed)
    logger = logging.getLogger(__name__)
    set_seed(seed)

    # Configure logging (write one log file per job+seed)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_file = f"{LOG_DIR}/{job['model_config']}-{job['dataset']}-tok_{job['tokenizer_vocab_size']}-seed_{seed}"
    log_file += "-full.log" if int(anchor) == -1 else f"-{anchor}M.log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )
    # log the size of the dataset in tokens
    print(f"[Run] Size of dataset in tokens: {train_dataset.total_tokens / 1e6:.2f}M")

    use_full = int(anchor) == -1
    output_dir = os.path.abspath(base_output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- GPU selection / visibility (dynamic single/multi GPU) ---
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = int(job.get("gpus_used", 1))
    num_gpus = max(1, min(requested, visible))  # clamp to what we actually have
    print(f"[Run] Requested GPUs: {requested} | Visible GPUs: {visible} | Using: {num_gpus}")
    if num_gpus > 1:
        try:
            devices = [torch.cuda.get_device_name(i) for i in range(visible)]
        except Exception:
            devices = ["<unavailable>"]
        print(f"[Run] Multi-GPU enabled for {job['model_config']} on {num_gpus} GPUs.")
        print(f"[Run] Devices: {devices}")
    else:
        print(f"[Run] Single-GPU/CPU training for {job['model_config']}.")

    # Effective per-device BS/GA (kept by your resolve function)
    # eff_bs, eff_ga = resolve_effective_batch_and_accum(job["model_config"], job["batch_size"], job["grad_accum"])
    eff_bs, eff_ga = resolve_effective_batch_and_accum(
        job["model_config"],
        job["batch_size"],
        job["grad_accum"],
        num_gpus=num_gpus,
    )

    # Per-step tokens processed per *GPU*
    tokens_per_step_per_gpu = eff_bs * job["seq_length"] * eff_ga

    # Global tokens/step across all GPUs (this is what matters for anchor/max_steps)
    tokens_per_step = tokens_per_step_per_gpu * num_gpus
    print(f"[Run] tokens/step per GPU: {tokens_per_step_per_gpu:,} | global tokens/step: {tokens_per_step:,}")
    global_batch_tokens = tokens_per_step  # total tokens per optimizer step across GPUs
    print(f"[Run] Effective per-GPU batch={eff_bs}, grad_accum={eff_ga}, num_gpus={num_gpus}")
    print(f"[Run] → Global batch = {eff_bs * eff_ga * num_gpus} sequences/step "
          f"({global_batch_tokens:,} tokens/step total)")
    
    steps_per_epoch = math.ceil(train_dataset.total_tokens / tokens_per_step)


    # Training arguments
    if use_full:
        training_args = get_training_args(
            dataset_name=job["dataset"],
            output_dir=output_dir,
            model_size=job["model_config"],
            gradient_accum=job["grad_accum"],
            batch_size=job["batch_size"],
            epochs=job["epochs"],
            max_steps=-1,
            num_gpus=num_gpus,  # NEW: inform TrainingArguments for DDP hints/logging
            steps_per_epoch=steps_per_epoch,  # NEW: specify steps/epoch for logging

        )
        scaled_eval_dataset = eval_dataset
    else:
        anchor_tokens = int(anchor) * 1_000_000
        max_steps = math.ceil(anchor_tokens / tokens_per_step)
        training_args = get_training_args(
            dataset_name=job["dataset"],
            output_dir=output_dir,
            model_size=job["model_config"],
            gradient_accum=job["grad_accum"],
            batch_size=job["batch_size"],
            epochs=job["epochs"],
            max_steps=max_steps,
            num_gpus=num_gpus,  
            steps_per_epoch=steps_per_epoch,  # NEW: specify steps/epoch for logging
        )
        eval_seq_budget = int(anchor_tokens * 0.1) // job["seq_length"]
        eval_indices = random.sample(range(len(eval_dataset)), min(eval_seq_budget, len(eval_dataset)))
        scaled_eval_dataset = Subset(eval_dataset, eval_indices)

    model = copy.deepcopy(base_model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loss_callback = LossCallback(logger)

    milestones = [25, 50, 75, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    repo_id = f"{job['hf_namespace']}/{job['dataset']}-{job['model_config']}-tok{job['tokenizer_vocab_size']}"

    milestone_store = job.get("milestone_store", "hf")

    hf_callback = HFTokenMilestoneCallback(
        repo_id=repo_id,
        hf_token=job["hf_token"] if milestone_store == "hf" else None,
        tokenizer=tokenizer,
        output_base=output_dir,
        save_intervals_millions=milestones,
        tokens_per_step=tokens_per_step,
        private=job.get("hf_private", True),
        storage_mode=milestone_store,
        dataset=job["dataset"],
        model_config=job["model_config"],
        seed=job["seed"],
        tokenizer_vocab_size=job["tokenizer_vocab_size"],
    )

    callbacks = [loss_callback, hf_callback]
    if use_full:
        callbacks.append(
            DelayedEarlyStoppingCallback(
                start_tokens=2_000_000_000,
                tokens_per_step=tokens_per_step,
                early_stopping_patience=10,
                early_stopping_threshold=0.002,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=scaled_eval_dataset,
        callbacks=callbacks,
    )

    logger.info(f"***** Running training for anchor: {'full' if use_full else f'{anchor}M'} *****")
    # amoutn of gpus
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    if training_args.max_steps > 0:
        logger.info(f"  Total tokens per step (all GPUs) = {tokens_per_step:,}")
        logger.info(f"  Total tokens for training = {tokens_per_step * training_args.max_steps:,}")
    # log the total tokens for training
    logger.info(f"  Total tokens for training = {train_dataset.total_tokens:,}")
    logger.info(f"  Using {num_gpus} GPU(s)")



    print(f"[DEBUG] per_device_train_batch_size: {trainer.args.per_device_train_batch_size}")
    print(f"[DEBUG] gradient_accumulation_steps: {trainer.args.gradient_accumulation_steps}")
    print(f"[DEBUG] total effective batch size: "
        f"{trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps * num_gpus}")


    # === Train ===
    trainer.train()

    # --- Append summary to trainer_state.json (preserve your previous logic) ---
    try:
        trainer_state_path = os.path.join(output_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}

        # Collect final metrics
        final_metrics = {
            "final_epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
            "final_step": int(trainer.state.global_step),
            "final_train_loss": float(trainer.state.log_history[-1].get("loss", float("nan"))) if trainer.state.log_history else None,
            "final_eval_loss": float(trainer.state.log_history[-1].get("eval_loss", float("nan"))) if trainer.state.log_history else None,
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "total_training_tokens": int(trainer.state.global_step * tokens_per_step),
            "gpus_used": num_gpus,
        }

        state["final_training_summary"] = final_metrics

        with open(trainer_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"✅ Appended final_training_summary to {trainer_state_path}")
    except Exception as e:
        print(f"⚠️ Failed to append final summary to trainer_state.json: {e}")

    # Prune non-milestone checkpoints but keep best
    milestone_steps = {math.ceil((M * 1_000_000) / tokens_per_step) for M in milestones if use_full or M <= int(anchor)}
    best_step = None
    if hasattr(trainer.state, "best_model_checkpoint") and trainer.state.best_model_checkpoint:
        try:
            best_step = int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
        except Exception:
            pass
    keep_steps = set(milestone_steps)
    if best_step:
        keep_steps.add(best_step)
    for p in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        if step not in keep_steps:
            shutil.rmtree(p, ignore_errors=True)

    # Save final model locally and to HF with seed-specific revision
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save final model locally with "final" anchor name
    final_local_dir = os.path.join(output_dir, "final")
    shutil.rmtree(final_local_dir, ignore_errors=True)
    os.makedirs(final_local_dir, exist_ok=True)

    api = HfApi()
    revision = f"seed-{seed}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=job.get("hf_private", True), token=job["hf_token"])
    final_temp = f".temp_hf_final"
    shutil.rmtree(final_temp, ignore_errors=True)
    os.makedirs(final_temp, exist_ok=True)

    root_files = [
        "config.json","generation_config.json","pytorch_model.bin","model.safetensors",
        "training_args.bin","trainer_state.json","rng_state.pth","tokenizer.json","tokenizer.model",
        "tokenizer_config.json","special_tokens_map.json","merges.txt","vocab.json","added_tokens.json"
    ]
    for fn in root_files:
        src = os.path.join(output_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(final_temp, fn))
            # Also copy to local final directory
            shutil.copy2(src, os.path.join(final_local_dir, fn))

    # copy state from best checkpoint if exists
    ckpts = sorted([p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()], key=lambda x: int(x.name.split("-")[-1]))
    chosen_ckpt = Path(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else (ckpts[-1] if ckpts else None)
    if chosen_ckpt:
        for fn in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth", "training_args.bin"]:
            src = chosen_ckpt / fn
            if src.exists():
                shutil.copy2(src, os.path.join(final_temp, fn))
                # Also copy to local final directory
                shutil.copy2(src, os.path.join(final_local_dir, fn))

    print(f"✅ Saved final model locally: {final_local_dir}")

    # Upload to HuggingFace with seed-specific revision
    # First, check if main branch exists and has commits
    try:
        api.list_repo_files(repo_id=repo_id, revision="main", token=job["hf_token"])
        main_exists = True
    except Exception:
        main_exists = False

    # If main doesn't exist or is empty, upload to main first to establish the repo
    if not main_exists:
        print(f"[HF Upload] Initializing main branch for repo: {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=final_temp,
            path_in_repo=".",
            revision="main",
            commit_message="Initialize repository",
            token=job["hf_token"],
        )

    # Now create the seed branch and upload to it
    print(f"[HF Upload] Creating/updating branch {revision}")
    try:
        # Create branch from main if it doesn't exist
        api.create_branch(repo_id=repo_id, branch=revision, token=job["hf_token"], repo_type="model")
    except Exception:
        # Branch might already exist, that's fine
        pass

    # Upload to the seed branch
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=final_temp,
        path_in_repo=".",
        revision=revision,
        create_pr=False,
        commit_message=f"Final model for {revision}" + (" (converged)" if use_full else f" at {anchor}M"),
        token=job["hf_token"],
    )
    print(f"✅ Uploaded final model to HuggingFace: {repo_id} (revision: {revision})")

    shutil.rmtree(final_temp, ignore_errors=True)


def get_evaluator(run_function, num_cpus, num_gpus, num_gpus_per_task: int = 1):
    """
    Ray evaluator wrapper.
    We reserve up to `num_gpus_per_task` GPUs per task so a single training job
    can run single- or multi-GPU on one node, depending on the CSV config.
    """
    method_kwargs = {
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_cpus_per_task": max(1, num_cpus // max(1, num_gpus or 1)),
        "num_gpus_per_task": max(1, num_gpus_per_task),
    }
    print(f"[Evaluator] CPUs total={num_cpus}, GPUs total={num_gpus} → per task: "
          f"CPUs={method_kwargs['num_cpus_per_task']}, GPUs={method_kwargs['num_gpus_per_task']}")
    evaluator = Evaluator.create(run_function, method="ray", method_kwargs=method_kwargs)
    evaluator.return_jobs_done_to_csv = types.MethodType(return_jobs_done_to_csv, evaluator)
    return evaluator

@profile
def run_lm_experiment(job, verbose=True):
    """
    Main experiment runner. If multi-GPU requested, launches via torchrun subprocess.
    Otherwise runs single-GPU training directly.
    """
    num_gpus = int(job.get("gpus_used", 1))
    
    # Single-GPU: run directly (original logic)
    if num_gpus <= 1:
        set_seed(int(job["seed"]))
        base_model, tokenizer, output_dir = load_model_and_tokenizer(job)
        train_dataset, eval_dataset = prepare_datasets(job, tokenizer)
        
        if job["init"]:
            print("Model initialized and saved. Exiting as --init was set.")
            return
        
        base_output_dir = f"{output_dir}/tokenizer-{job['tokenizer_vocab_size']}-seed-{job['seed']}"
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        for anchor in job["anchors"]:
            train_for_anchor(job, base_model, tokenizer, train_dataset, eval_dataset, anchor, job["seed"], base_output_dir)
        return
    
    # Multi-GPU: launch via torchrun
    import subprocess
    import tempfile
    
    print(f"[MultiGPU] Launching training with {num_gpus} GPUs via torchrun...")
    
    # Get the absolute path to ddp_worker.py (same directory as this script)
    script_dir = Path(__file__).parent.absolute()
    ddp_worker_path = script_dir / "ddp_worker.py"
    
    if not ddp_worker_path.exists():
        raise FileNotFoundError(f"ddp_worker.py not found at {ddp_worker_path}")
    
    print(f"[MultiGPU] Using ddp_worker.py at: {ddp_worker_path}")
    
    # Create a clean dict with only JSON-serializable values
    clean_job = {
        k: v for k, v in job.items() 
        if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    
    # Write job config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(clean_job, f, indent=2)
        job_config_path = f.name
    
    print(f"[MultiGPU] Job config written to: {job_config_path}")
    
    try:
        # Launch training for each anchor
        for anchor in clean_job["anchors"]:
            print(f"[MultiGPU] Starting anchor: {anchor}")
            
            # Build torchrun command with absolute path
            cmd = [
                'torchrun',
                '--nproc_per_node', str(num_gpus),
                '--master_port', str(29500 + abs(hash(job_config_path)) % 10000),
                str(ddp_worker_path),  # Use absolute path
                '--job-config', job_config_path,
                '--anchor', str(anchor),
            ]
            
            print(f"[MultiGPU] Running: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'}
            )
            
            if result.returncode != 0:
                print(f"[MultiGPU] ERROR: Training failed for anchor {anchor}")
                raise RuntimeError(f"torchrun failed with return code {result.returncode}")
            
            print(f"[MultiGPU] ✅ Completed anchor: {anchor}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(job_config_path):
            os.unlink(job_config_path)
    
    print("[MultiGPU] All anchors completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--milestone-store", type=str, default="hf", choices=["hf", "local"])
    args = parser.parse_args()

    if args.milestone_store == "hf":
        assert HF_TOKEN, "HF_TOKEN must be set in your environment or .env when using --milestone-store hf"
    else:
        if not HF_TOKEN:
            print("[Info] HF_TOKEN not set; continuing because --milestone-store local was selected.")

    df = pd.read_csv("configs.csv")
    df["anchors"] = df["anchors"].apply(json.loads)

    cpu_count = os.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Determine the maximum GPUs any job may request (from CSV)
    try:
        max_gpus_per_task = max(int(row.get("gpus_used", 1)) for _, row in df.iterrows())
    except Exception:
        max_gpus_per_task = 1
    if max_gpus_per_task < 1:
        max_gpus_per_task = 1

    evaluator = get_evaluator(run_lm_experiment, cpu_count, n_gpus, num_gpus_per_task=max_gpus_per_task)

    job_configs = []
    for _, row in df.iterrows():
        job_configs.append({
            "model_config": row["model_config"],
            "seq_length": int(row["seq_length"]),
            "dataset": row["dataset"],
            "epochs": int(row["epochs"]),
            "batch_size": int(row["batch_size"]),
            "grad_accum": int(row["grad_accum"]),
            "anchors": row["anchors"],
            "tokenizer_vocab_size": int(row["tokenizer_vocab_size"]),
            "gpus_used": int(row.get("gpus_used", 1)),  # NEW: per-job GPU request
            "seed": int(args.seed),
            "init": bool(args.init),
            "hf_token": HF_TOKEN,
            "hf_namespace": HF_NAMESPACE,
            "hf_private": True,
            "milestone_store": args.milestone_store,
        })

    print(f"Total jobs to run: {len(job_configs)}")
    for job in job_configs:
        print("Job:", job)

    evaluator.submit(job_configs)
    evaluator.gather("ALL")
