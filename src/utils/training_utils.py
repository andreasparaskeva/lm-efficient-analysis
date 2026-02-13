import os
import json
import shutil
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback
from huggingface_hub import HfApi


# ---------- Deephyper helper ----------
def return_jobs_done_to_csv(self):
    records_list = []
    for job in getattr(self, "jobs_done", []):
        result = {"job_id": int(job.id.split(".")[1])}
        result.update({f"p:{k}": v for k, v in job.args.items()})
        if isinstance(job.output, dict):
            output = {f"o:{k}": v for k, v in job.output.items()}
        else:
            output = {"o:": job.output}
        result.update(output)
        metadata = {f"m:{k}": v for k, v in job.metadata.items() if k and k[0] != "_"}
        result.update(metadata)
        records_list.append(result)
    return records_list if records_list else None


# ---------- Loss logging ----------
class LossCallback(TrainerCallback):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.epochs: List[float] = []
        self.steps: List[int] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        if "epoch" in logs:
            self.epochs.append(logs["epoch"])
        self.steps.append(state.global_step)
        self.logger.info(logs)


# ---------- Precision ----------
def get_precision_mode() -> str:
    """Return 'bf16' on A100/H100 if supported, else 'fp16' when CUDA, else 'fp32'."""
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "a100" in name or "h100" in name:
                return "bf16"
            return "fp16"
        return "fp32"
    except Exception:
        return "fp32"


# ---------- Model-size defaults ----------
@dataclass
class SizeDefaults:
    lr: float
    warmup_ratio: float
    weight_decay: float
    batch_size: int
    grad_accum: int
    scheduler: str = "cosine"


def _defaults_for_model(model_size: str) -> SizeDefaults:
    ms = model_size.lower()
    if ms == "20m":
        return SizeDefaults(lr=5e-4, warmup_ratio=0.05, weight_decay=0.005, batch_size=64, grad_accum=1, scheduler="cosine")
    if ms == "60m":
        return SizeDefaults(lr=4e-4, warmup_ratio=0.05, weight_decay=0.005, batch_size=128, grad_accum=2, scheduler="cosine")
    if ms == "180m":
        return SizeDefaults(lr=3e-4, warmup_ratio=0.10, weight_decay=0.01, batch_size=256, grad_accum=2, scheduler="cosine")
        # return SizeDefaults(lr=2.5e-4, warmup_ratio=0.10, weight_decay=0.01, batch_size=256, grad_accum=2, scheduler="cosine")

    raise ValueError(f"Unknown model size: {model_size}")


def resolve_effective_batch_and_accum(
    model_size: str,
    batch_size: int,
    grad_accum: int,
    num_gpus: int = 1,
) -> Tuple[int, int]:
    """
    Resolve per-device batch size and gradient accumulation dynamically.
    - Uses defaults as a baseline per model size.
    - Respects user overrides from CSV.
    - Optionally adjusts per-GPU batch size so total global batch stays similar.
    """
    d = _defaults_for_model(model_size)

    # Start with defaults
    eff_bs = d.batch_size
    eff_ga = d.grad_accum

    # NOTE: We intentionally ignore CSV overrides here to keep defaults stable
    # across experiments. If you want overrides, re-enable the block below.
    # if batch_size > 0:
    #     eff_bs = batch_size
    # if grad_accum > 0:
    #     eff_ga = grad_accum

    # If using multi-GPU, adjust per-device batch size so global batch ≈ same
    # This avoids LR drift when scaling GPUs.
    if num_gpus > 1:
        new_eff_bs = max(1, eff_bs // num_gpus)
        print(f"[BatchResolve] Adjusted per-GPU batch from {eff_bs} → {new_eff_bs} for {num_gpus} GPUs "
              f"(global batch ≈ {new_eff_bs * num_gpus * eff_ga})")
        eff_bs = new_eff_bs
    else:
        print(f"[BatchResolve] Using per-device batch={eff_bs}, grad_accum={eff_ga}")

    return eff_bs, eff_ga

def get_training_args(
    dataset_name: str,
    output_dir: str,
    model_size: str,
    gradient_accum: int,
    batch_size: int,
    epochs: int,
    max_steps: Optional[int],
    num_gpus: int = 1,
    steps_per_epoch: Optional[int] = None,
) -> TrainingArguments:
    """
    Builds Hugging Face TrainingArguments with dynamic multi-GPU support.
    - Scales batch size per GPU (via resolve_effective_batch_and_accum)
    - Adjusts LR if multi-GPU used
    - Provides detailed logging for debugging & reproducibility
    """
    if max_steps is None:
        max_steps = -1

    # 1️⃣ Resolve effective per-device batch size and gradient accumulation
    eff_bs, eff_ga = resolve_effective_batch_and_accum(
        model_size, batch_size, gradient_accum, num_gpus=num_gpus
    )
    d = _defaults_for_model(model_size)

    # 2️⃣ Precision auto-detection
    prec = get_precision_mode()
    use_bf16 = (prec == "bf16")
    use_fp16 = (prec == "fp16")
    

    # --- Simplified fixed evaluation/save schedule for fairness across datasets ---
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    WARMUP_MIN_STEPS = 300

    # if max_steps and max_steps > 0:
    #     warmup_steps = max(WARMUP_MIN_STEPS, int(d.warmup_ratio * max_steps))
    # else:
    #     if steps_per_epoch is None:
    #         raise ValueError(
    #             "steps_per_epoch must be computed in main and passed before calling get_training_args()"
    #         )
    #     total_steps = steps_per_epoch * epochs
    #     warmup_steps = max(WARMUP_MIN_STEPS, int(d.warmup_ratio * total_steps))

    eval_steps = EVAL_STEPS
    save_steps = SAVE_STEPS
    warmup_steps = WARMUP_MIN_STEPS

    # 4️⃣ Compute effective global batch (for info & reproducibility)
    global_batch = eff_bs * eff_ga * num_gpus

    # 5️⃣ Base arguments (unchanged)
    base_kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=eff_bs,
        per_device_eval_batch_size=eff_bs,
        gradient_accumulation_steps=eff_ga,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=d.lr,
        warmup_steps=warmup_steps,
        weight_decay=d.weight_decay,
        lr_scheduler_type=d.scheduler,
        save_total_limit=2,
        logging_steps=max(10, eval_steps // 10),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
    )

    # 6️⃣ Logging summary
    print(f"\n[TrainingArgs] Model: {model_size}")
    print(f"  → Dataset: {dataset_name}")
    print(f"  → GPUs: {num_gpus}")
    print(f"  → Per-device batch: {eff_bs}")
    print(f"  → Grad accum: {eff_ga}")
    print(f"  → Global batch: {global_batch} sequences/step")
    print(f"  → LR: {d.lr:.2e}, WD: {d.weight_decay}")
    print(f"  → Warmup: {warmup_steps}, Eval: {eval_steps}, Save: {save_steps}")
    print(f"  → Precision: {'BF16' if use_bf16 else ('FP16' if use_fp16 else 'FP32')}")

    # 7️⃣ Multi-GPU / Distributed toggle
    if num_gpus > 1:
        print(f"[TrainingArgs] ⚙️ Multi-GPU mode enabled ({num_gpus} GPUs)")
        try:
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"[TrainingArgs] Visible devices: {devices}")
        except Exception:
            print("[TrainingArgs] Could not list CUDA devices.")
        
        # Update this dict with proper DDP settings:
        base_kwargs.update(dict(
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",  # ADD THIS
        ))
    else:
        print("[TrainingArgs] Single-GPU or CPU mode")

    return TrainingArguments(**base_kwargs)


# ---------- Milestone saver: HF or local ----------
class HFTokenMilestoneCallback(TrainerCallback):
    """
    At each token milestone, force a checkpoint and export ONLY that checkpoint
    (plus tokenizer + meta). Storage backend:
      - storage_mode='hf'     -> upload to Hugging Face **branch** `tokens-<M>M`
      - storage_mode='local'  -> save locally only
    """

    def __init__(
        self,
        repo_id: str,
        hf_token: Optional[str],
        tokenizer,
        output_base: str,
        save_intervals_millions: List[int],
        tokens_per_step: int,
        private: bool = True,
        repo_type: str = "model",
        wait_for_checkpoint: float = 2.0,
        max_wait_retries: int = 10,
        # New:
        storage_mode: str = "hf",                 # 'hf' or 'local'
        dataset: Optional[str] = None,
        model_config: Optional[str] = None,
        seed: Optional[int] = None,
        tokenizer_vocab_size: Optional[int] = None,
    ):
        import math

        assert storage_mode in ("hf", "local"), "storage_mode must be 'hf' or 'local'"

        self.repo_id = repo_id
        self.hf_token = hf_token
        self.repo_type = repo_type
        self.tokenizer = tokenizer
        self.output_base = output_base
        self.tokens_per_step = tokens_per_step
        self.wait_for_checkpoint = wait_for_checkpoint
        self.max_wait_retries = max_wait_retries

        self.storage_mode = storage_mode
        self.dataset = dataset
        self.model_config = model_config
        self.seed = seed
        self.tokenizer_vocab_size = tokenizer_vocab_size

        # Precompute milestone steps
        self.milestone_steps = {
            M: math.ceil((M * 1_000_000) / tokens_per_step) for M in save_intervals_millions
        }
        self.fired: set[int] = set()
        self._hits_this_step: List[int] = []

        self.api = HfApi() if storage_mode == "hf" else None
        if self.storage_mode == "hf":
            # Ensure repo exists (branches are created per milestone on-demand)
            self.api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private, token=hf_token)

    def _milestone_branch(self, M: int) -> str:
        return f"tokens-{M}M"

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        hits = [M for M, s in self.milestone_steps.items() if s == step and M not in self.fired]
        if hits:
            control.should_save = True
            control.should_log = True
            self._hits_this_step = hits
        else:
            self._hits_this_step = []
        return control
    
    def on_save(self, args, state, control, **kwargs):
        if not self._hits_this_step:
            return control

        # ✅ Only save on rank 0 in DDP mode
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            # Non-zero ranks: just mark as fired and skip saving
            print(f"[Milestone] Rank {dist.get_rank()}: Skipping save (only rank 0 saves)")
            for M in self._hits_this_step:
                self.fired.add(M)
            self._hits_this_step = []
            return control

        # Only rank 0 reaches here
        saved_step = int(state.global_step)
        ckpt_src = os.path.join(self.output_base, f"checkpoint-{saved_step}")
        print(f"[Milestone] Expecting checkpoint at step {saved_step}: {ckpt_src}")

        # Wait for checkpoint dir
        retries = 0
        while not os.path.exists(ckpt_src) and retries < self.max_wait_retries:
            time.sleep(self.wait_for_checkpoint)
            retries += 1

        if not os.path.exists(ckpt_src):
            print(f"[Milestone] ⚠️ ERROR: checkpoint-{saved_step} not found after waiting.")
            self._hits_this_step = []
            return control

        # Save each milestone with a permanent, clearly-named directory
        for M in sorted(self._hits_this_step):
            milestone_dir = os.path.join(self.output_base, f"milestone-{M}M")

            # Copy checkpoint to milestone directory
            print(f"[Milestone] Saving {M}M tokens to {milestone_dir}")
            shutil.rmtree(milestone_dir, ignore_errors=True)
            shutil.copytree(ckpt_src, milestone_dir, dirs_exist_ok=True)
            self.tokenizer.save_pretrained(milestone_dir)

            # Add metadata file
            with open(os.path.join(milestone_dir, f"milestone_meta.json"), "w") as f:
                json.dump(
                    {
                        "tokens_seen": int(M * 1_000_000),
                        "global_step": saved_step,
                        "milestone": f"{M}M",
                        "dataset": self.dataset,
                        "model_config": self.model_config,
                        "seed": self.seed,
                        "tokenizer_vocab_size": self.tokenizer_vocab_size,
                    },
                    f,
                    indent=2,
                )

            self.fired.add(M)
            print(f"[Milestone] ✓ Saved {M}M milestone locally")

            # Upload to HuggingFace if storage_mode is 'hf'
            if self.storage_mode == "hf" and self.api is not None:
                milestone_revision = f"seed-{self.seed}-tokens-{M}M"
                print(f"[Milestone] Uploading {M}M to HuggingFace branch: {milestone_revision}")

                # Check if main branch exists (for first upload)
                try:
                    self.api.list_repo_files(repo_id=self.repo_id, revision="main", token=self.hf_token)
                    main_exists = True
                except Exception:
                    main_exists = False

                # If main doesn't exist, initialize it with this milestone
                if not main_exists:
                    print(f"[Milestone] Initializing main branch for repo: {self.repo_id}")
                    self.api.upload_folder(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        folder_path=milestone_dir,
                        path_in_repo=".",
                        revision="main",
                        commit_message=f"Initialize repository with {M}M milestone",
                        token=self.hf_token,
                    )

                # Create branch and upload milestone
                try:
                    self.api.create_branch(
                        repo_id=self.repo_id,
                        branch=milestone_revision,
                        token=self.hf_token,
                        repo_type=self.repo_type
                    )
                except Exception:
                    # Branch might already exist
                    pass

                # Upload to the milestone branch
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    folder_path=milestone_dir,
                    path_in_repo=".",
                    revision=milestone_revision,
                    create_pr=False,
                    commit_message=f"Milestone at {M}M tokens",
                    token=self.hf_token,
                )
                print(f"[Milestone] ✓ Uploaded {M}M to HuggingFace: {self.repo_id} (branch: {milestone_revision})")

        self._hits_this_step = []
        return control


# ---------- Early stopping only after N tokens ----------
class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, start_tokens: int, tokens_per_step: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_tokens = start_tokens
        self.tokens_per_step = tokens_per_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        tokens_seen = state.global_step * self.tokens_per_step
        if tokens_seen < self.start_tokens:
            return control
        return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
