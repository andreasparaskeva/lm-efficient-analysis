import json
import math
import os
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------
# Environment
# --------------------
load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE")

# --------------------
# Constants
# --------------------
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))

# BLiMP tasks list
BLIMP_TASKS = [
    'adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
    'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island',
    'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction',
    'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
    'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
    'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2',
    'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive',
    'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
    'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
    'left_branch_island_echo_question', 'left_branch_island_simple_question',
    'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2',
    'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2',
    'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2',
    'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3',
    'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present',
    'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1',
    'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2',
    'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance',
    'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'
]

# --------------------
# Arg parser
# --------------------

def get_arg_parser():
    parser = argparse.ArgumentParser(description="BLiMP evaluator for models from Hugging Face or local paths.")

    # What to evaluate (matches your existing script)
    parser.add_argument("-d", "--dataset_name", required=True,
                        choices=[
                            "babylm3", "tinystories", "hybrid_3.7B"
                        ])
    parser.add_argument("-m", "--model_size", required=True,
                        choices=["20m", "60m", "180m", "190m", "400m"],
                        help="Your model size key as used in paths/repo naming.")
    parser.add_argument(
        "-a", "--anchor_size", required=True,
        choices=["25", "50", "75", "100", "250", "500", "750", "1000", "1250", "1500", "1750", "2000", "final"],
        help="Anchor size (use -1 for automatic mode)"
    )

    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-t", "--tokenizer_vocab_size", type=int, choices=[4000, 8000, 16000, 32000, 50257],
                        help="Tokenizer vocab size used in training. Only used for local path bookkeeping.")

    # Source selection
    parser.add_argument("--source", choices=["hf", "local"], default="hf",
                        help="Where to load the model from.")
    parser.add_argument("--output-base-dir", type=str, default="./output",
                        help="Base directory for local models (default: ./output)")

    # Local path override
    parser.add_argument("--local-model-path", type=str, default=None,
                        help="Optional override for local model path.")

    # Hugging Face options
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="Explicit HF repo_id (e.g. namespace/dataset-model). If omitted, built from env and args.")
    parser.add_argument("--hf-namespace", type=str, default=None,
                        help="Override HF namespace (else env HF_NAMESPACE).")
    parser.add_argument("--hf-revision", type=str, default="main",
                        help="HF branch/revision to load: 'main' for final, or tokens-<M>M for milestones.")

    return parser

# --------------------
# Core scoring
# --------------------

@torch.no_grad()
def compute_token_log_likelihoods(sentences, tokenizer, model):
    enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    logits = model(input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask  # zero-out pads
    return token_log_probs.cpu(), shift_labels.cpu(), shift_mask.cpu()


def evaluate_task(task_name, tokenizer, model):
    dataset = load_dataset("blimp", task_name, split="train", cache_dir=HF_CACHE_DIR)

    def _add_bos(s):
        return (tokenizer.bos_token + s) if tokenizer.bos_token else s

    good_sentences = [_add_bos(ex["sentence_good"]) for ex in dataset]
    bad_sentences  = [_add_bos(ex["sentence_bad"])  for ex in dataset]

    correct = 0
    n = len(good_sentences)

    for i in range(0, n, BATCH_SIZE):
        good_batch = good_sentences[i:i+BATCH_SIZE]
        bad_batch  = bad_sentences[i:i+BATCH_SIZE]
        both = good_batch + bad_batch

        tok_lls, _, _ = compute_token_log_likelihoods(both, tokenizer, model)
        bs = len(good_batch)
        good_lls = tok_lls[:bs].sum(dim=1)
        bad_lls  = tok_lls[bs:].sum(dim=1)
        correct += (good_lls > bad_lls).sum().item()

    acc = correct / n
    stderr = math.sqrt(acc * (1 - acc) / n)
    return acc, stderr, n

# --------------------
# Loading utilities
# --------------------

def resolve_anchor_and_ckpt(anchor_size: str):
    """Return (anchor_dir, model_ckpt_name) to keep compatibility with old layout.
    -1 means full training; use 'full' and ckpt 'best' as in the original script.
    """
    if anchor_size == "-1":
        return "full", "best"
    return anchor_size, "final"


def default_local_results_dir(dataset: str, model_size: str, seed: int, tok_size: int | None, anchor_label: str, source_tag: str, revision: str | None = None) -> str:
    """Compose a local directory for saving results. We include the source (hf/local)
    and, for HF, we slot the revision (main or tokens-XXM)."""
    tok_part = f"/tokenizer-{tok_size}" if tok_size else ""
    rev_part = f"/rev-{revision}" if revision else ""
    return f"results/blimp/{dataset}/{model_size}/seed-{seed}/tok-{tok_size}/{anchor_label}"


def build_hf_repo_id(dataset: str, model_size: str, hf_namespace: str | None) -> str:
    """In tokenizer_exp.py, HF repos are named: {namespace}/{dataset}-{model_config}.
    Here we treat your --model_size as that model_config (e.g., '180m').
    """
    ns = hf_namespace or HF_NAMESPACE
    if not ns:
        raise ValueError("HF namespace is not set. Provide --hf-namespace or set HF_NAMESPACE env.")
    return f"{ns}/{dataset}-{model_size}"


def load_model_and_tokenizer_from_source(args) -> tuple[AutoModelForCausalLM, AutoTokenizer, str, str]:
    """Returns (model, tokenizer, results_dir, source_tag).
    results_dir determines where we save the BLiMP JSON locally.
    """
    dataset = args.dataset_name
    model_size = args.model_size
    anchor_label, model_ckpt = resolve_anchor_and_ckpt(args.anchor_size)

    # default results dir is filled later (we include source + possibly revision)

    if args.source == "hf":
        # Figure out repo id and revision
        repo_id = args.hf_repo or build_hf_repo_id(dataset, model_size, args.hf_namespace)
        # Determine revision based on anchor:
        # - final -> seed-{seed} (the final trained model)
        # - milestones -> seed-{seed}-tokens-{M}M
        if args.hf_revision:
            revision = args.hf_revision  # Manual override
        elif anchor_label == "final":
            revision = f"seed-{args.seed}"
        else:
            revision = f"seed-{args.seed}-tokens-{anchor_label}M"

        print(f"[hf] Loading {repo_id}@{revision}")
        model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, token=HF_TOKEN).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save locally under a clear path with revision name
        local_results = default_local_results_dir(dataset, model_size, args.seed, args.tokenizer_vocab_size, anchor_label=anchor_label, source_tag="hf", revision=revision)
        return model, tokenizer, local_results, "hf"

    else:  # local
        # Auto-construct local milestone path or use explicit override
        if args.local_model_path:
            model_path = args.local_model_path
        else:
            # Construct path to local milestone directory
            # Format: {base}/models/{dataset}/llama-{size}/tokenizer-{vocab}-seed-{seed}/milestone-{anchor}M
            # For final: {base}/models/{dataset}/llama-{size}/tokenizer-{vocab}-seed-{seed}/final
            model_name_map = {"20m": "llama-20m", "60m": "llama-60m", "180m": "llama-180m"}
            model_name = model_name_map.get(model_size, f"llama-{model_size}")
            tok_part = args.tokenizer_vocab_size if args.tokenizer_vocab_size else "unknown"
            base_dir = args.output_base_dir
            if anchor_label == "final":
                model_path = f"{base_dir}/models/{dataset}/{model_name}/tokenizer-{tok_part}-seed-{args.seed}/final"
            else:
                model_path = f"{base_dir}/models/{dataset}/{model_name}/tokenizer-{tok_part}-seed-{args.seed}/milestone-{anchor_label}M"

        print(f"[local] Loading from {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Local model path does not exist: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        local_results = default_local_results_dir(dataset, model_size, args.seed, args.tokenizer_vocab_size, anchor_label=anchor_label, source_tag="local")
        return model, tokenizer, local_results, "local"


# --------------------
# Driver
# --------------------

def run_blimp(model, tokenizer):
    # run all blimp tasks
    results = {}
    total_correct = 0
    total_count = 0

    for task in tqdm(BLIMP_TASKS, desc="Evaluating BLiMP"):
        acc, stderr, count = evaluate_task(task, tokenizer, model)
        total_correct += int(acc * count)
        total_count += count
        results[f"blimp_{task}"] = {
            "acc,none": acc,
            "acc_stderr,none": stderr,
            "alias": f" - blimp_{task}"
        }

    overall_acc = total_correct / total_count
    overall_stderr = math.sqrt(overall_acc * (1 - overall_acc) / total_count)

    final_output = {
        "results": {
            "blimp": {
                "acc,none": overall_acc,
                "acc_stderr,none": overall_stderr,
                "alias": "blimp"
            },
            **results
        },
        "groups": {
            "blimp": {
                "acc,none": overall_acc,
                "acc_stderr,none": overall_stderr,
                "alias": "blimp"
            }
        }
    }
    return final_output


def save_results_and_maybe_upload(final_output: dict, args, local_results_dir: str, *_, **__):
    # Always save results locally
    Path(local_results_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(local_results_dir, f"blimp_results.json")
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"✓ Saved BLiMP results locally: {out_file}")


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Map -1 anchor to (full, best) as in legacy script
    anchor_label, model_ckpt = resolve_anchor_and_ckpt(args.anchor_size)

    # Load model/tokenizer based on source
    model, tokenizer, local_results_dir, source_tag = load_model_and_tokenizer_from_source(args)

    # Evaluate
    final_output = run_blimp(model, tokenizer)

    # Persist results
    hf_rev = args.hf_revision if args.source == "hf" else None
    save_results_and_maybe_upload(final_output, args, local_results_dir, source_tag, model_ckpt_label=model_ckpt, hf_revision=hf_rev)


if __name__ == "__main__":
    main()
