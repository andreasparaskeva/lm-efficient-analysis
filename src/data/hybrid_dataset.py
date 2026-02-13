import os
import json
import random
import traceback
from glob import glob

from datasets import load_dataset, Dataset, Features, Value
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import snapshot_download

disable_progress_bar()
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
# Robust Hub settings (esp. on servers)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # resumable, parallel
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")       # default 10s is too low
os.environ.setdefault("HF_HUB_MAX_BACKOFF", "60")         # optional
os.environ.setdefault("HF_HUB_ENABLE_TELEMETRY", "0")     # optional

# =========================
# ===== CONFIGURATION =====
# =========================
TEST_MODE = False  # set True for quick smoke tests
OUTPUT_DIR = "./data/hybrid_3.7B"
os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/dev", exist_ok=True)

# Token budget for 180M Chinchilla-optimal training
# Your measurement: 2B words → 2.3B tokens (1.15 tokens/word ratio)
# Chinchilla-optimal for 180M: ~3.6B tokens (20x parameters)
MAX_WORDS_TEST = 100_000
MAX_WORDS_FULL = 3_200_000_000  # ~3.7B tokens for 180M model
MAX_WORDS = MAX_WORDS_TEST if TEST_MODE else MAX_WORDS_FULL

# Dataset fractions - removed WikiText (redundant with Wikipedia)
FRACTIONS = {
    "wikipedia":   0.15,  # Encyclopedic knowledge
    "bookcorpus":  0.20,  # Long-form books
    "openwebtext": 0.30,  # Reddit-curated web
    "c4":          0.25,  # Cleaned Common Crawl
    "cc_news":     0.10,  # News
}

DEV_FRACTION = 0.05
SEED = 1337

# Quality filtering (simple but effective)
MIN_WORDS_PER_EXAMPLE = 10      # Filter tiny fragments
MAX_WORDS_PER_EXAMPLE = 2000    # Filter huge documents
ENABLE_DEDUPLICATION = True     # Remove exact duplicates

# Mirrors / overrides
BOOKCORPUS_REPO = os.environ.get("BOOKCORPUS_REPO", "rojagtap/bookcorpus")
BOOKCORPUS_DIR  = os.environ.get("BOOKCORPUS_DIR")
OPENWEBTEXT_DIR = os.environ.get("OPENWEBTEXT_DIR")
OPENWEBTEXT_REPO = os.environ.get("OPENWEBTEXT_REPO", "Bingsu/openwebtext_20p")
OPENWEBTEXT_MIRROR = os.environ.get("OPENWEBTEXT_MIRROR", "Skylion007/openwebtext")

# ====================
# ===== HELPERS ======
# ====================
def renorm(fracs):
    d = {k: v for k, v in fracs.items() if v > 0}
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}

def count_words(text):
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str):
        text = str(text)
    return len(text.split()), text

def normalize_plaintext(s: str) -> str:
    """Flatten internal newlines and weird whitespace; keep one line per example."""
    if not isinstance(s, str):
        s = str(s)
    # Replace CRLF/CR, collapse internal newlines/tabs, trim
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    s = " ".join(s.split())  # collapses all whitespace incl. newlines into single spaces
    return s.strip()

def materialize_hub_dataset(repo_id: str, allow_patterns=None, force=False) -> str:
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=None,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
        force_download=force,
    )
    return local_path

def bookcorpus_load(split="train"):
    if BOOKCORPUS_DIR:
        if not os.path.exists(BOOKCORPUS_DIR):
            raise FileNotFoundError(f"BOOKCORPUS_DIR does not exist: {BOOKCORPUS_DIR}")
        ds = load_dataset("bookcorpus", split=split, streaming=False, trust_remote_code=True, data_dir=BOOKCORPUS_DIR)
        return ds, "bookcorpus(local)"
    try:
        ds = load_dataset(BOOKCORPUS_REPO, split=split, streaming=False, trust_remote_code=True)
        return ds, BOOKCORPUS_REPO
    except Exception as e1:
        print(f"  -> {BOOKCORPUS_REPO} failed ({repr(e1)}). Falling back to 'bookcorpus' (non-streaming).")
        ds = load_dataset("bookcorpus", split=split, streaming=False, trust_remote_code=True)
        return ds, "bookcorpus"

def openwebtext_load(split="train"):
    if OPENWEBTEXT_DIR:
        if not os.path.isdir(OPENWEBTEXT_DIR):
            raise FileNotFoundError(f"OPENWEBTEXT_DIR does not exist: {OPENWEBTEXT_DIR}")
        ds = load_dataset("openwebtext", split=split, streaming=False, trust_remote_code=True, data_dir=OPENWEBTEXT_DIR)
        return ds, "openwebtext(local)"

    snap = materialize_hub_dataset(OPENWEBTEXT_REPO, allow_patterns=["data/**", "dataset_info.json"])
    parquet_glob = os.path.join(snap, "data", "*.parquet")
    files = sorted(glob(parquet_glob))
    if not files:
        print(f"  -> No parquet files matched at {parquet_glob}. Forcing a fresh snapshot...")
        snap = materialize_hub_dataset(OPENWEBTEXT_REPO, allow_patterns=["data/**", "dataset_info.json"], force=True)
        parquet_glob = os.path.join(snap, "data", "*.parquet")
        files = sorted(glob(parquet_glob))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found at {parquet_glob}. "
                "Install 'huggingface_hub[hf_xet]' if the repo uses Xet storage, "
                "or check your network and cache permissions."
            )

    features = Features({"text": Value("string")})
    try:
        ds = load_dataset("parquet", data_files={"train": files}, split="train", features=features)
        return ds, f"{OPENWEBTEXT_REPO}(local)"
    except Exception as e_parquet:
        print(f"  -> Parquet builder failed: {repr(e_parquet)}. Trying Arrow fallback...")

    try:
        import pyarrow.dataset as pa_ds
        arrow_ds = pa_ds.dataset(files, format="parquet")
        schema_names = set(arrow_ds.schema.names)
        text_col = "text" if "text" in schema_names else ("content" if "content" in schema_names else next(iter(schema_names)))
        table = arrow_ds.scanner(columns=[text_col]).to_table()
        if text_col != "text":
            table = table.rename_columns(["text" if n == text_col else n for n in table.schema.names])
        ds = Dataset.from_arrow(table).cast(features)
        return ds, f"{OPENWEBTEXT_REPO}(local-arrow)"
    except Exception as e_arrow:
        print(f"  -> Arrow fallback failed: {repr(e_arrow)}. Falling back to streaming mirror {OPENWEBTEXT_MIRROR}...")

    ds = load_dataset(OPENWEBTEXT_MIRROR, split=split, streaming=True, trust_remote_code=True)
    return ds, OPENWEBTEXT_MIRROR

def write_dataset(name, frac, out_path, split="train", text_col="text"):
    """Write ONLY plain text lines (no JSON). Each example becomes one line."""
    if frac <= 0:
        return 0
    try:
        print(f"\nLoading {name}...")
        if name == "wikipedia":
            ds = load_dataset("wikipedia", name="20220301.en", split=split, streaming=True, trust_remote_code=True)
        elif name == "c4":
            ds = load_dataset("allenai/c4", name="en", split=split, streaming=True, trust_remote_code=True)
        elif name == "cc_news":
            ds = load_dataset("cc_news", split=split, streaming=True, trust_remote_code=True)
        elif name == "bookcorpus":
            ds, _ = bookcorpus_load(split)
        elif name == "openwebtext":
            ds, _ = openwebtext_load(split)
        else:
            ds = load_dataset(name, split=split, streaming=True, trust_remote_code=True)

        budget = int(MAX_WORDS * frac)
        total_words, wrote, filtered_length, filtered_dedup = 0, 0, 0, 0
        seen_hashes = set() if ENABLE_DEDUPLICATION else None

        with open(out_path, "w", encoding="utf-8") as fout:
            for ex in ds:
                # Extract text
                txt = None
                if isinstance(ex, dict):
                    txt = ex.get(text_col) or ex.get("content") or ex.get("text")
                    if txt is None and len(ex) > 0:
                        txt = next(iter(ex.values()))
                else:
                    txt = ex

                wc, raw = count_words(txt)

                # Quality filtering: length
                if wc < MIN_WORDS_PER_EXAMPLE or wc > MAX_WORDS_PER_EXAMPLE:
                    filtered_length += 1
                    continue

                if total_words + wc > budget:
                    break

                line = normalize_plaintext(raw)
                if not line:
                    filtered_length += 1
                    continue

                # Deduplication: exact line matching
                if ENABLE_DEDUPLICATION:
                    line_hash = hash(line)
                    if line_hash in seen_hashes:
                        filtered_dedup += 1
                        continue
                    seen_hashes.add(line_hash)

                fout.write(line + "\n")
                total_words += wc
                wrote += 1

        print(f"  → {name}: {total_words:,} words, {wrote:,} lines")
        if filtered_length > 0:
            print(f"     Filtered {filtered_length:,} by length")
        if filtered_dedup > 0:
            print(f"     Filtered {filtered_dedup:,} duplicates")

        return total_words

    except Exception as e:
        print(f"Failed {name}: {e}")
        traceback.print_exc()
        return 0

def merge(files, out_file):
    with open(out_file, "w", encoding="utf-8") as fout:
        for f in files:
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"Merged into {out_file}")

def split_dev(train_file, dev_file, frac=0.05, seed=1337):
    """Randomly sample lines to dev; keep the rest in train (line-level split)."""
    rnd = random.Random(seed)
    tmp = train_file + ".tmp"
    total = kept = dev = 0
    with open(train_file, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as ft, \
         open(dev_file, "w", encoding="utf-8") as fd:
        for line in fin:
            if rnd.random() < frac:
                fd.write(line); dev += 1
            else:
                ft.write(line); kept += 1
            total += 1
    # Guarantee at least 1 dev line
    if total > 0 and dev == 0:
        with open(tmp, "r", encoding="utf-8") as ft, open(dev_file, "a", encoding="utf-8") as fd:
            first = next(iter(ft), None)
            if first:
                # remove first occurrence from tmp -> tmp2
                with open(tmp, "r", encoding="utf-8") as ft2, open(train_file + ".tmp2", "w", encoding="utf-8") as ft3:
                    took = False
                    for l in ft2:
                        if not took and l == first:
                            took = True
                            continue
                        ft3.write(l)
                os.replace(train_file + ".tmp2", tmp)
                fd.write(first)
                dev = 1
                kept = total - dev
    os.replace(tmp, train_file)
    print(f"Split: {kept} train, {dev} dev (total {total})")

# ====================
# ===== EXECUTE ======
# ====================
fractions = renorm(FRACTIONS)
results = []
for name, frac in fractions.items():
    out = f"{OUTPUT_DIR}/train/{name}.txt"  # <-- plain text per dataset
    words = write_dataset(name, frac, out)
    if words > 0:
        results.append(out)

merged = f"{OUTPUT_DIR}/train/train.txt"     # <-- merged plain text
merge(results, merged)
for f in results:
    try:
        os.remove(f)
    except OSError:
        pass

dev_file = f"{OUTPUT_DIR}/dev/dev.txt"       # <-- dev plain text
split_dev(merged, dev_file, DEV_FRACTION, SEED)

print("\n✅ Done")
print(f"Train: {merged}")
print(f"Dev:   {dev_file}")
print(f"Mode:  {'TEST' if TEST_MODE else 'FULL'} | Target: {MAX_WORDS:,} words (~{int(MAX_WORDS * 1.15):,} tokens)")
