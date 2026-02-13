from pathlib import Path
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
import argparse

# === Imports unified cleanup functions ===
from src.data.mrclean import (
    cleanup_bnc_spoken,
    cleanup_childes,
    cleanup_gutenberg,
    cleanup_open_subtitles,
    cleanup_simple_wiki,
    cleanup_switchboard,
    cleanup_tiny_stories,
)

# === CONFIG ===
DATASET_VS_SPLITS = {
    'babylm3': {
        'locations': ['babylm3/train_100M', 'babylm3/dev'],
        'vocab_size': 16000,
        'data_dir': Path("./data/babylm3/")  # base data dir, splits cleaned separately
    },
    'tinystories': {
        'locations': ['tinystories/train', 'tinystories/dev'],
        'vocab_size': 8000,
        'data_dir': Path("./data/tinystories/")
    },
    'hybrid_3.7B': {
        'locations': ['hybrid_3.7B/train', 'hybrid_3.7B/dev'],
        'vocab_size': 32000,
        'data_dir': Path("./data/hybrid_3.7B/")
    },

    'hybrid_2B': {
        'locations': ['hybrid_2B/train', 'hybrid_2B/dev'],
        'vocab_size': 32000,
        'data_dir': Path("./data/hybrid_2B/")
    }
}

DATA_ROOT = Path("./")
SEQ_LENGTH = 128
VOCAB_SIZES = [8000, 16000, 32000, 50257]

# === Default Fallback ===
def identity_cleanup(text, seq_length):
    return text

CLEANUP_FUNCTIONS = {
    'bnc_spoken': cleanup_bnc_spoken,
    'childes': cleanup_childes,
    'gutenberg': cleanup_gutenberg,
    'open_subtitles': cleanup_open_subtitles,
    'simple_wiki': cleanup_simple_wiki,
    'switchboard': cleanup_switchboard,
    'TinyStoriesV2-GPT4': cleanup_tiny_stories,
    'wiki_openweb': cleanup_tiny_stories,
    'td': cleanup_tiny_stories,
    'train': identity_cleanup,
    'dev': identity_cleanup,
}

# === Argument Parser ===
def get_parser():
    parser = argparse.ArgumentParser(description="Clean and Tokenize Text Data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_VS_SPLITS.keys(),
        help="The dataset to clean and tokenize",
    )
    return parser

# === File Cleaner ===
def process_file(input_file, output_file, cleanup_function, seq_length):
    buffer_size = 1024 * 1024
    total_size = 0
    total_cleaned_size = 0

    with input_file.open('r') as infile, output_file.open('w') as outfile:
        buffer = ''
        for line in infile:
            buffer += line
            total_size += len(line)
            if len(buffer) > buffer_size:
                cleaned_text = cleanup_function(buffer, seq_length)
                outfile.write(cleaned_text)
                total_cleaned_size += len(cleaned_text)
                buffer = ''
        if buffer:
            cleaned_text = cleanup_function(buffer, seq_length)
            outfile.write(cleaned_text)
            total_cleaned_size += len(cleaned_text)

    print(f"🧹 Cleaned '{input_file.name}' (size {total_size} -> {total_cleaned_size})")

# === File Collector ===
def collect_file_paths(directories):
    paths = []
    for data_dir in directories:
        print(f"Collecting files from: {data_dir}")
        for f in data_dir.glob("*"):
            if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", '.txt', '.jsonl']:
                print(f"  Adding file: {f}")
                paths.append(str(f))
            else:
                print(f"  Skipping file: {f}")
    return paths

# === Main Entry ===
if __name__ == '__main__':
    args = get_parser().parse_args()
    dataset_key = args.dataset
    dataset_config = DATASET_VS_SPLITS[dataset_key]

    base_output_dir = dataset_config['data_dir']
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean each split separately into its own clean folder
    for split in dataset_config['locations']:
        input_dir = DATA_ROOT / 'data' / split
        split_name = Path(split).name  # e.g. 'train' or 'dev'

        output_dir = base_output_dir / f"{split_name}_clean"
        output_dir.mkdir(parents=True, exist_ok=True)

        files_to_process = [f for f in input_dir.iterdir() if f.is_file()]
        print(f"\nProcessing split '{split_name}' from {input_dir}")
        print(f"Found {len(files_to_process)} files:")
        for f in files_to_process:
            print(f"  - {f.name}")

        cleanup_function = CLEANUP_FUNCTIONS.get(split_name, identity_cleanup)

        for file in files_to_process:
            output_file = output_dir / file.name
            process_file(file, output_file, cleanup_function, SEQ_LENGTH)
            print(f"Written cleaned file to: {output_file}")

    # Collect all cleaned data folders for tokenizer training
    cleaned_data_dirs = [base_output_dir / f"{Path(split).name}_clean" for split in dataset_config['locations']]
    print("\nLooking for cleaned training files in these directories:")
    for d in cleaned_data_dirs:
        if d.exists():
            print(f" - {d}: {[p.name for p in d.iterdir()]}")
        else:
            print(f" - {d} (does not exist!)")

    files = collect_file_paths(cleaned_data_dirs)
    print(f"\nCollected {len(files)} files for tokenizer training")

    assert len(files) > 0, "No data files found for tokenizer training"

    for vocab_size in VOCAB_SIZES:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.normalizer = NFKC()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<s>", "</s>"]
        )

        print(f"\nTraining tokenizer with vocab_size={vocab_size} on {len(files)} files...")
        tokenizer.train(files, trainer)

        vocab_tokenizer_dir = base_output_dir / f"tokenizers"
        vocab_tokenizer_dir.mkdir(exist_ok=True)
        tokenizer_path = vocab_tokenizer_dir / f"tokenizer_{vocab_size}.json"
        tokenizer.save(str(tokenizer_path), pretty=True)
        print(f"Tokenizer saved to: {tokenizer_path}")
