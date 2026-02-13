# Data Efficient Analysis
Code for the paper on data-efficient language model pre-training. We train small LLaMA-style decoder-only models (up to 180M parameters) on curated datasets, study learning curves at token-based anchors, and evaluate linguistic competence and downstream performance.

Released models for this project are available in the Hugging Face collection:
https://huggingface.co/collections/paraskeva/lm-efficiency-analysis

Project website:
https://andreasparaskeva.github.io/lm-efficient-analysis/

## What’s in this repo
- Data preparation for multiple datasets.
- Pre-training with token-based milestone checkpoints.
- BLiMP evaluation (linguistic competence).
- Scripts for local and HPC (Snellius) runs.

## Datasets
We train on:
- TinyStories
- BabyLM (babylm3)
- A composite English corpus (hybrid_3.7B) assembled from multiple sources (e.g., Wikipedia, BookCorpus, OpenWebText, C4, CC-News).

## Environment
Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Optional: create a `.env` (see `.env.example`) for Hugging Face uploads:
- `HF_TOKEN`
- `HF_NAMESPACE`

If you use `--milestone-store local`, `HF_TOKEN` is not required.


## Data preparation
### Download and preprocess data
```bash
./scripts/local/data_preparation.sh
```
Note: this script is one-shot and will fail if `data/babylm3/train_clean` already exists.

## Training
Training is configured via `configs.csv`. Each row defines a dataset, model size, sequence length, anchors, and tokenizer vocab size.

### Editing `configs.csv`
Add or edit rows in `configs.csv` to define experiments. Columns:
- `model_config`: model size key. Supported by default: `20m`, `60m`, `180m`. You can add more by creating new YAML configs in `models/configs/` and updating mappings in `src/models/training/tokenizer_exp.py`.
- `dataset`: dataset key (e.g., `babylm3`, `tinystories`, `hybrid_3.7B`). You can add more datasets as long as they are prepared under `data/<dataset>/` and supported by `src/utils/data_utils.py`.
- `epochs`: set to a high value (we use `50`) and rely on anchors to stop. Full training uses all tokens (`-1` anchor).
- `batch_size`: nominal batch size (currently ignored; defaults are used in code).
- `grad_accum`: nominal gradient accumulation (currently ignored; defaults are used in code).
- `seq_length`: sequence length (e.g., `256`).
- `anchors`: JSON list of token anchors in millions. We typically use:
  `[25,50,75,100,250,500,750,1000,1250,1500,1750,2000]`
  or `[-1]` for full training.
- `tokenizer_vocab_size`: one of `[8000,16000,32000,50257]`.

Example:
```csv
model_config,dataset,epochs,batch_size,grad_accum,seq_length,anchors,tokenizer_vocab_size
60m,tinystories,50,64,1,256,"[25,50,75,100,250,500,750,1000,1250,1500,1750,2000]",8000
60m,hybrid_3.7B,50,64,1,256,"[-1]",50257
```

### Create random initializations (before training)
This creates and saves randomly initialized model weights in `./output/models/random/<model_name>/`.
```bash
./scripts/local/init_random.sh
```

### Local scripts
Run training:
```bash
./scripts/local/pretrain.sh
```

## Evaluation
### BLiMP (linguistic competence)
```bash
python -m src.models.eval.blimp_flexible \
  --source local \
  --dataset_name tinystories \
  --model_size 60m \
  --anchor_size final \
  --seed 0 \
  --tokenizer_vocab_size 8000 \
  --output-base-dir ./output
```

Evaluate a Hugging Face upload:
```bash
python -m src.models.eval.blimp_flexible \
  --source hf \
  --dataset_name tinystories \
  --model_size 60m \
  --anchor_size final \
  --seed 0 \
  --tokenizer_vocab_size 8000 \
  --hf-revision seed-0
```

## Outputs
- Models and checkpoints: `./output/`
- BLiMP results: `./results/blimp/...`

## Reproducibility notes
- Token-based anchors are defined in `src/models/training/tokenizer_exp.py`.
- Effective batch size defaults are in `src/utils/training_utils.py`.

## Quick checks
```bash
python -m src.models.eval.blimp_flexible -h
```

## Citation
If you use this code or the released models, please cite the paper.
