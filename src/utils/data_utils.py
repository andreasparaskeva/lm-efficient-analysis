import os
from random import sample, seed
from torch.utils.data import Subset

# Local imports
from src.data.dataset import CustomDataset

def get_dataset(dataset_name, seq_lenght, tokenizer):
    truncate=False
    _clean = "_clean"
    num_workers = os.cpu_count()
    if dataset_name == 'tinystories':
        truncate=True
    train_dataset = CustomDataset(f"data/{dataset_name}/train{_clean}", seq_lenght, tokenizer=tokenizer, random_chunk=True, truncate=truncate, num_workers=num_workers)
    full_eval_dataset = CustomDataset(f"data/{dataset_name}/dev{_clean}", seq_lenght, tokenizer=tokenizer, offset=0, truncate=truncate, num_workers=num_workers)
    # seed(0) # we fix the same subset for all models
    full_eval_length = len(full_eval_dataset)
    sample_size = full_eval_length

    eval_indices = sample(range(len(full_eval_dataset)), sample_size)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    return train_dataset, eval_dataset