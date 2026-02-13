import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir: str, seq_length: int, tokenizer, offset: int = 0, random_chunk: bool = False, chunk_size: int = 1024, num_workers: int = 32, truncate=False):
        self.seq_length = seq_length
        self.offset = offset
        self.tokenizer = tokenizer
        self.random_chunk = random_chunk
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.truncate = truncate
        self.max_seq_length = seq_length

        tokenizer_name = tokenizer.__class__.__name__
        tokenized_file = Path(os.path.join(data_dir, f"tokenized_{tokenizer_name}_{tokenizer.vocab_size}.bin"))

        if tokenized_file.exists():
            print(f"Loading data from {tokenized_file}")
            self.data = np.memmap(tokenized_file, dtype=np.int32, mode='r')
        else:
            print(f"Tokenized file not found. Processing source files...")
            self.process_and_save_data(data_dir, tokenized_file)
            self.data = np.memmap(tokenized_file, dtype=np.int32, mode='r')

        self.total_tokens = len(self.data)

    def process_and_save_data(self, data_dir: str, tokenized_file: Path):
        src_files = [str(f) for f in Path(data_dir).glob("**/*")
                    if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", ".dev", ".txt", ".jsonl"]]
        print(f"Found source files: {src_files}")

        # Store results from all workers in a list
        results = []

        # Process the files and collect results
        with tqdm(total=len(src_files), desc="Processing files") as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit tasks for processing each file
                future_to_file = {executor.submit(self.process_file, src_file): src_file for src_file in src_files}
                for future in tqdm(future_to_file, desc="Processing file contents", unit="file"):
                    result = future.result()  # Get the result from the worker
                    results.append(result)
                    pbar.update(1)

        # After processing all files, write the collected results to the output file
        with open(tokenized_file, "wb") as out_file:
            for result in results:
                out_file.write(result)
        
        print(f"Saved tokenized data to {tokenized_file}")

    def process_file(self, src_file: str):
        file_size = os.path.getsize(src_file)
        file_result = bytearray()  # To store the result for this file

        with open(src_file, "r", encoding="utf-8") as file:
            with tqdm(total=file_size, desc=f"Processing {os.path.basename(src_file)}", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    text = file.read(self.chunk_size)
                    if not text:
                        break
                    encoded = self.tokenizer.encode(
                        text,
                        truncation=self.truncate,
                        max_length=self.max_seq_length
                    )
                    # Append the encoded tokens to the result
                    file_result.extend(np.array(encoded, dtype=np.int32).tobytes())
                    pbar.update(len(text))

        print(f"Finished processing {src_file}")
        return file_result  # Return the result so it can be combined later


    def __len__(self):
        return self.total_tokens // self.seq_length

    def __getitem__(self, i):
        token_start = i * self.seq_length
        token_end = token_start + self.seq_length
        chunk = self.data[token_start:token_end]
        return torch.tensor(chunk, dtype=torch.long)
