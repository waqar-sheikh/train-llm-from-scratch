from transformers import GPT2Tokenizer
from tqdm import tqdm
import numpy as np

chunk = ""
chunk_count = 0
iter_count = 0
total_tokens = 0
MAX_TOKENS = 10*1024*1024
numpy_tokens = np.lib.format.open_memmap("data/openwebtext/val-dataset.npy", mode="w+", dtype=np.uint16, shape=(MAX_TOKENS,))
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

with open('data/openwebtext/val.txt', "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading file"):
        chunk += line
        chunk_count += 1

        if chunk_count > 50000:
            tokens = tokenizer.encode(chunk, add_special_tokens=False, truncation=False)
            num_tokens = len(tokens)
            
            if total_tokens == 0:
                print("first tokens:", tokens[0:10])

            if total_tokens + num_tokens > MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS - total_tokens]
                num_tokens = len(tokens)
                
            numpy_tokens[total_tokens:total_tokens + num_tokens] = tokens
            total_tokens += len(tokens)
            chunk = ""
            chunk_count = 0
            iter_count += 1
            
            if total_tokens >= MAX_TOKENS:
                break
            
            if iter_count % 10 == 0:
                print()
                print("tokens encoded:", total_tokens)

numpy_tokens.flush()
print(total_tokens)
print("last tokens:", tokens[-10:])
