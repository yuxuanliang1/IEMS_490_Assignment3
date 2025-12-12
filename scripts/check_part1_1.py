from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer

path = r"artifacts/part1_1/processed\hhrlhf_train_gpt2len512"
ds = load_from_disk(path)

train = ds["train"]
val = ds["test"]

print("train:", len(train), "val:", len(val))
print("columns:", train.column_names)

# 1) sequence length check
ex = train[0]
L = len(ex["input_ids_chosen"])
print("seq_len =", L)
assert L == 512, "Sequence length is not 512!"

# 2) attention mask sanity
assert set(ex["attention_mask_chosen"]).issubset({0, 1})
assert set(ex["attention_mask_rejected"]).issubset({0, 1})
print("attention masks are binary ✅")

# 3) tie check (should be ~0 if ties were dropped)
tie_rate_train = float(np.mean(train["is_tie"]))
tie_rate_val = float(np.mean(val["is_tie"]))
print("tie_rate(train) =", tie_rate_train)
print("tie_rate(val)   =", tie_rate_val)

# 4) truncation rates
trunc_c = float(np.mean(train["truncated_chosen"]))
trunc_r = float(np.mean(train["truncated_rejected"]))
print("trunc_rate_chosen(train)  =", trunc_c)
print("trunc_rate_rejected(train)=", trunc_r)

# 5) quick decode spot-check
tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tok.pad_token = tok.eos_token

print("\n--- PROMPT (tail) ---")
print(ex["prompt"][-300:])
print("\n--- CHOSEN_RESPONSE (head) ---")
print(ex["chosen_response"][:300])
print("\n--- REJECTED_RESPONSE (head) ---")
print(ex["rejected_response"][:300])

decoded = tok.decode(ex["input_ids_chosen"][:200], skip_special_tokens=False)
print("\n--- DECODED chosen (first 200 tokens) ---")
print(decoded)
