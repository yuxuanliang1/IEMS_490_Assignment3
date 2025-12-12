import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_gpt2_padding(tokenizer):
    # GPT-2 has no pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For decoder-only generation, left padding is recommended.
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy(model_dir_or_name: str, device: torch.device, torch_dtype: Optional[torch.dtype] = None):
    tok = ensure_gpt2_padding(AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True))
    mdl = AutoModelForCausalLM.from_pretrained(model_dir_or_name, torch_dtype=torch_dtype)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


def load_reward_model(model_dir_or_name: str, device: torch.device, torch_dtype: Optional[torch.dtype] = None):
    tok = ensure_gpt2_padding(AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True))
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, torch_dtype=torch_dtype)
    mdl.to(device)
    mdl.eval()
    return tok, mdl


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
