# src/data/hhrlhf.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase


ASSISTANT_MARKER = "\n\nAssistant:"


@dataclass
class PreprocessConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    val_size: float = 0.05
    seed: int = 42
    max_samples: Optional[int] = None  # for quick prototyping
    drop_ties: bool = True
    padding: str = "max_length"


def _safe_extract_prompt_and_responses(chosen: str, rejected: str) -> Tuple[str, str, str]:
    """
    Robustly extract:
      prompt = common prefix up to the last Assistant marker (inclusive)
      chosen_resp = chosen[len(prompt):]
      rejected_resp = rejected[len(prompt):]

    If common-prefix parsing fails, fall back to using the last Assistant marker in each string.
    """
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return "", "", ""

    prefix = os.path.commonprefix([chosen, rejected])

    # Prefer prompt ending at last Assistant marker inside the shared prefix
    idx = prefix.rfind(ASSISTANT_MARKER)
    if idx != -1:
        prompt = prefix[: idx + len(ASSISTANT_MARKER)]
    else:
        # Fallback: try to find marker in chosen
        idx2 = chosen.rfind(ASSISTANT_MARKER)
        if idx2 != -1:
            prompt = chosen[: idx2 + len(ASSISTANT_MARKER)]
        else:
            # Worst-case fallback: treat entire prefix as prompt
            prompt = prefix

    if not chosen.startswith(prompt) or not rejected.startswith(prompt):
        # Another fallback: use chosen's own last marker
        idx3 = chosen.rfind(ASSISTANT_MARKER)
        if idx3 != -1:
            prompt = chosen[: idx3 + len(ASSISTANT_MARKER)]

    # Slice responses (may be empty if parsing failed)
    chosen_resp = chosen[len(prompt):] if len(prompt) <= len(chosen) else ""
    rejected_resp = rejected[len(prompt):] if len(prompt) <= len(rejected) else ""

    return prompt, chosen_resp, rejected_resp


def _build_padded_input(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    max_length: int,
) -> Tuple[List[int], List[int], bool]:
    """
    Create a single packed sequence for a causal LM reward model:
      input_ids = [prompt_tokens] + [response_tokens] + [eos]
    Truncation policy:
      - Keep response as much as possible
      - Truncate prompt from the LEFT if needed
      - If response itself is too long, truncate response from the RIGHT

    Returns: input_ids, attention_mask, was_truncated
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have eos_token_id (GPT-2 does).")

    # ensure response ends with EOS
    resp_ids = resp_ids + [eos_id]

    was_truncated = False
    total_len = len(prompt_ids) + len(resp_ids)

    if total_len > max_length:
        was_truncated = True
        # Keep response; truncate prompt from left
        max_prompt_len = max_length - len(resp_ids)
        if max_prompt_len < 0:
            # response alone exceeds max_length -> truncate response
            resp_ids = resp_ids[: max_length - 1] + [eos_id]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + resp_ids
    attn = [1] * len(input_ids)

    # pad to max_length
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id (we set it to eos for GPT-2).")

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attn += [0] * pad_len

    return input_ids, attn, was_truncated


def load_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # GPT-2 has no official pad token -> set pad to EOS for batching
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def add_parsed_columns(ds: Dataset) -> Dataset:
    def _map_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        prompts, ch_resps, rj_resps, ties = [], [], [], []
        for c, r in zip(batch["chosen"], batch["rejected"]):
            p, cr, rr = _safe_extract_prompt_and_responses(c, r)
            prompts.append(p)
            ch_resps.append(cr)
            rj_resps.append(rr)
            ties.append(int(cr.strip() == rr.strip()))
        return {
            "prompt": prompts,
            "chosen_response": ch_resps,
            "rejected_response": rj_resps,
            "is_tie": ties,
        }

    cols = ds.column_names
    if "chosen" not in cols or "rejected" not in cols:
        raise ValueError(f"Expected columns 'chosen' and 'rejected', got: {cols}")

    return ds.map(_map_fn, batched=True, desc="Extract prompt/response")


def tokenize_pairs(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> Dataset:
    def _tok_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        ic, ac, ir, ar = [], [], [], []
        trunc_c, trunc_r = [], []
        prompt_lens = []
        resp_lens_c, resp_lens_r = [], []

        for p, cr, rr in zip(batch["prompt"], batch["chosen_response"], batch["rejected_response"]):
            # lengths (untruncated) for analysis/stratification
            p_len = len(tokenizer(p, add_special_tokens=False).input_ids)
            cr_len = len(tokenizer(cr, add_special_tokens=False).input_ids)
            rr_len = len(tokenizer(rr, add_special_tokens=False).input_ids)

            prompt_lens.append(p_len)
            resp_lens_c.append(cr_len)
            resp_lens_r.append(rr_len)

            ids_c, att_c, t_c = _build_padded_input(tokenizer, p, cr, max_length)
            ids_r, att_r, t_r = _build_padded_input(tokenizer, p, rr, max_length)

            ic.append(ids_c); ac.append(att_c)
            ir.append(ids_r); ar.append(att_r)
            trunc_c.append(int(t_c)); trunc_r.append(int(t_r))

        return {
            "input_ids_chosen": ic,
            "attention_mask_chosen": ac,
            "input_ids_rejected": ir,
            "attention_mask_rejected": ar,
            "prompt_len_tokens": prompt_lens,
            "chosen_len_tokens": resp_lens_c,
            "rejected_len_tokens": resp_lens_r,
            "truncated_chosen": trunc_c,
            "truncated_rejected": trunc_r,
        }

    return ds.map(_tok_fn, batched=True, desc="Tokenize preference pairs")


def add_length_bins(ds: Dataset, n_bins: int = 8) -> Dataset:
    """
    Create bins based on prompt token length for stratified split.
    """
    lens = np.array(ds["prompt_len_tokens"], dtype=np.int32)
    # quantile bins are usually more balanced than fixed bins
    qs = np.quantile(lens, q=np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs).astype(np.int32)

    def _bin_one(x: int) -> str:
        # digitize into bins [qs[i], qs[i+1])
        idx = int(np.digitize([x], qs[1:-1], right=False)[0])
        return str(idx)

    return ds.map(lambda ex: {"len_bin": _bin_one(int(ex["prompt_len_tokens"]))}, desc="Add len bins")


def make_train_val_split(ds: Dataset, val_size: float, seed: int) -> DatasetDict:
    # stratify by prompt-length bins
    if "len_bin" not in ds.column_names:
        ds = add_length_bins(ds)

    # IMPORTANT: stratify requires ClassLabel (datasets limitation)
    try:
        from datasets import ClassLabel
        if not isinstance(ds.features["len_bin"], ClassLabel):
            ds = ds.class_encode_column("len_bin")
    except Exception:
        # fallback (older datasets): manually encode + cast to ClassLabel
        from datasets import ClassLabel
        uniq = sorted(set(ds["len_bin"]))
        mapping = {v: i for i, v in enumerate(uniq)}
        ds = ds.map(lambda ex: {"len_bin": mapping[ex["len_bin"]]})
        ds = ds.cast_column("len_bin", ClassLabel(names=[str(u) for u in uniq]))

    return ds.train_test_split(
        test_size=val_size,
        seed=seed,
        stratify_by_column="len_bin",
    )



def preprocess_hhrlhf_split(
    ds: Dataset,
    cfg: PreprocessConfig,
) -> DatasetDict:
    if cfg.max_samples is not None:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.max_samples, len(ds))))

    ds = add_parsed_columns(ds)

    # Handle ties (edge case)
    if cfg.drop_ties:
        ds = ds.filter(lambda ex: int(ex["is_tie"]) == 0, desc="Drop ties")

    tok = load_tokenizer(cfg.model_name)
    ds = tokenize_pairs(ds, tok, max_length=cfg.max_length)

    # Create balanced train/val splits (stratified on prompt length bins)
    splits = make_train_val_split(ds, val_size=cfg.val_size, seed=cfg.seed)
    return splits
