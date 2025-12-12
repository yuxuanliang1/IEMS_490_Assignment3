# scripts/part1_1_explore_hhrlhf.py
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.data.hhrlhf import add_parsed_columns, load_tokenizer


REFUSAL_PHRASES = [
    "i'm sorry", "i cannot", "i can't", "i won't", "unable to", "as an ai", "i do not",
    "i'm not able", "i cannot help", "i can't help"
]


def contains_refusal(text: str) -> int:
    t = (text or "").lower()
    return int(any(p in t for p in REFUSAL_PHRASES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts/part1_1/explore")
    ap.add_argument("--config_name", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("Anthropic/hh-rlhf", name=args.config_name)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    data = ds[args.split].shuffle(seed=args.seed)
    data = data.select(range(min(args.max_samples, len(data))))

    # basic structure
    info = {
        "available_splits": list(ds.keys()),
        "split_used": args.split,
        "n_samples_used": len(data),
        "columns": data.column_names,
    }

    # parse prompt/response
    data = add_parsed_columns(data)

    tok = load_tokenizer(args.tokenizer)

    # length stats (tokens)
    prompt_lens = []
    chosen_lens = []
    rejected_lens = []
    ties = 0

    chosen_longer = 0
    refusal_chosen = 0
    refusal_rejected = 0

    for ex in data:
        p = ex["prompt"]
        cr = ex["chosen_response"]
        rr = ex["rejected_response"]

        if (cr or "").strip() == (rr or "").strip():
            ties += 1

        pl = len(tok(p, add_special_tokens=False).input_ids)
        cl = len(tok(cr, add_special_tokens=False).input_ids)
        rl = len(tok(rr, add_special_tokens=False).input_ids)

        prompt_lens.append(pl)
        chosen_lens.append(cl)
        rejected_lens.append(rl)

        if cl > rl:
            chosen_longer += 1

        refusal_chosen += contains_refusal(cr)
        refusal_rejected += contains_refusal(rr)

    prompt_lens = np.array(prompt_lens)
    chosen_lens = np.array(chosen_lens)
    rejected_lens = np.array(rejected_lens)

    stats: Dict[str, Any] = {
        **info,
        "ties_count": int(ties),
        "chosen_longer_rate": float(chosen_longer / len(data)),
        "refusal_rate_chosen": float(refusal_chosen / len(data)),
        "refusal_rate_rejected": float(refusal_rejected / len(data)),
        "prompt_len_tokens": {
            "mean": float(prompt_lens.mean()),
            "p50": float(np.percentile(prompt_lens, 50)),
            "p90": float(np.percentile(prompt_lens, 90)),
            "p99": float(np.percentile(prompt_lens, 99)),
            "max": int(prompt_lens.max()),
        },
        "chosen_len_tokens": {
            "mean": float(chosen_lens.mean()),
            "p50": float(np.percentile(chosen_lens, 50)),
            "p90": float(np.percentile(chosen_lens, 90)),
            "p99": float(np.percentile(chosen_lens, 99)),
            "max": int(chosen_lens.max()),
        },
        "rejected_len_tokens": {
            "mean": float(rejected_lens.mean()),
            "p50": float(np.percentile(rejected_lens, 50)),
            "p90": float(np.percentile(rejected_lens, 90)),
            "p99": float(np.percentile(rejected_lens, 99)),
            "max": int(rejected_lens.max()),
        },
    }

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # histograms
    plt.figure()
    plt.hist(prompt_lens, bins=50)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Count")
    plt.title("HH-RLHF Prompt Length Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prompt_len_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.hist(chosen_lens - rejected_lens, bins=50)
    plt.xlabel("Chosen - Rejected length (tokens)")
    plt.ylabel("Count")
    plt.title("Length Difference Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "len_diff_hist.png"), dpi=200)
    plt.close()

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
