# scripts/part1_1_prepare_hhrlhf.py
from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from src.data.hhrlhf import PreprocessConfig, preprocess_hhrlhf_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts/part1_1/processed")
    ap.add_argument("--config_name", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--val_size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--keep_ties", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("Anthropic/hh-rlhf", name=args.config_name)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    cfg = PreprocessConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        val_size=args.val_size,
        seed=args.seed,
        max_samples=args.max_samples,
        drop_ties=(not args.keep_ties),
    )

    splits = preprocess_hhrlhf_split(ds[args.split], cfg)

    # Save to disk
    save_path = os.path.join(args.out_dir, f"hhrlhf_{args.split}_gpt2len{args.max_length}")
    splits.save_to_disk(save_path)

    print(f"Saved processed DatasetDict to: {save_path}")
    print(splits)


if __name__ == "__main__":
    main()
