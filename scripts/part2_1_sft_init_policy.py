# scripts/part2_1_sft_init_policy.py
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HHSFTDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, max_length: int):
        self.ds = hf_ds
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        ex = self.ds[idx]
        prompt = ex["prompt"]
        resp = ex["chosen_response"]

        # Encode prompt alone (truncate) to know where to mask labels
        prompt_ids = self.tok(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # Encode full (prompt + response)
        full_text = prompt + resp
        enc = self.tok(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        # If prompt already fills the whole context, nothing to learn
        if prompt_len >= int(attn.sum().item()):
            return None

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # only train on response tokens

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def collate_drop_none(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # return a dummy batch (won't be used if DataLoader drop_last=True)
        return {"input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long)}
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
    }


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if batch["input_ids"].numel() == 0:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += float(out.loss.item()) * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="openai-community/gpt2")
    ap.add_argument("--out_dir", type=str, default="artifacts/part2_1/sft_gpt2")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_train_samples", type=int, default=2000)
    ap.add_argument("--max_val_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_fp16)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_dir)
    train_ds = ds["train"].shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(ds["train"]))))
    val_ds = ds["test"].shuffle(seed=args.seed).select(range(min(args.max_val_samples, len(ds["test"]))))

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    train_set = HHSFTDataset(train_ds, tok, args.max_length)
    val_set = HHSFTDataset(val_ds, tok, args.max_length)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_drop_none, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_drop_none, drop_last=False
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        pass

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            if batch["input_ids"].numel() == 0:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": global_step, "train_loss": float(loss.item() * args.grad_accum)}) + "\n")

            global_step += 1

        val_loss = eval_loss(model, val_loader, device)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": ep, "val_loss": float(val_loss)}) + "\n")
        print(f"[SFT] epoch {ep}/{args.epochs} val_loss={val_loss:.4f}")

    # Save "last"
    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    tok.save_pretrained(last_dir)
    print(f"Saved SFT policy to: {last_dir}")


if __name__ == "__main__":
    main()
