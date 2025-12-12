# scripts/part1_2_train_reward_model.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    model_name: str = "openai-community/gpt2"
    batch_size: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 500
    seed: int = 42
    num_workers: int = 0
    fp16: bool = True
    max_train_steps: int | None = None  # optional cap
    eval_batches: int | None = None     # optional cap for faster dev


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad_global_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return math.sqrt(total)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # Items are already torch tensors (we set_format(type="torch"))
    return {
        "input_ids_chosen": torch.stack([x["input_ids_chosen"] for x in batch], dim=0),
        "attention_mask_chosen": torch.stack([x["attention_mask_chosen"] for x in batch], dim=0),
        "input_ids_rejected": torch.stack([x["input_ids_rejected"] for x in batch], dim=0),
        "attention_mask_rejected": torch.stack([x["attention_mask_rejected"] for x in batch], dim=0),
    }


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, max_batches: int | None = None) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # move to device
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        # ONE forward: concat chosen+rejected
        input_ids = torch.cat([batch["input_ids_chosen"], batch["input_ids_rejected"]], dim=0)
        attn_mask = torch.cat([batch["attention_mask_chosen"], batch["attention_mask_rejected"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            rewards = model(input_ids=input_ids, attention_mask=attn_mask).logits.squeeze(-1)  # [2B]
            r_c, r_r = rewards.chunk(2, dim=0)  # [B], [B]
            loss = -F.logsigmoid(r_c - r_r).mean()

        losses.append(loss.item())
        correct += (r_c > r_r).sum().item()
        total += r_c.size(0)

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_acc": float(correct / max(total, 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"artifacts/part1_1/processed\hhrlhf_train_gpt2len512")
    ap.add_argument("--out_dir", type=str, default=r"artifacts/part1_2\reward_model_gpt2")
    ap.add_argument("--model_name", type=str, default="openai-community/gpt2")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--eval_batches", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seed=args.seed,
        num_workers=args.num_workers,
        fp16=(not args.no_fp16),
        max_train_steps=args.max_train_steps,
        eval_batches=args.eval_batches,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        # TF32 can speed matmul on Ampere+; safe for training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load dataset
    ds = load_from_disk(cfg.data_dir)
    train_ds = ds["train"]
    val_ds = ds["test"]

    cols = [
        "input_ids_chosen", "attention_mask_chosen",
        "input_ids_rejected", "attention_mask_rejected",
    ]
    # Critical speed-up: return torch tensors directly (avoid Python list->tensor each batch)
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1)
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    # Dataloader
    pin = (device.type == "cuda")
    persistent = (cfg.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_amp = (device.type == "cuda" and cfg.fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    best_val_acc = -1.0

    def log_json(obj: Dict[str, Any]) -> None:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            # move to device (non_blocking helps with pin_memory=True)
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            # ONE forward: concat chosen+rejected
            input_ids = torch.cat([batch["input_ids_chosen"], batch["input_ids_rejected"]], dim=0)
            attn_mask = torch.cat([batch["attention_mask_chosen"], batch["attention_mask_rejected"]], dim=0)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                rewards = model(input_ids=input_ids, attention_mask=attn_mask).logits.squeeze(-1)  # [2B]
                r_c, r_r = rewards.chunk(2, dim=0)  # [B], [B]
                loss = -F.logsigmoid(r_c - r_r).mean()

            scaler.scale(loss).backward()

            gnorm = grad_global_norm(model.parameters())
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            acc = (r_c > r_r).float().mean().item()

            global_step += 1

            # progress display
            if global_step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                it_s = global_step / max(elapsed, 1e-6)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}", "gnorm": f"{gnorm:.2f}", "it/s": f"{it_s:.2f}"})
                log_json({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.item()),
                    "train_acc": float(acc),
                    "grad_norm": float(gnorm),
                    "iter_per_sec": float(it_s),
                })

            # eval
            if cfg.eval_every and (global_step % cfg.eval_every == 0):
                val = evaluate(model, val_loader, device, use_amp=use_amp, max_batches=cfg.eval_batches)
                log_json({"step": global_step, **val})
                print(f"\n[eval step {global_step}] val_loss={val['val_loss']:.4f} val_acc={val['val_acc']:.4f}")

                if val["val_acc"] > best_val_acc:
                    best_val_acc = val["val_acc"]
                    save_dir = os.path.join(cfg.out_dir, "best")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tok.save_pretrained(save_dir)

            # optional cap
            if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
                break

        if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
            break

    # final eval + save
    final = evaluate(model, val_loader, device, use_amp=use_amp, max_batches=cfg.eval_batches)
    log_json({"step": global_step, **final, "final": True})
    print("Final:", final)

    last_dir = os.path.join(cfg.out_dir, "last")
    os.makedirs(last_dir, exist_ok=True)
    model.save_pretrained(last_dir)
    tok.save_pretrained(last_dir)

    print("Saved:", cfg.out_dir)


if __name__ == "__main__":
    main()
