# scripts/part1_2_error_analysis.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pick_model_dir(run_dir: str) -> str:
    run = Path(run_dir)
    # prefer best/, else last/
    if (run / "best").exists():
        return str(run / "best")
    if (run / "last").exists():
        return str(run / "last")
    # allow passing a direct model folder
    if run.exists():
        return str(run)
    raise FileNotFoundError(f"Cannot find model dir under {run_dir} (checked best/last).")


def collate_fixed(batch: List[Dict[str, Any]], pad_id: int, seq_len: int) -> Dict[str, torch.Tensor]:
    def fix_stack(key: str, pad_value: int):
        seqs = [x[key] for x in batch]  # 1D torch tensors
        out = []
        for s in seqs:
            s = s[:seq_len]
            if s.size(0) < seq_len:
                s = torch.cat([s, torch.full((seq_len - s.size(0),), pad_value, dtype=s.dtype)], dim=0)
            out.append(s)
        return torch.stack(out, dim=0)

    return {
        "idx": torch.stack([x["idx"] for x in batch], dim=0).long(),
        "input_ids_chosen": fix_stack("input_ids_chosen", pad_id).long(),
        "attention_mask_chosen": fix_stack("attention_mask_chosen", 0).long(),
        "input_ids_rejected": fix_stack("input_ids_rejected", pad_id).long(),
        "attention_mask_rejected": fix_stack("attention_mask_rejected", 0).long(),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"artifacts\part1_1\processed\hhrlhf_train_gpt2len512")
    ap.add_argument("--run_dir", type=str, default=r"artifacts\part1_2\reward_model_gpt2")
    ap.add_argument("--out_dir", type=str, default=r"artifacts\part1_2\analysis_len512")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_errors", type=int, default=50)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.fp16 and device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = pick_model_dir(args.run_dir)

    print("device:", device)
    print("model_dir:", model_dir)
    print("data_dir:", args.data_dir)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    model.config.pad_token_id = pad_id
    model.to(device)
    model.eval()

    ds = load_from_disk(args.data_dir)
    val_ds = ds["test"]

    # Add stable indices to map back to text columns
    if "idx" not in val_ds.column_names:
        val_ds = val_ds.add_column("idx", list(range(len(val_ds))))

    cols = [
        "idx",
        "input_ids_chosen", "attention_mask_chosen",
        "input_ids_rejected", "attention_mask_rejected",
    ]
    val_torch = val_ds.with_format("torch", columns=cols)

    loader = DataLoader(
        val_torch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fixed(b, pad_id=pad_id, seq_len=args.seq_len),
    )

    rows = []
    correct = 0
    total = 0
    losses = []

    for batch in tqdm(loader, desc="eval"):
        idx = batch["idx"].cpu().numpy().tolist()

        for k in batch:
            if k == "idx":
                continue
            batch[k] = batch[k].to(device, non_blocking=True)

        # One forward: concat chosen + rejected
        input_ids = torch.cat([batch["input_ids_chosen"], batch["input_ids_rejected"]], dim=0)
        attn_mask = torch.cat([batch["attention_mask_chosen"], batch["attention_mask_rejected"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            rewards = model(input_ids=input_ids, attention_mask=attn_mask).logits.squeeze(-1)
            r_c, r_r = rewards.chunk(2, dim=0)
            loss = -F.logsigmoid(r_c - r_r).mean()

        r_c = r_c.float().cpu().numpy()
        r_r = r_r.float().cpu().numpy()
        margin = r_c - r_r

        losses.append(float(loss.item()))
        correct += int((margin > 0).sum())
        total += int(margin.shape[0])

        for j, ex_idx in enumerate(idx):
            rows.append({
                "idx": int(ex_idx),
                "r_chosen": float(r_c[j]),
                "r_rejected": float(r_r[j]),
                "margin": float(margin[j]),
            })

    val_acc = correct / max(total, 1)
    val_loss = float(np.mean(losses)) if losses else float("nan")

    errors = [r for r in rows if r["margin"] <= 0.0]
    errors_sorted = sorted(errors, key=lambda x: x["margin"])  # most wrong first
    top_errors = errors_sorted[: args.max_errors]

    enriched = []
    for r in top_errors:
        ex = val_ds[r["idx"]]
        enriched.append({
            **r,
            "prompt": ex.get("prompt"),
            "chosen_response": ex.get("chosen_response"),
            "rejected_response": ex.get("rejected_response"),
            "prompt_len_tokens": ex.get("prompt_len_tokens"),
            "chosen_len_tokens": ex.get("chosen_len_tokens"),
            "rejected_len_tokens": ex.get("rejected_len_tokens"),
            "truncated_chosen": ex.get("truncated_chosen"),
            "truncated_rejected": ex.get("truncated_rejected"),
            "len_bin": ex.get("len_bin"),
            "is_tie": ex.get("is_tie"),
        })

    def trunc_any(ex):
        return bool(ex.get("truncated_chosen")) or bool(ex.get("truncated_rejected"))

    trunc_flags = np.array([trunc_any(val_ds[r["idx"]]) for r in rows], dtype=np.int32)
    err_flags = np.array([r["margin"] <= 0.0 for r in rows], dtype=np.int32)

    summary = {
        "n_val": int(total),
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "n_errors": int(len(errors)),
        "error_rate": float(len(errors) / max(total, 1)),
        "mean_margin": float(np.mean([r["margin"] for r in rows])),
        "mean_margin_errors": float(np.mean([r["margin"] for r in errors])) if errors else None,
        "error_rate_trunc_any": float(err_flags[trunc_flags == 1].mean()) if (trunc_flags == 1).any() else None,
        "error_rate_no_trunc": float(err_flags[trunc_flags == 0].mean()) if (trunc_flags == 0).any() else None,
        "model_dir": model_dir,
        "data_dir": args.data_dir,
        "seq_len_used": int(args.seq_len),
    }

    (out_dir / "val_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved:", str(out_dir))
    print("Summary:", summary)
    print(f"Exported errors: {len(enriched)} (need >= 20 for Task B)")


if __name__ == "__main__":
    main()
