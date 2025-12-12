from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_response_logp_mem_efficient(
    logits: torch.Tensor,         # [B,T,V]
    input_ids: torch.Tensor,      # [B,T]
    attention_mask: torch.Tensor, # [B,T]
    prompt_lens: torch.Tensor,    # [B]
) -> torch.Tensor:
    # predict token t+1 from logits at t
    logits = logits[:, :-1, :]          # [B,T-1,V]
    targets = input_ids[:, 1:]          # [B,T-1]

    # log p = logit(target) - logsumexp(logits)
    lse = torch.logsumexp(logits, dim=-1)  # [B,T-1]
    tgt = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
    token_logp = tgt - lse

    start = torch.clamp(prompt_lens - 1, min=0).unsqueeze(1)
    pos = torch.arange(token_logp.size(1), device=token_logp.device).unsqueeze(0)
    resp_mask = (pos >= start) & (attention_mask[:, 1:] == 1)

    return (token_logp * resp_mask).sum(dim=1)  # [B]


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float) -> torch.Tensor:
    z = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return -F.logsigmoid(z).mean()


@torch.no_grad()
def pref_acc(pi_c, pi_r) -> float:
    return (pi_c > pi_r).float().mean().item()


def make_collate(tok, seq_len: int):
    pad_id = tok.eos_token_id

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in batch]
        chosen_rs = [x["chosen_response"] for x in batch]
        rejected_rs = [x["rejected_response"] for x in batch]

        p_ids = tok(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding=False,
        )["input_ids"]
        prompt_lens = torch.tensor([len(p) for p in p_ids], dtype=torch.long).clamp(max=seq_len)

        chosen_text = [p + r for p, r in zip(prompts, chosen_rs)]
        rej_text = [p + r for p, r in zip(prompts, rejected_rs)]

        enc_c = tok(
            chosen_text,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc_r = tok(
            rej_text,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        enc_c["input_ids"][enc_c["attention_mask"] == 0] = pad_id
        enc_r["input_ids"][enc_r["attention_mask"] == 0] = pad_id

        return {
            "input_ids_c": enc_c["input_ids"],
            "attn_c": enc_c["attention_mask"],
            "input_ids_r": enc_r["input_ids"],
            "attn_r": enc_r["attention_mask"],
            "prompt_lens": prompt_lens,
        }

    return collate


@torch.no_grad()
def evaluate(policy, ref, loader, device: str, ref_device: str, beta: float, use_amp: bool) -> Dict[str, float]:
    policy.eval()
    ref.eval()
    losses, accs = [], []

    for batch in loader:
        b_pol = {k: v.to(device) for k, v in batch.items()}
        b_ref = {k: v.to(ref_device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pc = policy(input_ids=b_pol["input_ids_c"], attention_mask=b_pol["attn_c"]).logits
            pr = policy(input_ids=b_pol["input_ids_r"], attention_mask=b_pol["attn_r"]).logits
            pi_c = masked_response_logp_mem_efficient(pc, b_pol["input_ids_c"], b_pol["attn_c"], b_pol["prompt_lens"])
            pi_r = masked_response_logp_mem_efficient(pr, b_pol["input_ids_r"], b_pol["attn_r"], b_pol["prompt_lens"])

        rc = ref(input_ids=b_ref["input_ids_c"], attention_mask=b_ref["attn_c"]).logits
        rr = ref(input_ids=b_ref["input_ids_r"], attention_mask=b_ref["attn_r"]).logits
        ref_c = masked_response_logp_mem_efficient(rc, b_ref["input_ids_c"], b_ref["attn_c"], b_ref["prompt_lens"]).to(device)
        ref_r = masked_response_logp_mem_efficient(rr, b_ref["input_ids_r"], b_ref["attn_r"], b_ref["prompt_lens"]).to(device)

        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
        losses.append(loss.item())
        accs.append(pref_acc(pi_c, pi_r))

    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_acc": float(sum(accs) / max(1, len(accs))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--init_policy_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/part3/dpo_run")

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--max_train_samples", type=int, default=2000)
    ap.add_argument("--max_val_samples", type=int, default=256)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--ref_device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_fp16)
    print("device:", device, "ref_device:", args.ref_device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    ds = load_from_disk(args.data_dir)
    train_ds = ds["train"]
    val_ds = ds["test"] if "test" in ds else ds["validation"]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    tok = AutoTokenizer.from_pretrained(args.init_policy_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # IMPORTANT: load policy in FP16 to reduce optimizer state memory
    if device == "cuda":
        policy = AutoModelForCausalLM.from_pretrained(args.init_policy_dir, torch_dtype=torch.float16).to(device)
    else:
        policy = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(device)
    policy.config.use_cache = False

    # ref on CPU by default
    ref = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(args.ref_device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    collate = make_collate(tok, seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=(device == "cuda")
    )

    # IMPORTANT: foreach=False avoids huge temporary allocations during optimizer.step()
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    accum = max(1, args.grad_accum)
    optim.zero_grad(set_to_none=True)

    for ep in range(1, args.epochs + 1):
        policy.train()
        for batch in train_loader:
            global_step += 1
            b_pol = {k: v.to(device) for k, v in batch.items()}
            b_ref = {k: v.to(args.ref_device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pc = policy(input_ids=b_pol["input_ids_c"], attention_mask=b_pol["attn_c"]).logits
                pr = policy(input_ids=b_pol["input_ids_r"], attention_mask=b_pol["attn_r"]).logits
                pi_c = masked_response_logp_mem_efficient(pc, b_pol["input_ids_c"], b_pol["attn_c"], b_pol["prompt_lens"])
                pi_r = masked_response_logp_mem_efficient(pr, b_pol["input_ids_r"], b_pol["attn_r"], b_pol["prompt_lens"])

            with torch.no_grad():
                rc = ref(input_ids=b_ref["input_ids_c"], attention_mask=b_ref["attn_c"]).logits
                rr = ref(input_ids=b_ref["input_ids_r"], attention_mask=b_ref["attn_r"]).logits
                ref_c = masked_response_logp_mem_efficient(rc, b_ref["input_ids_c"], b_ref["attn_c"], b_ref["prompt_lens"]).to(device)
                ref_r = masked_response_logp_mem_efficient(rr, b_ref["input_ids_r"], b_ref["attn_r"], b_ref["prompt_lens"]).to(device)

            loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=args.beta) / accum
            acc = pref_acc(pi_c.detach(), pi_r.detach())

            scaler.scale(loss).backward()

            if global_step % accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                rec = {"epoch": ep, "step": global_step, "train_loss": float(loss.item() * accum), "train_acc": float(acc)}
                print(rec)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                metrics = evaluate(policy, ref, val_loader, device=device, ref_device=args.ref_device, beta=args.beta, use_amp=use_amp)
                rec = {"epoch": ep, "step": global_step, **metrics}
                print(rec)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

    ckpt = out_dir / "last"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt, safe_serialization=False, max_shard_size="200MB")
    tok.save_pretrained(ckpt)

    final_metrics = evaluate(policy, ref, val_loader, device=device, ref_device=args.ref_device, beta=args.beta, use_amp=use_amp)
    print("Final:", final_metrics)
    print("Saved:", str(ckpt))


if __name__ == "__main__":
    main()
