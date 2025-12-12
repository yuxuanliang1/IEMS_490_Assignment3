# scripts/part2_1_train_ppo.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from src.rlhf.ppo import ppo_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_left(input_ids: List[int], max_len: int) -> List[int]:
    if len(input_ids) <= max_len:
        return input_ids
    return input_ids[-max_len:]


@torch.no_grad()
def generate_batch(
    policy,
    tok,
    prompts: List[str],
    prompt_max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      input_ids: [B, T]
      attn_mask: [B, T]
      prompt_lens: [B] (token count of prompt in the returned sequence)
    """
    device = next(policy.parameters()).device
    pad_id = tok.eos_token_id

    prompt_ids_list = []
    prompt_lens = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        ids = truncate_left(ids, prompt_max_length)
        prompt_ids_list.append(torch.tensor(ids, dtype=torch.long))
        prompt_lens.append(len(ids))

    max_plen = max(prompt_lens)
    input_ids = torch.full((len(prompts), max_plen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(prompts), max_plen), dtype=torch.long)

    for i, ids in enumerate(prompt_ids_list):
        input_ids[i, : ids.numel()] = ids
        attn[i, : ids.numel()] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)

    gen = policy.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
        eos_token_id=tok.eos_token_id,
    )

    # Build attention mask by stopping at first EOS after prompt (treat as end-of-response)
    B, T = gen.shape
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    for i in range(B):
        start = int(prompt_lens_t[i].item())
        tail = gen[i, start:]
        eos_positions = (tail == tok.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            end = start + int(eos_positions[0].item()) + 1  # include EOS
            lengths[i] = end

    attn2 = torch.zeros((B, T), dtype=torch.long, device=device)
    pos = torch.arange(T, device=device).unsqueeze(0)
    attn2 = (pos < lengths.unsqueeze(1)).long()

    return gen, attn2, prompt_lens_t


def seq_logprob_and_entropy(
    model,
    input_ids: torch.Tensor,      # [B,T]
    attention_mask: torch.Tensor, # [B,T]
    prompt_lens: torch.Tensor,    # [B]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute:
      logp_sum: [B] sum log prob of response tokens (including first response token)
      ent_mean: [B] mean entropy over response token positions
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B,T,V]

    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # predict tokens 1..T-1
    targets = input_ids[:, 1:]                           # [B,T-1]
    token_logp = torch.gather(logp_all, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    # entropy per position (over vocab) for positions predicting targets (0..T-2)
    p_all = torch.softmax(logits[:, :-1, :], dim=-1)
    ent = -(p_all * logp_all).sum(dim=-1)  # [B,T-1]

    # mask for response tokens in "targets" space:
    # response starts at position prompt_len in input_ids -> corresponds to target index (prompt_len-1)
    start = torch.clamp(prompt_lens - 1, min=0).unsqueeze(1)  # [B,1]
    pos = torch.arange(token_logp.size(1), device=input_ids.device).unsqueeze(0)  # [1,T-1]
    resp_mask = (pos >= start) & (attention_mask[:, 1:] == 1)

    logp_sum = (token_logp * resp_mask).sum(dim=1)
    ent_mean = (ent * resp_mask).sum(dim=1) / (resp_mask.sum(dim=1).clamp(min=1))

    return logp_sum, ent_mean


@torch.no_grad()
def reward_scores(
    rm_model,
    rm_tok,
    prompts: List[str],
    responses: List[str],
    seq_len: int,
) -> torch.Tensor:
    device = next(rm_model.parameters()).device
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = rm_tok(
        texts,
        truncation=True,
        max_length=seq_len,
        padding=True,
        return_tensors="pt",
    ).to(device)
    logits = rm_model(**enc).logits.squeeze(-1)  # [B]
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--init_policy_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/part2_1/ppo_run")
    ap.add_argument("--seed", type=int, default=42)

    # PPO hyperparams (you will sweep these)
    ap.add_argument("--clip_ratio", type=float, default=0.2)
    ap.add_argument("--kl_coef", type=float, default=0.1)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # rollout/training control
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--ppo_epochs", type=int, default=1)

    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--prompt_max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_fp16)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        pass

    # Load prompts
    ds = load_from_disk(args.data_dir)
    prompts_all = [x["prompt"] for x in ds["train"]]

    # Load policy + reference (frozen)
    tok = AutoTokenizer.from_pretrained(args.init_policy_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Load reward model (sequence classifier)
    rm_tok = AutoTokenizer.from_pretrained(args.reward_model_dir)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm = AutoModelForSequenceClassification.from_pretrained(args.reward_model_dir).to(device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step in range(1, args.total_steps + 1):
        policy.train()

        batch_prompts = random.sample(prompts_all, args.batch_size)

        # Rollout: generate responses
        with torch.no_grad():
            gen_ids, gen_attn, prompt_lens = generate_batch(
                policy, tok, batch_prompts,
                prompt_max_length=args.prompt_max_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # decode responses (only the generated tail)
            responses = []
            gen_ids_cpu = gen_ids.detach().cpu()
            prompt_lens_cpu = prompt_lens.detach().cpu()
            for i in range(args.batch_size):
                pl = int(prompt_lens_cpu[i].item())
                # response tokens = after prompt
                resp_ids = gen_ids_cpu[i, pl:]
                responses.append(tok.decode(resp_ids, skip_special_tokens=True))

            # reward model scores
            rewards = reward_scores(rm, rm_tok, batch_prompts, responses, seq_len=args.seq_len)  # [B]

            # logp_old under current policy (behavior)
            logp_old, _ = seq_logprob_and_entropy(policy, gen_ids, gen_attn, prompt_lens)
            logp_ref, _ = seq_logprob_and_entropy(ref, gen_ids, gen_attn, prompt_lens)

            # simple advantage (no value baseline)
            adv = rewards - rewards.mean()

        # PPO update(s)
        for _ in range(args.ppo_epochs):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logp_new, ent = seq_logprob_and_entropy(policy, gen_ids, gen_attn, prompt_lens)
                loss_out = ppo_loss(
                    logp_new=logp_new,
                    logp_old=logp_old.detach(),
                    logp_ref=logp_ref.detach(),
                    advantages=adv.detach(),
                    entropy=ent,
                    clip_ratio=args.clip_ratio,
                    kl_coef=args.kl_coef,
                    ent_coef=args.ent_coef,
                )
                loss = loss_out.total_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        # logging
        if step % args.log_every == 0:
            with torch.no_grad():
                logp_new2, ent2 = seq_logprob_and_entropy(policy, gen_ids, gen_attn, prompt_lens)
                kl_now = torch.mean(logp_new2 - logp_ref)
                mean_reward = float(rewards.mean().item())
                rec = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "policy_loss": float(loss_out.policy_loss.item()),
                    "kl_mean": float(kl_now.item()),
                    "entropy_mean": float(loss_out.entropy_mean.item()),
                    "clip_frac": float(loss_out.clip_frac.item()),
                    "clip_ratio": args.clip_ratio,
                    "kl_coef": args.kl_coef,
                    "ent_coef": args.ent_coef,
                    "lr": args.lr,
                }
            print(rec)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        # save checkpoints
        if step % args.save_every == 0 or step == args.total_steps:
            ckpt_dir = out_dir / "last"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)

    print(f"Saved PPO policy to: {out_dir / 'last'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
