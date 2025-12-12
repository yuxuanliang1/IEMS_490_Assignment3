# scripts/part2_2_train_grpo.py
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_left(ids: List[int], max_len: int) -> List[int]:
    return ids if len(ids) <= max_len else ids[-max_len:]


@torch.no_grad()
def generate_many_leftpad(
    policy,
    tok,
    prompts: List[str],
    group_size: int,
    prompt_max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[str], List[int]]:
    """
    Left-pad prompts for decoder-only correctness.
    Returns:
      gen_ids: [B, T]
      gen_attn: [B, T] (pads are 0)
      prompt_ctx_len: int (the padded prompt length fed into generate)
      prompts_rep: length B list
      pad_lens: length B list of left-pad lengths (for masking)
    """
    device = next(policy.parameters()).device
    pad_id = tok.eos_token_id

    prompts_rep: List[str] = []
    prompt_ids_list: List[List[int]] = []
    pad_lens: List[int] = []

    # tokenize prompts once per prompt, then repeat G times
    tmp_ids = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        ids = truncate_left(ids, prompt_max_length)
        tmp_ids.append((p, ids))

    max_plen = max(len(ids) for _, ids in tmp_ids)
    prompt_ctx_len = max_plen  # all prompts right-aligned to this length

    for p, ids in tmp_ids:
        for _ in range(group_size):
            prompts_rep.append(p)
            prompt_ids_list.append(ids)
            pad_lens.append(max_plen - len(ids))

    B = len(prompts_rep)
    input_ids = torch.full((B, max_plen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_plen), dtype=torch.long)

    # left-pad: place ids at the end
    for i, ids in enumerate(prompt_ids_list):
        pl = len(ids)
        start = max_plen - pl
        input_ids[i, start:max_plen] = torch.tensor(ids, dtype=torch.long)
        attn[i, start:max_plen] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)

    # generation (cache OK here)
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

    # build attention mask: valid tokens are [pad_len, length)
    B2, T = gen.shape
    lengths = torch.full((B2,), T, dtype=torch.long, device=device)

    for i in range(B2):
        start_resp = prompt_ctx_len  # response begins right after padded prompt ctx
        tail = gen[i, start_resp:]
        eos_pos = (tail == tok.eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            end = start_resp + int(eos_pos[0].item()) + 1
            lengths[i] = end

    pos = torch.arange(T, device=device).unsqueeze(0)
    pad_lens_t = torch.tensor(pad_lens, dtype=torch.long, device=device).unsqueeze(1)
    gen_attn = ((pos >= pad_lens_t) & (pos < lengths.unsqueeze(1))).long()

    return gen, gen_attn, prompt_ctx_len, prompts_rep, pad_lens


def decode_responses(tok, gen_ids: torch.Tensor, prompt_ctx_len: int) -> List[str]:
    gen_ids_cpu = gen_ids.detach().cpu()
    outs = []
    for i in range(gen_ids_cpu.size(0)):
        resp_ids = gen_ids_cpu[i, prompt_ctx_len:]
        outs.append(tok.decode(resp_ids, skip_special_tokens=True))
    return outs


def seq_logprob(
    model,
    input_ids: torch.Tensor,      # [B,T]
    attention_mask: torch.Tensor, # [B,T]
    resp_start_input_pos: int,    # response begins at this input position
) -> torch.Tensor:
    """
    Return sum log prob over response tokens only: [B]
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B,T,V]

    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # predict tokens 1..T-1
    targets = input_ids[:, 1:]                           # [B,T-1]
    token_logp = torch.gather(logp_all, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    # response tokens start at input pos resp_start_input_pos => target index resp_start_input_pos-1
    start = resp_start_input_pos - 1
    pos = torch.arange(token_logp.size(1), device=input_ids.device).unsqueeze(0)
    resp_mask = (pos >= start) & (attention_mask[:, 1:] == 1)

    return (token_logp * resp_mask).sum(dim=1)


def seq_entropy_mean(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resp_start_input_pos: int,
) -> torch.Tensor:
    """
    Mean entropy over response token positions: [B]
    (Expensive: full softmax over vocab. Only compute if ent_coef>0.)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B,T,V]

    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)
    p_all = torch.softmax(logits[:, :-1, :], dim=-1)
    ent = -(p_all * logp_all).sum(dim=-1)  # [B,T-1]

    start = resp_start_input_pos - 1
    pos = torch.arange(ent.size(1), device=input_ids.device).unsqueeze(0)
    resp_mask = (pos >= start) & (attention_mask[:, 1:] == 1)

    ent_sum = (ent * resp_mask).sum(dim=1)
    denom = resp_mask.sum(dim=1).clamp(min=1)
    return ent_sum / denom


@torch.no_grad()
def reward_scores(
    rm_model,
    rm_tok,
    prompts: List[str],
    responses: List[str],
    seq_len: int,
    device: str,
) -> torch.Tensor:
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = rm_tok(
        texts,
        truncation=True,
        max_length=seq_len,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return rm_model(**enc).logits.squeeze(-1)  # [B]


def save_checkpoint(
    policy,
    tok,
    ckpt_dir: Path,
    *,
    safe_serialization: bool,
    max_shard_size: str,
    save_on_cpu: bool,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orig_device = next(policy.parameters()).device
    moved = False

    policy.eval()
    try:
        if save_on_cpu and orig_device.type == "cuda":
            policy.to("cpu")
            moved = True
            torch.cuda.empty_cache()
            gc.collect()

        policy.save_pretrained(
            ckpt_dir,
            safe_serialization=safe_serialization,  # default False (saves .bin)
            max_shard_size=max_shard_size,
        )
        tok.save_pretrained(ckpt_dir)
    finally:
        if moved:
            policy.to(orig_device)
        policy.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--init_policy_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/part2_2/grpo_run")
    ap.add_argument("--seed", type=int, default=42)

    # GRPO spec knobs
    ap.add_argument("--group_size", type=int, default=4)      # 4-8
    ap.add_argument("--prompt_batch", type=int, default=1)    # prompts per step
    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # memory/stability
    ap.add_argument("--microbatch", type=int, default=2, help="split B=P*G into smaller chunks for backward")
    ap.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing on policy")
    ap.add_argument("--disable_cache_train", action="store_true", help="set policy.use_cache=False during training forward")

    # optional regularizers (set to 0 to save VRAM)
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--ent_coef", type=float, default=0.0)

    # rollout params
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--prompt_max_length", type=int, default=320)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # device placement
    ap.add_argument("--rm_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--ref_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # logging/saving
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0, help="0 means only save at end")
    ap.add_argument("--no_fp16", action="store_true")

    # checkpoint options (fix MemoryError)
    ap.add_argument("--save_safetensors", action="store_true")
    ap.add_argument("--max_shard_size", type=str, default="200MB")
    ap.add_argument("--save_on_cpu", action="store_true")

    # CLI compatibility (unused)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()
    assert 4 <= args.group_size <= 8

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_fp16)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    # dataset prompts
    ds = load_from_disk(args.data_dir)
    prompts_all = [x["prompt"] for x in ds["train"]]

    # tokenizer (LEFT padding!)
    tok = AutoTokenizer.from_pretrained(args.init_policy_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # policy on GPU
    policy = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(device)
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable()

    # reward model (optionally CPU to save VRAM)
    rm_tok = AutoTokenizer.from_pretrained(args.reward_model_dir)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm_tok.padding_side = "left"

    rm = AutoModelForSequenceClassification.from_pretrained(args.reward_model_dir).to(args.rm_device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    # reference model only if kl_coef>0
    ref: Optional[AutoModelForCausalLM] = None
    if args.kl_coef > 0.0:
        ref = AutoModelForCausalLM.from_pretrained(args.init_policy_dir).to(args.ref_device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    P = args.prompt_batch
    G = args.group_size

    for step in range(1, args.total_steps + 1):
        t0 = time.time()
        policy.train()

        batch_prompts = random.sample(prompts_all, P)

        # rollout
        with torch.no_grad():
            # generation should use cache
            policy.config.use_cache = True
            gen_ids, gen_attn, prompt_ctx_len, prompts_rep, pad_lens = generate_many_leftpad(
                policy, tok, batch_prompts,
                group_size=G,
                prompt_max_length=args.prompt_max_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            responses = decode_responses(tok, gen_ids, prompt_ctx_len)

            rewards = reward_scores(
                rm, rm_tok, prompts_rep, responses,
                seq_len=args.seq_len,
                device=args.rm_device,
            )

            # group-relative advantage
            rewards_pg = rewards.view(P, G)
            group_mean = rewards_pg.mean(dim=1, keepdim=True)
            adv = (rewards_pg - group_mean).view(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # reference logp for KL (optional)
            logp_ref = None
            if args.kl_coef > 0.0 and ref is not None:
                # move ids/mask to ref device if needed
                ids_ref = gen_ids.to(args.ref_device) if args.ref_device != device else gen_ids
                attn_ref = gen_attn.to(args.ref_device) if args.ref_device != device else gen_attn
                logp_ref = seq_logprob(ref, ids_ref, attn_ref, resp_start_input_pos=prompt_ctx_len).to(device)

        # training forward can disable cache to save VRAM
        if args.disable_cache_train:
            policy.config.use_cache = False

        # microbatch backward to fit 6GB
        B = gen_ids.size(0)
        mb = max(1, args.microbatch)
        total_loss = 0.0
        total_kl = 0.0
        total_ent = 0.0

        optim.zero_grad(set_to_none=True)

        for s in range(0, B, mb):
            e = min(B, s + mb)
            ids = gen_ids[s:e]
            attn = gen_attn[s:e]
            adv_mb = adv[s:e]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logp_new = seq_logprob(policy, ids, attn, resp_start_input_pos=prompt_ctx_len)

                pg_loss = -(adv_mb.detach() * logp_new).mean()

                kl = torch.tensor(0.0, device=device)
                if args.kl_coef > 0.0 and logp_ref is not None:
                    kl = (logp_new - logp_ref[s:e].detach()).mean()

                ent = torch.tensor(0.0, device=device)
                if args.ent_coef > 0.0:
                    ent = seq_entropy_mean(policy, ids, attn, resp_start_input_pos=prompt_ctx_len).mean()

                loss = pg_loss + args.kl_coef * kl - args.ent_coef * ent
                loss = loss * ((e - s) / B)  # weight microbatches

            scaler.scale(loss).backward()

            total_loss += float(loss.detach().item())
            total_kl += float(kl.detach().item()) * ((e - s) / B)
            total_ent += float(ent.detach().item()) * ((e - s) / B)

        scaler.unscale_(optim)
        gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).item())
        scaler.step(optim)
        scaler.update()

        dt = time.time() - t0

        if step % args.log_every == 0:
            rec = {
                "step": step,
                "mean_reward": float(rewards.mean().item()),
                "mean_group_std": float(rewards.view(P, G).std(dim=1, unbiased=False).mean().item()),
                "loss": total_loss,
                "kl_mean": float(total_kl),
                "entropy_mean": float(total_ent),
                "grad_norm": gnorm,
                "sec_per_step": dt,
                "group_size": G,
                "prompt_batch": P,
                "microbatch": mb,
                "lr": args.lr,
                "kl_coef": args.kl_coef,
                "ent_coef": args.ent_coef,
                "rm_device": args.rm_device,
                "ref_device": args.ref_device if args.kl_coef > 0 else "none",
            }
            print(rec)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if args.save_every > 0 and (step % args.save_every == 0):
            save_checkpoint(
                policy, tok, out_dir / "last",
                safe_serialization=args.save_safetensors,
                max_shard_size=args.max_shard_size,
                save_on_cpu=args.save_on_cpu,
            )

    # always save at end
    save_checkpoint(
        policy, tok, out_dir / "last",
        safe_serialization=args.save_safetensors,
        max_shard_size=args.max_shard_size,
        save_on_cpu=args.save_on_cpu,
    )

    print(f"Saved GRPO policy to: {out_dir / 'last'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
