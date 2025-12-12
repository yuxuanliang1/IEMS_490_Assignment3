import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk

from src.eval.common import get_device, set_seed, save_json, load_policy, load_reward_model
from src.eval.generation import generate_responses
from src.eval.reward_scoring import score_with_reward_model
from src.eval.logprob_kl import kl_on_samples
from src.eval.deepseek_judge import DeepSeekJudge, JudgeConfig


def parse_model_arg(s: str):
    if "=" not in s:
        raise ValueError("--models must be name=path_or_id")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def pareto_frontier(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    pts = sorted(points, key=lambda x: x[1])  # sort by KL
    best_r = -1e30
    frontier = []
    for name, kl, r in pts:
        if r > best_r:
            frontier.append((name, kl, r))
            best_r = r
    return frontier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--reference_model", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--n_prompts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--prompt_max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--rm_batch_size", type=int, default=8)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--do_judge", action="store_true")
    ap.add_argument("--judge_model", type=str, default="deepseek-chat")
    ap.add_argument("--judge_cache", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_dir)
    test = ds["test"]
    prompts_all = [test[i]["prompt"] for i in range(len(test))]
    rng.shuffle(prompts_all)
    prompts = prompts_all[: min(args.n_prompts, len(prompts_all))]

    rm_tok, rm = load_reward_model(args.reward_model_dir, device=device, torch_dtype=None)
    ref_tok, ref_m = load_policy(args.reference_model, device=device, torch_dtype=None)

    judge = None
    if args.do_judge:
        cache_path = Path(args.judge_cache) if args.judge_cache else (out_dir / "judge_cache.json")
        judge = DeepSeekJudge(JudgeConfig(model=args.judge_model), cache_path=cache_path)

    results_rows = []
    for name, model_id in map(parse_model_arg, args.models):
        tok, mdl = load_policy(model_id, device=device, torch_dtype=None)

        gens = generate_responses(
            tok,
            mdl,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            seq_len=args.seq_len,
            prompt_max_length=args.prompt_max_length,
            batch_size=args.gen_batch_size,
        )
        responses = [g["response_text"] for g in gens]

        rewards = score_with_reward_model(
            rm_tok,
            rm,
            prompts=[g["prompt"] for g in gens],
            responses=responses,
            seq_len=args.seq_len,
            batch_size=args.rm_batch_size,
        )

        input_ids = torch.stack([g["input_ids"] for g in gens], dim=0)
        attn = torch.stack([g["attention_mask"] for g in gens], dim=0)
        prompt_lens = torch.tensor([g["prompt_len"] for g in gens], dtype=torch.long)
        kl = kl_on_samples(mdl, ref_m, input_ids, attn, prompt_lens).detach().cpu().numpy()

        win = loss = tie = None
        win_rate = None
        if judge is not None:
            ref_gens = generate_responses(
                ref_tok,
                ref_m,
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                seq_len=args.seq_len,
                prompt_max_length=args.prompt_max_length,
                batch_size=args.gen_batch_size,
            )
            ref_resps = [g["response_text"] for g in ref_gens]

            wins = losses = ties = 0
            for p, a, b in zip(prompts, responses, ref_resps):
                if rng.random() < 0.5:
                    A, B = a, b
                    a_is_model = True
                else:
                    A, B = b, a
                    a_is_model = False
                verdict = judge.judge(p, A, B)
                w = verdict.get("winner", "tie")
                if w == "tie":
                    ties += 1
                elif (w == "A" and a_is_model) or (w == "B" and not a_is_model):
                    wins += 1
                else:
                    losses += 1
            win, loss, tie = wins, losses, ties
            denom = max(1, win + loss)
            win_rate = float(win / denom)

        rewards_np = np.array(rewards, dtype=np.float32)

        row = dict(
            model=name,
            model_id=model_id,
            n=len(prompts),
            mean_reward=float(rewards_np.mean()),
            std_reward=float(rewards_np.std()),
            mean_kl=float(np.mean(kl)),
            std_kl=float(np.std(kl)),
        )
        if judge is not None:
            row.update(dict(wins=win, losses=loss, ties=tie, win_rate=win_rate))
        results_rows.append(row)

        save_json(out_dir / f"details_{name}.json", dict(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            kl=kl.tolist(),
            summary=row,
        ))

        plt.figure()
        plt.hist(rewards_np, bins=30)
        plt.title(f"Reward distribution: {name}")
        plt.xlabel("Reward model score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"reward_hist_{name}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.hist(kl, bins=30)
        plt.title(f"KL drift distribution: {name} vs ref")
        plt.xlabel("Approx KL per sample")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"kl_hist_{name}.png", dpi=200)
        plt.close()

    df = pd.DataFrame(results_rows).sort_values("mean_kl")
    df.to_csv(out_dir / "quant_results.csv", index=False)

    pts = [(r["model"], r["mean_kl"], r["mean_reward"]) for r in results_rows]
    frontier = pareto_frontier(pts)
    save_json(out_dir / "pareto_frontier.json", [{"model": n, "mean_kl": kl, "mean_reward": rw} for n, kl, rw in frontier])

    plt.figure()
    plt.scatter(df["mean_kl"], df["mean_reward"])
    for _, r in df.iterrows():
        plt.annotate(r["model"], (r["mean_kl"], r["mean_reward"]))
    plt.xlabel("Mean KL drift (lower is better)")
    plt.ylabel("Mean reward score (higher is better)")
    plt.title("Reward vs KL (models)")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_kl.png", dpi=220)
    plt.close()

    md = []
    md.append("# Part 4.1 Quantitative Evaluation\n")
    md.append(f"Evaluated on **n={len(prompts)}** held-out prompts from HH-RLHF test split.\n")
    md.append("## Summary Table\n")
    md.append(df.to_markdown(index=False))
    md.append("\n\n## Pareto Frontier (Reward vs KL)\n")
    md.append("| model | mean_kl | mean_reward |\n|---|---:|---:|\n")
    for n, klv, rw in frontier:
        md.append(f"| {n} | {klv:.4f} | {rw:.4f} |\n")
    md.append("\n![Reward vs KL](reward_vs_kl.png)\n")
    (out_dir / "part4_1_quant_eval.md").write_text("\n".join(md), encoding="utf-8")

    save_json(out_dir / "quant_results.json", {"args": vars(args), "results": results_rows, "frontier": frontier})
    print(f"Saved quantitative eval to: {out_dir}")


if __name__ == "__main__":
    main()
