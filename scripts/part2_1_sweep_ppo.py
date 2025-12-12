# scripts/part2_1_sweep_ppo.py
from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out", type=str, default="artifacts/part2_1/ppo_sweeps")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--init_policy_dir", type=str, required=True)

    ap.add_argument("--clip_ratios", nargs="+", type=float, default=[0.1, 0.2])
    ap.add_argument("--kl_coefs", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--lrs", nargs="+", type=float, default=[5e-7, 1e-6])

    ap.add_argument("--total_steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_out = Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    for clip_ratio, kl_coef, lr in itertools.product(args.clip_ratios, args.kl_coefs, args.lrs):
        run_name = f"clip{clip_ratio}_kl{kl_coef}_lr{lr}"
        out_dir = base_out / run_name

        cmd = [
            "python", "-m", "scripts.part2_1_train_ppo",
            "--data_dir", args.data_dir,
            "--reward_model_dir", args.reward_model_dir,
            "--init_policy_dir", args.init_policy_dir,
            "--out_dir", str(out_dir),
            "--clip_ratio", str(clip_ratio),
            "--kl_coef", str(kl_coef),
            "--lr", str(lr),
            "--total_steps", str(args.total_steps),
            "--batch_size", str(args.batch_size),
            "--max_new_tokens", str(args.max_new_tokens),
            "--seed", str(args.seed),
        ]
        print("\nRUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
