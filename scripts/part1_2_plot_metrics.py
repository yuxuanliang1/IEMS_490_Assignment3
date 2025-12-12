# scripts/part1_2_plot_metrics.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_xy(xs, ys, title, xlabel, ylabel, out_path: Path):
    if len(xs) == 0 or len(ys) == 0:
        return
    if len(xs) != len(ys):
        # Skip silently if malformed logs
        return
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=r"artifacts\part1_2\reward_model_gpt2")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(metrics_path)

    train_steps, train_loss, train_acc, grad_norm = [], [], [], []
    val_loss_steps, val_loss_vals = [], []
    val_acc_steps, val_acc_vals = [], []

    for r in rows:
        step = r.get("step")
        if step is None:
            continue

        if "train_loss" in r:
            train_steps.append(step)
            train_loss.append(r["train_loss"])
        if "train_acc" in r:
            # train_acc logged at same steps as train_loss (in our script)
            train_acc.append(r["train_acc"])
        if "grad_norm" in r:
            grad_norm.append(r["grad_norm"])

        if "val_loss" in r:
            val_loss_steps.append(step)
            val_loss_vals.append(r["val_loss"])
        if "val_acc" in r:
            val_acc_steps.append(step)
            val_acc_vals.append(r["val_acc"])

    # If train_acc/grad_norm length mismatches due to logging format, align by truncation
    min_train_len = min(len(train_steps), len(train_loss), len(train_acc) if train_acc else len(train_steps), len(grad_norm) if grad_norm else len(train_steps))
    train_steps = train_steps[:min_train_len]
    train_loss = train_loss[:min_train_len]
    if train_acc:
        train_acc = train_acc[:min_train_len]
    if grad_norm:
        grad_norm = grad_norm[:min_train_len]

    plot_xy(train_steps, train_loss, "Train Loss", "Step", "Loss", out_dir / "train_loss.png")
    plot_xy(train_steps, train_acc, "Train Accuracy", "Step", "Accuracy", out_dir / "train_acc.png")
    plot_xy(train_steps, grad_norm, "Gradient Norm", "Step", "Grad Norm (L2)", out_dir / "grad_norm.png")
    plot_xy(val_loss_steps, val_loss_vals, "Validation Loss", "Step", "Loss", out_dir / "val_loss.png")
    plot_xy(val_acc_steps, val_acc_vals, "Validation Accuracy", "Step", "Accuracy", out_dir / "val_acc.png")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
