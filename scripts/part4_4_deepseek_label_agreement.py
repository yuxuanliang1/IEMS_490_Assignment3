import argparse
import random
from pathlib import Path

from datasets import load_from_disk

from src.eval.common import save_json, set_seed
from src.eval.deepseek_judge import DeepSeekJudge, JudgeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="processed HH-RLHF DatasetDict dir")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge_model", type=str, default="deepseek-chat")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_dir)[args.split]
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    judge = DeepSeekJudge(JudgeConfig(model=args.judge_model), cache_path=out_dir / "judge_cache_pairs.json")

    records = []
    agree = 0
    ties = 0

    for k, i in enumerate(idxs):
        ex = ds[i]
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", ex.get("response_chosen", ""))
        rejected = ex.get("rejected", ex.get("response_rejected", ""))
        # Human label: chosen is preferred
        verdict = judge.judge(prompt, chosen, rejected)  # A=chosen, B=rejected
        w = verdict.get("winner", "tie")
        if w == "A":
            agree += 1
        elif w == "tie":
            ties += 1
        records.append(
            dict(
                idx=int(i),
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                judge_winner=w,
                judge_rationale=verdict.get("rationale", ""),
            )
        )

    n = len(records)
    summary = dict(
        n=n,
        agree_with_human=agree,
        tie=ties,
        disagree=n - agree - ties,
        agree_rate=agree / max(1, n),
        tie_rate=ties / max(1, n),
        disagree_rate=(n - agree - ties) / max(1, n),
        split=args.split,
        judge_model=args.judge_model,
    )

    save_json(out_dir / "label_agreement_summary.json", summary)
    save_json(out_dir / "label_agreement_examples.json", records)

    # markdown
    md = []
    md.append("# Step 4: DeepSeek Label Agreement vs Human Preferences\n")
    md.append(f"Split: **{args.split}**, n={n}, judge_model={args.judge_model}\n")
    md.append("## Summary\n")
    md.append("\n".join([f"- {k}: {v}" for k, v in summary.items()]))
    md.append("\n\n## Example Disagreements (first 20)\n")
    shown = 0
    for r in records:
        if r["judge_winner"] == "B" and shown < 20:
            shown += 1
            md.append(f"### Example {shown} (idx={r['idx']})\n")
            md.append("**Prompt**\n```text\n" + r["prompt"].strip() + "\n```\n")
            md.append("**Human preferred (Chosen)**\n```text\n" + r["chosen"].strip() + "\n```\n")
            md.append("**Rejected**\n```text\n" + r["rejected"].strip() + "\n```\n")
            md.append("**DeepSeek verdict**: prefers Rejected (B)\n")
            md.append("**Rationale**: " + (r["judge_rationale"] or "").strip() + "\n")
    (out_dir / "step4_label_agreement.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
