import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

from src.eval.common import save_json, set_seed
from src.eval.deepseek_judge import DeepSeekJudge, JudgeConfig


TAG_SYSTEM = '''You are analyzing failure cases of a learned reward model on preference data.
Given a user prompt, a human-preferred response (chosen), and a less-preferred response (rejected),
explain why a model might mistakenly rank them, and assign a short tag.

Return JSON only with keys:
- tag: one of [verbosity_bias, instruction_following, factuality, safety_refusal, style_tone, missing_context, ambiguity, other]
- explanation: 2-4 sentences
- which_is_better: A (chosen) or B (rejected) or tie (your own judgment).'''

TAG_USER_TEMPLATE = '''Prompt:
{prompt}

A (human preferred / chosen):
{chosen}

B (rejected):
{rejected}

Model scores (optional, may be empty): {scores}

Return JSON only.'''


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors_path", type=str, required=True, help="path to errors.jsonl exported by part1_2_error_analysis.py")
    ap.add_argument("--n", type=int, default=30, help="tag this many errors (>=20 often required)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge_model", type=str, default="deepseek-chat")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.errors_path))
    rng.shuffle(rows)
    rows = rows[: min(args.n, len(rows))]

    # We reuse DeepSeekJudge but with a different system/user prompt.
    judge = DeepSeekJudge(JudgeConfig(model=args.judge_model), cache_path=out_dir / "judge_cache_error_tags.json")

    tagged = []
    for r in rows:
        prompt = r.get("prompt", "")
        chosen = r.get("chosen", r.get("response_chosen", ""))
        rejected = r.get("rejected", r.get("response_rejected", ""))
        scores = {k: r.get(k) for k in ["reward_chosen", "reward_rejected", "margin"] if k in r}

        # call underlying API directly via judge.client with our prompts
        msg = [
            {"role": "system", "content": TAG_SYSTEM},
            {"role": "user", "content": TAG_USER_TEMPLATE.format(prompt=prompt, chosen=chosen, rejected=rejected, scores=scores)},
        ]
        # simple retry loop via DeepSeekJudge.judge is not usable (it expects A/B winner schema),
        # so we use the client directly with similar retry behavior.
        import time
        last_err = None
        for attempt in range(6):
            try:
                resp = judge.client.chat.completions.create(
                    model=args.judge_model,
                    messages=msg,
                    temperature=0.0,
                    stream=False,
                )
                content = resp.choices[0].message.content
                out = json.loads(content)
                tagged.append({**r, "tag": out.get("tag", "other"), "llm_explanation": out.get("explanation", ""), "llm_which_is_better": out.get("which_is_better", "tie")})
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (2 ** attempt))
        else:
            raise RuntimeError(f"Failed on one example: {last_err}")

    # summary counts
    from collections import Counter
    cnt = Counter([t.get("tag", "other") for t in tagged])
    summary = {"n": len(tagged), "tag_counts": dict(cnt)}
    save_json(out_dir / "error_tags_summary.json", summary)
    save_json(out_dir / "error_tags.json", tagged)

    # markdown
    md = []
    md.append("# Step 4: DeepSeek Error Tagging (Reward Model Failure Modes)\n")
    md.append("## Tag counts\n")
    for k, v in cnt.most_common():
        md.append(f"- {k}: {v}")
    md.append("\n\n## Examples\n")
    for i, t in enumerate(tagged[: min(20, len(tagged))]):
        md.append(f"### Case {i+1}\n")
        md.append("**Prompt**\n```text\n" + str(t.get("prompt","")).strip() + "\n```\n")
        md.append("**Chosen**\n```text\n" + str(t.get("chosen","")).strip() + "\n```\n")
        md.append("**Rejected**\n```text\n" + str(t.get("rejected","")).strip() + "\n```\n")
        md.append(f"**Tag**: {t.get('tag','other')}\n")
        md.append("**Explanation**: " + str(t.get("llm_explanation","")).strip() + "\n")
    (out_dir / "step4_error_tags.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
