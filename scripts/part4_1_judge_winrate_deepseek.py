import argparse, json, os, random, time
from pathlib import Path
from openai import OpenAI

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def index_by_prompt_id(rows):
    return {r["prompt_id"]: r for r in rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--reference", required=True, help="e.g. sft")
    ap.add_argument("--candidate", required=True, help="e.g. ppo / dpo / grpo / base")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge_model", default="deepseek-chat")  # cheaper; can set deepseek-reasoner
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment variables.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    samples_dir = Path(args.samples_dir)
    ref_path = samples_dir / f"samples_{args.reference}.jsonl"
    cand_path = samples_dir / f"samples_{args.candidate}.jsonl"
    ref = index_by_prompt_id(load_jsonl(ref_path))
    cand = index_by_prompt_id(load_jsonl(cand_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_{args.candidate}_vs_{args.reference}.jsonl"

    done = set()
    if args.resume and out_path.exists():
        for r in load_jsonl(out_path):
            done.add(r["prompt_id"])
        print(f"[resume] loaded {len(done)} judged examples")

    rng = random.Random(args.seed)
    prompt_ids = sorted(set(ref.keys()) & set(cand.keys()))[:args.n_eval]

    system = (
        "You are a careful evaluator. Output MUST be valid json.\n"
        "Decide which assistant response is better for the given user prompt.\n"
        "Criteria: helpfulness, correctness, clarity, harmlessness.\n"
        "Return ONLY a json object in this format:\n"
        '{ "winner": "A" | "B" | "tie", "confidence": 0.0, "reason": "..." }\n'
    )

    def judge_one(prompt, a_text, b_text):
        user = (
            "Please judge in json.\n\n"
            f"USER PROMPT:\n{prompt}\n\n"
            f"RESPONSE A:\n{a_text}\n\n"
            f"RESPONSE B:\n{b_text}\n"
        )

        for attempt in range(3):
            resp = client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=args.max_tokens,
                stream=False,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return json.loads(content)
            time.sleep(1.0 + attempt)
        return {"winner": "tie", "confidence": 0.0, "reason": "empty_response_after_retries"}

    with out_path.open("a", encoding="utf-8") as f:
        for pid in prompt_ids:
            if pid in done:
                continue

            p = ref[pid]["prompt"]
            r_ref = ref[pid]["response"]
            r_cand = cand[pid]["response"]

            # randomize A/B
            if rng.random() < 0.5:
                A, B = r_cand, r_ref
                map_back = {"A": args.candidate, "B": args.reference}
            else:
                A, B = r_ref, r_cand
                map_back = {"A": args.reference, "B": args.candidate}

            j = judge_one(p, A, B)
            winner = j.get("winner", "tie")
            winner_model = map_back.get(winner, "tie") if winner in ("A", "B") else "tie"

            row = {
                "prompt_id": pid,
                "reference": args.reference,
                "candidate": args.candidate,
                "judge_model": args.judge_model,
                "winner_raw": winner,
                "winner_model": winner_model,
                "confidence": j.get("confidence", 0.0),
                "reason": j.get("reason", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    # summary
    rows = load_jsonl(out_path)
    wins = sum(r["winner_model"] == args.candidate for r in rows)
    ties = sum(r["winner_model"] == "tie" for r in rows)
    n = len(rows)
    print({"n": n, "wins": wins, "ties": ties, "win_rate": wins / n if n else None, "file": str(out_path)})

if __name__ == "__main__":
    main()
