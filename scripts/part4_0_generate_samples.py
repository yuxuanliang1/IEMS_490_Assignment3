import argparse
import random
from pathlib import Path

from datasets import load_from_disk

from src.eval.common import get_device, set_seed, save_json, load_policy
from src.eval.generation import generate_responses


def parse_model_arg(s: str):
    if "=" not in s:
        raise ValueError("--models must be name=path_or_id")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--prompt_max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.data_dir)
    test = ds["test"]

    prompts = [test[i]["prompt"] for i in range(len(test))]
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: min(args.n_prompts, len(prompts))]

    all_samples = {"meta": vars(args), "prompts": prompts, "models": {}}

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
            batch_size=args.batch_size,
        )
        all_samples["models"][name] = [{"prompt": g["prompt"], "response": g["response_text"]} for g in gens]
        save_json(out_dir / f"samples_{name}.json", all_samples["models"][name])

    save_json(out_dir / "samples_all.json", all_samples)

    md = []
    md.append("# Generated Samples (≈20 per model)\n")
    for i, p in enumerate(prompts):
        md.append(f"## Prompt {i+1}\n")
        md.append("```text\n" + p.strip() + "\n```\n")
        for name in all_samples["models"].keys():
            r = all_samples["models"][name][i]["response"]
            md.append(f"### {name}\n")
            md.append("```text\n" + r.strip() + "\n```\n")
    (out_dir / "samples_all.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved samples to: {out_dir}")


if __name__ == "__main__":
    main()
