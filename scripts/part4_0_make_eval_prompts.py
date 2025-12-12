import argparse, json, random
from pathlib import Path
from datasets import load_from_disk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    ds = load_from_disk(args.data_dir)
    if args.split not in ds:
        raise ValueError(f"split {args.split} not in {list(ds.keys())}")

    d = ds[args.split]
    n = min(args.n_prompts, len(d))
    rng = random.Random(args.seed)
    idxs = list(range(len(d)))
    rng.shuffle(idxs)
    idxs = idxs[:n]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(idxs):
            ex = d[int(idx)]
            prompt = ex["prompt"]
            f.write(json.dumps({"prompt_id": i, "prompt": prompt}, ensure_ascii=False) + "\n")

    print(f"Saved prompts: {n} -> {out_path}")

if __name__ == "__main__":
    main()
