import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant_dir", type=str, required=True)
    ap.add_argument("--qual_dir", type=str, required=True)
    ap.add_argument("--samples_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, default="ANALYSIS.md")
    args = ap.parse_args()

    qdir = Path(args.quant_dir)
    ldir = Path(args.qual_dir)
    sdir = Path(args.samples_dir)
    out_path = Path(args.out_path)

    parts = []
    parts.append("# ANALYSIS\n")
    parts.append("This document contains Part 4 outputs (quantitative + qualitative evaluation).\n")

    qmd = qdir / "part4_1_quant_eval.md"
    parts.append(qmd.read_text(encoding="utf-8") if qmd.exists() else "## Part 4.1 Quantitative Evaluation\n(Missing)\n")

    parts.append("\n---\n")

    lmd = ldir / "part4_2_qual_analysis.md"
    parts.append(lmd.read_text(encoding="utf-8") if lmd.exists() else "## Part 4.2 Qualitative Analysis\n(Missing)\n")

    parts.append("\n---\n")
    parts.append("## Generated Samples\n")
    smd = sdir / "samples_all.md"
    parts.append(smd.read_text(encoding="utf-8") if smd.exists() else "(Missing)\n")

    out_path.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
