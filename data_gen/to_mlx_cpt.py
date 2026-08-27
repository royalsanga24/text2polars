"""Turn the screened real-code corpus into CPT training data.

    python -m data_gen.to_mlx_cpt

Writes data/mlx-cpt/{train,valid}.jsonl as {"text": ...} — mlx_lm's TextDataset
format. No prompts, no answers: CPT is next-token prediction over raw text, and
the loss applies to every token. `--mask-prompt` is meaningless here and
mlx_lm rejects it for text datasets.

Chunking matters: mlx_lm TRUNCATES anything longer than max_seq_length rather
than splitting it, so a 40 KB source file would contribute its first ~1000
tokens and silently discard the rest. We split on line boundaries instead, so
the whole corpus is actually used.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real" / "corpus_clean.jsonl"
OUT = ROOT / "data" / "mlx-cpt"


def chunk(text: str, max_chars: int):
    """Split on line boundaries, never mid-line."""
    out, buf, n = [], [], 0
    for line in text.splitlines(keepends=True):
        if n + len(line) > max_chars and buf:
            out.append("".join(buf))
            buf, n = [], 0
        buf.append(line)
        n += len(line)
    if buf:
        out.append("".join(buf))
    return [c for c in out if len(c.strip()) > 120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=3200, help="~800 tokens")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    chunks = []
    for d in docs:
        for c in chunk(d["source"], args.max_chars):
            chunks.append({"text": c})

    random.Random(args.seed).shuffle(chunks)
    n_val = max(1, int(len(chunks) * args.val_frac))
    val, train = chunks[:n_val], chunks[n_val:]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", val)):
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in part))
        chars = sum(len(c["text"]) for c in part)
        print(f"wrote {(OUT / (name + '.jsonl')).relative_to(ROOT)}  "
              f"{len(part)} chunks, {chars/1e6:.2f} MB, ~{chars/4/1e6:.2f}M tokens")

    print(f"\n  from {len(docs)} source files, all MIT (pola-rs/polars)")
    print(f"  median chunk: {sorted(len(c['text']) for c in chunks)[len(chunks)//2]} chars")


if __name__ == "__main__":
    main()
