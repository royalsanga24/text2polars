"""Convert our verified SFT set into the format `mlx_lm lora` expects.

    python -m data_gen.to_mlx

Writes data/mlx/{train,valid}.jsonl as {"prompt": ..., "completion": ...}.
With `--mask-prompt`, mlx_lm computes loss only on the completion half — the
model learns to WRITE answers, not to reproduce questions.

TRAIN/SERVE SKEW is the danger here. The prompt written below must be byte-for
-byte what the eval sends at inference. Both come from prompts.py so they
cannot drift; never inline the prompt text in two places.

We train on v1 (no few-shot examples) deliberately: baking the format into the
weights is how you stop paying for few-shot examples on every single call.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from execution import PREAMBLE          # noqa: E402
from prompts import PROMPTS             # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "data" / "sft"
OUT = Path(__file__).resolve().parent.parent / "data" / "mlx"
TRAIN_PROMPT = "v1"


def build_prompt(setup: str, instruction: str, preamble: str = PREAMBLE) -> str:
    return PROMPTS[TRAIN_PROMPT].format(
        preamble=preamble or "# plain Python, no imports needed",
        setup=setup, instruction=instruction)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sft")
    ap.add_argument("--out", default="mlx")
    a = ap.parse_args()
    global SRC, OUT
    SRC = SRC.parent / a.src
    OUT = OUT.parent / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, out_name in (("train", "train"), ("val", "valid")):
        rows = [json.loads(l) for l in (SRC / f"{src_name}.jsonl").read_text().splitlines() if l.strip()]
        out = OUT / f"{out_name}.jsonl"
        with out.open("w") as fh:
            for r in rows:
                # Replay examples carry preamble="" — they must NOT be shown
                # "import polars as pl", or the replay data would still be
                # teaching "polars is always in scope", which is half the
                # problem it exists to fix. Must match what the eval sends.
                fh.write(json.dumps({
                    "prompt": build_prompt(r["setup"], r["instruction"],
                                           r.get("preamble", PREAMBLE)),
                    "completion": r["solution"],
                }) + "\n")
        print(f"wrote {out.relative_to(OUT.parent.parent)}  ({len(rows)} examples)")

    sample = json.loads((OUT / "train.jsonl").read_text().splitlines()[0])
    print(f"\n--- one example, exactly as the trainer sees it ---")
    print("PROMPT (masked out of the loss):")
    for line in sample["prompt"].splitlines():
        print(f"  | {line}")
    print("COMPLETION (what the loss is computed on):")
    for line in sample["completion"].splitlines():
        print(f"  | {line}")


if __name__ == "__main__":
    main()
