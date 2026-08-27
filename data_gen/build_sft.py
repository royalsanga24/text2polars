"""Generate, verify, decontaminate, and save the SFT training set.

    python -m data_gen.build_sft --n 3000

Four gates, in order. Each one only lets through what it can prove:

  1. GENERATE   template families produce candidates
  2. VERIFY     run every solution; drop it unless it produces `expected`
  3. DEDUPE     drop candidates identical to one already kept
  4. DECONTAMINATE  drop anything resembling a dev-set task

Nothing is trusted. A template I wrote wrong simply yields nothing, which is
why the per-template yield table at the end is worth reading — a family at 0%
is a bug in my code, not a hard task.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curation.decontaminate import Decontaminator, task_fingerprint  # noqa: E402
from data_gen.general_generators import generate as generate_general  # noqa: E402
from data_gen.generators import FAMILIES, generate                    # noqa: E402
from evals.dev_tasks import TASKS as DEV_TASKS                        # noqa: E402
from evals.general_tasks import TASKS as GENERAL_TASKS                # noqa: E402
from execution import execute_batch, matches                          # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "sft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="candidates to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default="sft", help="subdirectory under data/")
    ap.add_argument("--replay-frac", type=float, default=0.25,
                    help="fraction of the final set that is ordinary Python (NOTES F37). "
                         "0 reproduces the arm-1 dataset that caused catastrophic forgetting.")
    args = ap.parse_args()

    n_replay = int(args.n * args.replay_frac / max(1e-9, 1 - args.replay_frac))
    print(f"generating {args.n} polars candidates from {len(FAMILIES)} families")
    print(f"        + {n_replay} ordinary-Python replay candidates")
    cands = list(generate(args.n, seed=args.seed))
    replay = list(generate_general(n_replay, seed=args.seed + 1)) if n_replay else []
    for c in replay:
        c["preamble"] = ""

    # ---- gate 2: verify by execution --------------------------------------
    # Two batches: polars examples need `import polars as pl`, replay ones must
    # NOT have it — if the import were present the replay data would still be
    # teaching "polars is always in scope", which is half the problem.
    results = execute_batch([{"setup": c["setup"], "code": c["solution"]} for c in cands])
    if replay:
        results = results + execute_batch(
            [{"setup": c["setup"], "code": c["solution"]} for c in replay], preamble="")
        cands = cands + replay
    verified, drop = [], Counter()
    per_template = defaultdict(lambda: [0, 0])
    for c, (ok, got) in zip(cands, results):
        per_template[c["template"]][1] += 1
        if not ok:
            drop["solution crashed"] += 1
        elif not matches(got, c["expected"]):
            drop["solution disagreed with expected"] += 1
        else:
            per_template[c["template"]][0] += 1
            verified.append(c)
    print(f"  verified:        {len(verified):>5} / {len(cands)}")

    # ---- gate 3: dedupe ----------------------------------------------------
    seen, deduped = set(), []
    for c in verified:
        fp = task_fingerprint(c["setup"], c["expected"])
        if fp in seen:
            drop["duplicate of another candidate"] += 1
            continue
        seen.add(fp)
        deduped.append(c)
    print(f"  after dedupe:    {len(deduped):>5}")

    # ---- gate 4: decontaminate --------------------------------------------
    # Screen against BOTH eval sets. Screening only the polars eval would let
    # replay data memorise the general eval — "fixing" forgetting by learning
    # its test.
    d = Decontaminator(DEV_TASKS + GENERAL_TASKS)
    clean, hits = d.screen(deduped)
    drop["matched a dev task"] = len(hits)
    print(f"  after decontam:  {len(clean):>5}")
    if hits:
        print(f"    (blocked: {', '.join(sorted({h.dev_id for h in hits}))[:80]})")

    # ---- report ------------------------------------------------------------
    print(f"\n  dropped, by reason:")
    for reason, n in drop.most_common():
        if n:
            print(f"    {n:>5}  {reason}")

    print(f"\n  yield by template (low = my template is buggy, not hard):")
    for name, (ok, tot) in sorted(per_template.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        bar = "#" * round(20 * ok / max(tot, 1))
        print(f"    {name:<18} {ok:>4}/{tot:<4} {ok/max(tot,1):>5.0%} {bar}")

    import random as _r
    _r.Random(args.seed).shuffle(clean)   # interleave polars and replay

    print(f"\n  by category:")
    for cat, n in Counter(c["category"] for c in clean).most_common():
        print(f"    {cat:<20} {n:>5}")

    # ---- save --------------------------------------------------------------
    global OUT
    OUT = OUT.parent / args.out
    n_val = max(1, int(len(clean) * args.val_frac))
    val, train = clean[:n_val], clean[n_val:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(
            json.dumps({k: v for k, v in c.items()}, default=str) + "\n" for c in part))
        print(f"\nwrote {path.relative_to(OUT.parent.parent)}  ({len(part)} examples)")


if __name__ == "__main__":
    main()
