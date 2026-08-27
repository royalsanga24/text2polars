"""Compare two runs properly: which tasks flipped, and is the split real?

    python -m evals.compare v1 v3
    python -m evals.compare v3 v4 --model ollama:qwen2.5-coder:1.5b

Two accuracies side by side invite a conclusion the data may not support.
What matters is the PAIRED comparison — the same tasks, run both ways — and
whether the tasks that changed direction did so more lopsidedly than chance.

The test is McNemar's, done exactly: of the N tasks whose outcome changed,
if the change had no real cause each one is a coin flip. How surprising is
the split we saw?
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.dev_tasks import CATEGORIES  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_latest(prompt, model=None, tasks="dev"):
    best = None
    for f in RESULTS.glob("*.json"):
        d = json.loads(f.read_text())
        if d["prompt"] != prompt:
            continue
        # Without this, a `--tasks general` run (24 tasks) can be picked up as
        # the match for a dev run (48 tasks). The task-set hash caught it, but
        # a guard you rely on tripping is a guard you should not need.
        if d.get("tasks", "dev") != tasks:
            continue
        if model and d["model"] != model:
            continue
        if best is None or d["timestamp"] > best["timestamp"]:
            best = d
    if best is None:
        raise SystemExit(f"no run found for prompt={prompt}" + (f" model={model}" if model else ""))
    return best


def exact_two_sided(n01, n10):
    """P(split at least this lopsided | each flip is a coin toss)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--model", help="model for both sides")
    ap.add_argument("--model-a", help="override model for side a")
    ap.add_argument("--model-b", help="override model for side b")
    ap.add_argument("--tasks", default="dev", help="dev | general")
    args = ap.parse_args()

    A = load_latest(args.a, args.model_a or args.model, args.tasks)
    B = load_latest(args.b, args.model_b or args.model, args.tasks)
    la, lb = args.a, args.b
    if args.model_a or args.model_b:
        la = f"{A['model'].split(':', 1)[1]} {args.a}"
        lb = f"{B['model'].split(':', 1)[1]} {args.b}"

    if A["task_set_hash"] != B["task_set_hash"]:
        raise SystemExit(
            f"REFUSING to compare: different task sets "
            f"({A['task_set_hash']} vs {B['task_set_hash']}). Re-run both.")
    if A["model"] != B["model"]:
        print(f"note: different models ({A['model']} vs {B['model']})\n")

    pa = {r["id"]: r["passed"] for r in A["records"]}
    pb = {r["id"]: r["passed"] for r in B["records"]}
    cat = {r["id"]: r["category"] for r in A["records"]}

    fixed  = sorted(i for i in pa if not pa[i] and pb[i])
    broke  = sorted(i for i in pa if pa[i] and not pb[i])
    p = exact_two_sided(len(fixed), len(broke))

    print(f"  {la}: {A['overall']:.1%}      {lb}: {B['overall']:.1%}"
          f"      ({B['overall'] - A['overall']:+.1%})\n")
    print(f"  {lb} fixed  ({len(fixed)}): {', '.join(fixed) or '-'}")
    print(f"  {lb} broke  ({len(broke)}): {', '.join(broke) or '-'}")

    print(f"\n  {len(fixed) + len(broke)} tasks changed outcome; exact two-sided p = {p:.3f}")
    if p < 0.05:
        print(f"  -> REAL: {lb} differs from {la}")
    elif p < 0.15:
        print("  -> SUGGESTIVE but not conclusive. Needs more tasks to settle.")
    else:
        print(f"  -> NOT distinguishable from chance at this sample size")

    net = defaultdict(int)
    for i in fixed: net[cat[i]] += 1
    for i in broke: net[cat[i]] -= 1
    print("\n  net change by category (tasks):")
    for c in CATEGORIES:
        v = net[c]
        print(f"    {c:<20} {v:+d}  {'+' * max(v, 0)}{'-' * max(-v, 0)}")


if __name__ == "__main__":
    main()
