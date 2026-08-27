"""Prove every task is well-formed before any model is measured against it.

    python -m evals.validate

Checks, per task:
  1. the reference solution runs without error
  2. its output equals `expected`

`expected` and `solution` were written independently, so a disagreement means
one of them is wrong. Either way the task is unusable until fixed.

Why this exists: if a task is broken, EVERY model fails it. You'd see a low
score and conclude something about the model, when the truth is your test was
wrong. A broken task is worse than a missing one — it actively misleads.
"""

import argparse
import sys
from collections import Counter

sys.path.insert(0, ".")
from execution import execute, matches           # noqa: E402
import importlib                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="dev", help="dev | general")
    ap.add_argument("--quiet", action="store_true", help="only show failures")
    ap.add_argument("--review", metavar="CATEGORY", nargs="?", const="all",
                    help="print tasks for human reading instead of validating")
    args = ap.parse_args()

    _m = importlib.import_module(f"evals.{args.tasks}_tasks")
    TASKS, CATEGORIES = _m.TASKS, _m.CATEGORIES
    PREAMBLE = getattr(_m, "PREAMBLE", "import polars as pl")

    if args.review:
        tasks = [t for t in TASKS if args.review in ("all", t["category"])]
        print(f"{len(tasks)} task(s). For each, ask ONE question:\n"
              f"  does the instruction describe exactly what `expected` shows?\n"
              f"You do not need to read any polars.\n")
        for i, t in enumerate(tasks, 1):
            print(f"{'-'*70}\n[{i}/{len(tasks)}] {t['id']}   ({t['category']})")
            print(f"\n  GIVEN:")
            for line in t["setup"].splitlines():
                print(f"      {line}")
            print(f"\n  ASKED:    {t['instruction']}")
            print(f"\n  ANSWER:   {t['expected']!r}\n")
        return 0

    broken = []
    for t in TASKS:
        ok, got = execute(t["setup"], t["solution"], PREAMBLE)
        good = ok and matches(got, t["expected"])
        if not good:
            broken.append((t, ok, got))
        elif not args.quiet:
            print(f"  ok       {t['category']:<18} {t['id']}")

    print()
    if broken:
        print(f"{'='*66}\n  {len(broken)} BROKEN TASK(S) — fix or drop before measuring anything\n{'='*66}")
        for t, ok, got in broken:
            print(f"\n  {t['id']}  [{t['category']}]")
            print(f"    instruction: {t['instruction']}")
            print(f"    solution:    {t['solution'][:110]}")
            print(f"    {'produced' if ok else 'ERROR'}:    {got!r}")
            print(f"    expected:    {t['expected']!r}")

    n_ok = len(TASKS) - len(broken)
    print(f"\n{'='*66}")
    print(f"  {n_ok}/{len(TASKS)} tasks validated")
    for c in CATEGORIES:
        tot = sum(1 for t in TASKS if t["category"] == c)
        bad = sum(1 for t, _, _ in broken if t["category"] == c)
        print(f"    {c:<20} {tot-bad:>2}/{tot}")
    print("=" * 66)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
