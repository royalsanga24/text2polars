"""Score a model on the dev set. The number-producing machine.

    python -m evals.run_eval --model ollama:qwen2.5-coder:1.5b
    python -m evals.run_eval --model claude:claude-opus-5 --prompt v2
    python -m evals.run_eval --list                  # every past run, side by side

Every run is saved to results/ with enough metadata to be comparable weeks
later: which model, which prompt, and a hash of the task set. That last one
matters — if you add or edit tasks, old scores are no longer comparable to new
ones, and without the hash you will not notice. You will just see a number move
and believe you caused it.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probe"))

from execution import execute, matches                # noqa: E402
import importlib                                      # noqa: E402
from prompts import PROMPTS                                          # noqa: E402
from run_probe import ask_claude, ask_mlx, ask_ollama, extract_code  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def task_set_hash(tasks) -> str:
    """Fingerprint the tasks. Changes if any instruction, setup or answer changes."""
    blob = json.dumps(
        [[t["id"], t["instruction"], t["setup"], t["expected"]] for t in tasks],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def get_model_fn(spec):
    kind, _, name = spec.partition(":")
    try:
        return {"ollama": ask_ollama, "claude": ask_claude, "mlx": ask_mlx}[kind], name
    except KeyError:
        raise SystemExit(f"unknown backend {kind!r}; use ollama:<name>, claude:<id> or mlx:<repo>[@adapter]")


def list_runs(CATEGORIES):
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print("no runs yet")
        return
    runs = [json.loads(f.read_text()) for f in files]
    runs = [r for r in runs if r.get("tasks", "dev") == "dev"] or runs
    hashes = {r["task_set_hash"] for r in runs}
    w = max(len(r["model"]) for r in runs) + 2

    header = f"{'model':<{w}}{'prompt':<9}{'overall':>9}" + "".join(f"{c[:9]:>11}" for c in CATEGORIES)
    print(header + "\n" + "-" * len(header))
    for r in sorted(runs, key=lambda r: -r["overall"]):
        row = f"{r['model']:<{w}}{r['prompt']:<9}{r['overall']:>8.1%} "
        row += "".join(f"{r['by_category'][c]:>10.1%} " for c in CATEGORIES)
        print(row)
    if len(hashes) > 1:
        print("\n  ⚠️  runs above used DIFFERENT task sets — the numbers are not")
        print("      comparable to each other. Task-set hashes seen: " + ", ".join(sorted(hashes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="ollama:<name> or claude:<model-id>")
    ap.add_argument("--tasks", default="dev", help="dev | general")
    ap.add_argument("--prompt", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--limit", type=int, help="only run the first N tasks (for debugging)")
    ap.add_argument("--note", default="", help="free text stored with the result")
    ap.add_argument("--list", action="store_true", help="show past runs and exit")
    args = ap.parse_args()

    _m = importlib.import_module(f"evals.{args.tasks}_tasks")
    TASKS, CATEGORIES = _m.TASKS, _m.CATEGORIES
    PREAMBLE = getattr(_m, "PREAMBLE", "import polars as pl")

    if args.list:
        return list_runs(CATEGORIES)
    if not args.model:
        raise SystemExit("--model is required (or use --list)")

    ask, name = get_model_fn(args.model)
    tasks = TASKS[: args.limit] if args.limit else TASKS
    tshash = task_set_hash(tasks)

    print(f"model:    {args.model}")
    print(f"prompt:   {args.prompt}")
    print(f"tasks:    {len(tasks)}  (set {tshash})\n")

    records, by_cat = [], defaultdict(lambda: [0, 0])
    t0 = time.time()

    for i, t in enumerate(tasks, 1):
        prompt = PROMPTS[args.prompt].format(
            preamble=PREAMBLE or "# plain Python, no imports needed", setup=t["setup"], instruction=t["instruction"])
        try:
            raw = ask(name, prompt)
            code = extract_code(raw)
            ok, got = execute(t["setup"], code, PREAMBLE)
        except Exception as e:                     # model call failed, not the code
            raw, code, ok, got = "", "", False, f"{type(e).__name__}: {e}"

        good = bool(ok and matches(got, t["expected"]))
        by_cat[t["category"]][0] += good
        by_cat[t["category"]][1] += 1
        records.append(dict(id=t["id"], category=t["category"], passed=good,
                            ran=ok, produced=got if ok else str(got),
                            expected=t["expected"], code=code,
                            # RAW length, not the extracted snippet: a model can
                            # emit clean code then ramble for 400 more tokens.
                            # Measuring `code` hid exactly that (NOTES F35).
                            raw_chars=len(raw)))
        print(f"  [{i:>2}/{len(tasks)}] {t['id']:<24} "
              f"{'PASS' if good else ('WRONG' if ok else 'ERROR')}")

    lens = sorted(r.get("raw_chars", len(r["code"])) for r in records)
    med = lens[len(lens) // 2] if lens else 0
    runaway = sum(1 for L in lens if L > 400)
    n_pass = sum(r["passed"] for r in records)
    overall = n_pass / len(tasks)
    per_cat = {c: (by_cat[c][0] / by_cat[c][1] if by_cat[c][1] else 0.0) for c in CATEGORIES}
    elapsed = time.time() - t0

    print("\n" + "=" * 58)
    for c in CATEGORIES:
        hit, tot = by_cat[c]
        if tot:
            print(f"  {c:<20} {hit:>2}/{tot}  {hit/tot:>6.1%}  {'#' * round(20*hit/tot)}")
    print("-" * 58)
    print(f"  {'OVERALL':<20} {n_pass:>2}/{len(tasks)}  {overall:>6.1%}")
    print(f"  {'ran but wrong':<20} {sum(1 for r in records if r['ran'] and not r['passed']):>2}")
    print(f"  {'failed to run':<20} {sum(1 for r in records if not r['ran']):>2}")
    print(f"  {'median raw chars':<20} {med:>2}")
    print(f"  {'runaway (>400 ch)':<20} {runaway:>2}"
          + ("   <- model is not stopping cleanly" if runaway else ""))
    print(f"  {'seconds':<20} {elapsed:>2.0f}")
    print("=" * 58)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = args.model.replace("/", "_").replace(":", "-")
    out = RESULTS / f"{safe}__{args.tasks}__{args.prompt}__{stamp}.json"
    out.write_text(json.dumps(dict(
        model=args.model, prompt=args.prompt, note=args.note, tasks=args.tasks,
        task_set_hash=tshash, n_tasks=len(tasks), timestamp=stamp,
        overall=overall, by_category=per_cat, elapsed_seconds=elapsed,
        median_chars=med, runaway=runaway,
        records=records,
    ), indent=2, default=str))
    print(f"\nwrote {out.relative_to(RESULTS.parent)}")

    fails = [r for r in records if not r["passed"]]
    if fails:
        print("\nfirst 3 failures — read these, they are the information:")
        for r in fails[:3]:
            print(f"\n  {r['id']} ({r['category']})")
            print(f"    wrote:    {r['code'][:120].replace(chr(10),' ; ')}")
            print(f"    produced: {str(r['produced'])[:90]}")
            print(f"    expected: {r['expected']!r}")


if __name__ == "__main__":
    main()
