"""Screen a RAW TEXT corpus (for CPT) against the eval sets.

Different risk from the task-level screening in decontaminate.py. There we
asked "is this training example secretly an eval example?". Here the corpus is
raw source code, and the danger is narrower but real:

    the corpus contains a snippet that IS one of our eval tasks —
    the same toy DataFrame and the same answer

That is plausible: our eval tasks use small illustrative DataFrames, and so
does polars' own test suite. A model that memorises `pl.DataFrame({'a': [1, 2],
'b': ['x','y']}).to_dict()` and its output has not learned `to_dict`; it has
learned our answer key.

What is NOT contamination here: the corpus containing the OPERATIONS our evals
test. That is the entire point of CPT — see NOTES F54. Screening those out
would remove the most common operations in the library and defeat the exercise.

    python -m curation.decontaminate_text --corpus data/real/corpus.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from curation.decontaminate import exact_jaccard, normalize, shingles  # noqa: E402


def load_eval_tasks():
    import importlib
    tasks = []
    for mod in ("dev_tasks", "general_tasks", "test_tasks"):
        m = importlib.import_module(f"evals.{mod}")
        for t in m.TASKS:
            tasks.append({**t, "source_set": mod})
    return tasks


def frame_literals(text: str):
    """Extract DataFrame constructor literals — the fingerprint of a task's data."""
    return re.findall(r"pl\.DataFrame\(\s*\{[^}]{0,400}\}", text)


def screen(corpus, tasks, k=4, threshold=0.55, window=600):
    """Return (clean, hits). A hit needs the task's DATA **and** its ANSWER.

    Matching the data alone flags almost everything: `pl.DataFrame({'a':
    [1,2,3]})` appears in nearly every polars test file, and a model memorising
    that literal has learned nothing about our answer. Contamination is the
    data AND the answer appearing together — the same rule as D20 for
    task-level screening, arrived at the same way (an over-aggressive first
    version flagged 26% of the corpus, all of it innocent).
    """
    task_sh = [(t, shingles(t["setup"], k), normalize(t["setup"]),
                normalize(t["solution"])) for t in tasks]
    clean, hits = [], []
    for doc in corpus:
        src = doc["source"]
        src_n = normalize(src)
        worst = None
        for m in re.finditer(r"pl\.DataFrame\(\s*\{[^}]{0,400}\}", src):
            lit = m.group(0)
            lit_n, lit_sh = normalize(lit), shingles(lit, k)
            # only look for the answer near the data, not anywhere in the file
            near = normalize(src[m.start(): m.end() + window])
            for t, sh, setup_n, sol_n in task_sh:
                data_match = (setup_n and lit_n and
                              (setup_n in lit_n or lit_n in setup_n))
                if not data_match and exact_jaccard(lit_sh, sh) < threshold:
                    continue
                # the answer must be nearby too; short one-liners are excluded
                # because generic idioms are what we are trying to TEACH
                if len(sol_n) > 45 and sol_n in near:
                    worst = (t, 1.0, "task data AND solution together")
                    break
            if worst:
                break
        if worst:
            hits.append({"path": doc["path"], "task": worst[0]["id"],
                         "set": worst[0]["source_set"], "score": worst[1],
                         "reason": worst[2]})
        else:
            clean.append(doc)
    return clean, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/real/corpus.jsonl")
    ap.add_argument("--out", default="data/real/corpus_clean.jsonl")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    corpus = [json.loads(l) for l in open(args.corpus)]
    tasks = load_eval_tasks()
    print(f"  corpus     : {len(corpus)} documents")
    print(f"  eval tasks : {len(tasks)} across dev / general / held-out\n")

    clean, hits = screen(corpus, tasks, threshold=args.threshold)
    print(f"  flagged    : {len(hits)}")
    print(f"  clean      : {len(clean)}")
    for h in hits[:12]:
        print(f"    {h['path'][:58]:<60} ~ {h['task']} ({h['set']}, {h['score']:.2f}) {h['reason']}")
    if len(hits) > 12:
        print(f"    ... and {len(hits) - 12} more")

    Path(args.out).write_text("".join(json.dumps(d) + "\n" for d in clean))
    chars = sum(len(d["source"]) for d in clean)
    print(f"\n  wrote {args.out}  ({chars/1e6:.2f} MB, ~{chars/4/1e6:.2f}M tokens)")


if __name__ == "__main__":
    main()
