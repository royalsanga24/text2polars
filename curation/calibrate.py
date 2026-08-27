"""Test the decontamination filter against realistic contamination.

Two failure modes, and they are NOT symmetric:

  MISS        — contaminated example gets through. Eval score inflated, and you
                never find out. Catastrophic and silent.
  FALSE ALARM — legitimate training example discarded. You lose a little data.
                Annoying, harmless.

So prefer catching everything at the cost of throwing some good data away.

The test set below encodes what we decided contamination actually means:
same DATA and same ANSWER is fatal; same skill on different data is just
training data and must NOT be flagged.
"""

import sys

sys.path.insert(0, ".")
from curation.decontaminate import Decontaminator  # noqa: E402
from evals.dev_tasks import TASKS                  # noqa: E402


def reword(text):
    for a, b in [("Keep only the rows", "Filter to records"), ("Return", "Give back"),
                 ("greater than", "above"), ("as a list", "as a Python list"),
                 ("column", "field"), ("Group by", "Aggregate by")]:
        text = text.replace(a, text and b)
    return text


def mutate_data(setup):
    """Change the actual DATA — digits and string literals both.

    An earlier version only bumped digits, so tasks whose data is all strings
    were left identical and correctly flagged. The test was wrong, not the
    detector. When a test fails, suspect the test.
    """
    out = []
    for ch in setup:
        if ch.isdigit():
            out.append(str((int(ch) + 3) % 10))
        elif ch.islower() and ch.isalpha():
            out.append(ch)
        else:
            out.append(ch)
    s = "".join(out)
    # rename the single-letter string values that appear inside quotes
    for a, b in [("'a'", "'p'"), ("'b'", "'q'"), ("'c'", "'r'"), ("'x'", "'u'"),
                 ("'y'", "'w'"), ("'z'", "'t'"), ("alice", "zara"), ("bob", "kai"),
                 ("cara", "nils"), ("hr", "ops"), ("eng", "sales")]:
        s = s.replace(a, b)
    return s


def main():
    d = Decontaminator(TASKS)

    # --- SHOULD be caught -----------------------------------------------
    contaminated = []
    for t in TASKS:
        contaminated.append(("verbatim copy", dict(
            instruction=t["instruction"], setup=t["setup"],
            expected=t["expected"], solution=t["solution"])))
        contaminated.append(("reworded instruction, same data", dict(
            instruction=reword(t["instruction"]), setup=t["setup"],
            expected=t["expected"], solution="")))

    # --- should NOT be caught -------------------------------------------
    legitimate = []
    for t in TASKS:
        legitimate.append(("same skill, different data", dict(
            instruction=t["instruction"], setup=mutate_data(t["setup"]),
            expected="DIFFERENT", solution="")))
    # Genuinely novel tasks — NOT other dev tasks, which really are in the
    # dev set and should be flagged.
    NOVEL = [
        ("Return the median of 'price' as a float.",
         "df = pl.DataFrame({'price': [10.0, 20.0, 30.0]})", 20.0),
        ("Count rows where 'status' equals 'open'.",
         "df = pl.DataFrame({'status': ['open', 'shut', 'open']})", 2),
        ("Return the 'sku' values that start with 'A', as a list.",
         "df = pl.DataFrame({'sku': ['A1', 'B2', 'A3']})", ["A1", "A3"]),
        ("Return the standard deviation of 'temp' rounded to 3 places.",
         "df = pl.DataFrame({'temp': [10.0, 12.0, 14.0]})", 2.0),
        ("Concatenate all values of 'w' with a hyphen into one string.",
         "df = pl.DataFrame({'w': ['red', 'blue']})", "red-blue"),
        ("Return the index positions where 'flag' is true, as a list.",
         "df = pl.DataFrame({'flag': [False, True, True]})", [1, 2]),
        ("Return the product of all values in 'q' as an integer.",
         "df = pl.DataFrame({'q': [2, 3, 4]})", 24),
        ("Return the 'day' column reversed, as a list.",
         "df = pl.DataFrame({'day': ['mon', 'tue', 'wed']})", ["wed", "tue", "mon"]),
    ]
    for instr, setup, exp in NOVEL:
        legitimate.append(("genuinely novel task", dict(
            instruction=instr, setup=setup, expected=exp, solution="")))

    print(f"{'group':<34}{'n':>5}{'flagged':>9}{'verdict':>12}")
    print("-" * 60)

    def run(items, want_flagged):
        by_kind = {}
        for kind, c in items:
            hit = d.check(c["instruction"], c["setup"], c["solution"], c["expected"])
            n, f = by_kind.get(kind, (0, 0))
            by_kind[kind] = (n + 1, f + (hit is not None))
        for kind, (n, f) in by_kind.items():
            ok = (f == n) if want_flagged else (f == 0)
            print(f"{kind:<34}{n:>5}{f:>9}{'OK' if ok else 'PROBLEM':>12}")
        return by_kind

    print("MUST be caught:")
    caught = run(contaminated, True)
    print("\nMUST NOT be caught:")
    clean = run(legitimate, False)

    missed = sum(n - f for n, f in caught.values())
    false = sum(f for _, f in clean.values())
    total_c = sum(n for n, _ in caught.values())
    total_l = sum(n for n, _ in clean.values())
    print("\n" + "=" * 60)
    print(f"  missed contamination : {missed}/{total_c}   <- must be 0")
    print(f"  false alarms         : {false}/{total_l}")
    print("=" * 60)


if __name__ == "__main__":
    main()
