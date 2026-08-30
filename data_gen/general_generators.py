"""Template families for ORDINARY Python — the replay data.

Why this exists: training only on "here is a df, write polars" made the model
forget how to write plain Python (NOTES F37). Mixing these back in reminds it
that not every task involves a DataFrame.

These must NOT overlap with evals/general_tasks.py — using the eval as training
data would be "fixing" forgetting by memorising its test. build_sft.py screens
against both eval sets.

Independence note: for polars tasks, `expected` (plain Python) and `solution`
(polars) are genuinely independent implementations. Here both are Python, so
the check is weaker. We keep what independence we can by computing `expected`
with plain loops and writing `solution` in idiomatic comprehension style — a
mistake in one is unlikely to be mirrored exactly in the other.
"""

import random
from typing import Callable, List

WORDS = ["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "ibis", "jay",
         "kiwi", "lark", "mole", "newt", "owl", "pig"]
NAMES = ["alice", "bob", "cara", "dan", "eve", "finn", "gia", "hal"]
LISTV = ["nums", "values", "items", "data", "xs", "scores", "counts"]
STRV = ["s", "text", "line", "word", "name", "label"]
DICTV = ["d", "cfg", "mapping", "lookup", "opts"]


def _lit(v):
    return repr(v)


def f_filter_list(rng):
    v = rng.choice(LISTV)
    nums = [rng.randint(-20, 60) for _ in range(rng.randint(4, 7))]
    kind = rng.choice(["even", "odd", "positive", "above"])
    thr = rng.randint(5, 30)
    if kind == "even":
        keep, desc, cond = [n for n in nums if n % 2 == 0], "even", "n % 2 == 0"
    elif kind == "odd":
        keep, desc, cond = [n for n in nums if n % 2 == 1], "odd", "n % 2 == 1"
    elif kind == "positive":
        keep, desc, cond = [n for n in nums if n > 0], "positive", "n > 0"
    else:
        keep, desc, cond = [n for n in nums if n > thr], f"greater than {thr}", f"n > {thr}"
    phrasing = (f"Return only the numbers in `{v}` that are {desc}, as a list, in the original order."
                if desc.startswith("greater")
                else f"Return only the {desc} numbers from `{v}`, as a list, in the original order.")
    return dict(category="general_python",
                instruction=phrasing,
                setup=f"{v} = {_lit(nums)}",
                expected=keep,
                solution=f"result = [n for n in {v} if {cond}]")


def f_map_list(rng):
    v = rng.choice(LISTV)
    nums = [rng.randint(1, 20) for _ in range(rng.randint(3, 6))]
    k = rng.randint(2, 5)
    kind, py, expr = rng.choice([
        ("multiplied by", lambda n: n * k, f"n * {k}"),
        ("increased by", lambda n: n + k, f"n + {k}"),
        ("squared", lambda n: n * n, "n * n"),
    ])
    label = f"{kind} {k}" if kind != "squared" else "squared"
    out = []
    for n in nums:
        out.append(py(n))
    return dict(category="general_python",
                instruction=f"Return every value in `{v}` {label}, as a list.",
                setup=f"{v} = {_lit(nums)}",
                expected=out,
                solution=f"result = [{expr} for n in {v}]")


def f_reduce(rng):
    v = rng.choice(LISTV)
    nums = [rng.randint(1, 40) for _ in range(rng.randint(3, 6))]
    kind, expr = rng.choice([("sum", "sum"), ("largest value", "max"),
                             ("smallest value", "min"), ("number of items", "len")])
    tot = 0
    if kind == "sum":
        for n in nums:
            tot += n
    elif kind == "largest value":
        tot = nums[0]
        for n in nums:
            tot = n if n > tot else tot
    elif kind == "smallest value":
        tot = nums[0]
        for n in nums:
            tot = n if n < tot else tot
    else:
        for _ in nums:
            tot += 1
    return dict(category="general_python",
                instruction=f"Return the {kind} in `{v}` as a plain integer.",
                setup=f"{v} = {_lit(nums)}",
                expected=tot,
                solution=f"result = {expr}({v})")


def f_sort_take(rng):
    v = rng.choice(LISTV)
    nums = rng.sample(range(1, 90), rng.randint(4, 7))
    k = rng.randint(2, 3)
    desc = rng.choice([True, False])
    order = "largest" if desc else "smallest"
    srt = sorted(nums, reverse=desc)
    return dict(category="general_python",
                instruction=f"Return the {k} {order} values in `{v}`, ordered from "
                            f"{'highest to lowest' if desc else 'lowest to highest'}, as a list.",
                setup=f"{v} = {_lit(nums)}",
                expected=srt[:k],
                solution=f"result = sorted({v}, reverse={desc})[:{k}]")


def f_dedupe(rng):
    v = rng.choice(LISTV)
    pool = rng.sample(WORDS, 4)
    items = [rng.choice(pool) for _ in range(rng.randint(5, 8))]
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return dict(category="general_python",
                instruction=f"Remove duplicates from `{v}` while keeping the original order. Return the list.",
                setup=f"{v} = {_lit(items)}",
                expected=out,
                solution=f"result = list(dict.fromkeys({v}))")


def f_string_transform(rng):
    v = rng.choice(STRV)
    w = rng.choice(WORDS + NAMES)
    kind, py, expr = rng.choice([
        ("uppercase", lambda x: x.upper(), f"{v}.upper()"),
        ("reversed", lambda x: x[::-1], f"{v}[::-1]"),
        ("with its first letter capitalised", lambda x: x.capitalize(), f"{v}.capitalize()"),
    ])
    return dict(category="general_python",
                instruction=f"Return `{v}` {kind}.",
                setup=f"{v} = {_lit(w)}",
                expected=py(w),
                solution=f"result = {expr}")


def f_split_join(rng):
    v = rng.choice(STRV)
    parts = rng.sample(WORDS, rng.randint(3, 4))
    sep, joiner = rng.choice([(",", " - "), ("|", ", "), (";", "/")])
    line = sep.join(f" {p} " for p in parts)
    out = []
    for p in line.split(sep):
        out.append(p.strip())
    return dict(category="general_python",
                instruction=f"Split `{v}` on {sep!r}, strip whitespace from each piece, "
                            f"and return the pieces as a list.",
                setup=f"{v} = {_lit(line)}",
                expected=out,
                solution=f"result = [p.strip() for p in {v}.split({sep!r})]")


def f_zip_dict(rng):
    ks, vs = rng.sample(WORDS, 3), [rng.randint(1, 50) for _ in range(3)]
    out = {}
    for k, val in zip(ks, vs):
        out[k] = val
    return dict(category="general_python",
                instruction="Combine `keys` and `vals` into a dictionary.",
                setup=f"keys = {_lit(ks)}\nvals = {_lit(vs)}",
                expected=out,
                solution="result = dict(zip(keys, vals))")


def f_dict_invert(rng):
    v = rng.choice(DICTV)
    d = {w: rng.choice(WORDS) for w in rng.sample(NAMES, 3)}
    out = {}
    for k, val in d.items():
        out[val] = k
    return dict(category="general_python",
                instruction=f"Return `{v}` with its keys and values swapped.",
                setup=f"{v} = {_lit(d)}",
                expected=out,
                solution=f"result = {{v: k for k, v in {v}.items()}}")


def f_dict_merge(rng):
    a = {w: rng.randint(1, 9) for w in rng.sample(WORDS, 2)}
    shared = rng.choice(list(a))
    b = {shared: rng.randint(10, 20), rng.choice([w for w in WORDS if w not in a]): rng.randint(1, 9)}
    out = dict(a)
    for k, val in b.items():
        out[k] = val
    return dict(category="general_python",
                instruction="Merge `d1` and `d2` into one dictionary. Where a key appears in both, "
                            "`d2` wins. Return the merged dictionary.",
                setup=f"d1 = {_lit(a)}\nd2 = {_lit(b)}",
                expected=out,
                solution="result = {**d1, **d2}")


def f_group_by_letter(rng):
    v = rng.choice(LISTV)
    items = rng.sample(WORDS, rng.randint(4, 6))
    out = {}
    for w in items:
        if w[0] not in out:
            out[w[0]] = []
        out[w[0]].append(w)
    return dict(category="general_python",
                instruction=f"Group the words in `{v}` into a dictionary keyed by their first "
                            f"letter. Each value is a list of words in the original order.",
                setup=f"{v} = {_lit(items)}",
                expected=out,
                solution=(f"result = {{}}\nfor w in {v}:\n"
                          f"    result.setdefault(w[0], []).append(w)"))


def f_count_items(rng):
    v = rng.choice(LISTV)
    pool = rng.sample(WORDS, 3)
    items = [rng.choice(pool) for _ in range(rng.randint(5, 9))]
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(category="general_python",
                instruction=f"Count how many times each value appears in `{v}`. "
                            f"Return a dictionary mapping value to count.",
                setup=f"{v} = {_lit(items)}",
                expected=out,
                solution=f"result = {{x: {v}.count(x) for x in set({v})}}")


def f_label_values(rng):
    v = rng.choice(LISTV)
    nums = [rng.randint(1, 100) for _ in range(rng.randint(3, 6))]
    thr = rng.choice([10, 25, 50])
    hi, lo = rng.choice([("high", "low"), ("pass", "fail"), ("big", "small")])
    out = []
    for n in nums:
        out.append(hi if n >= thr else lo)
    return dict(category="general_python",
                instruction=f"Label each value in `{v}` as {hi!r} when it is {thr} or more, "
                            f"and {lo!r} otherwise. Return the labels as a list.",
                setup=f"{v} = {_lit(nums)}",
                expected=out,
                solution=f"result = [{hi!r} if n >= {thr} else {lo!r} for n in {v}]")


def f_dict_default(rng):
    v = rng.choice(DICTV)
    d = {w: rng.randint(1, 9) for w in rng.sample(WORDS, 2)}
    missing = rng.choice([w for w in WORDS if w not in d])
    default = rng.choice([0, -1, 100])
    return dict(category="general_python",
                instruction=f"Return the value stored under the key {missing!r} in `{v}`, "
                            f"or {default} if that key is not present.",
                setup=f"{v} = {_lit(d)}",
                expected=default,
                solution=f"result = {v}.get({missing!r}, {default})")


def f_flatten(rng):
    v = rng.choice(LISTV)
    nested = [[rng.randint(1, 9) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(2, 4))]
    out = []
    for sub in nested:
        for x in sub:
            out.append(x)
    return dict(category="general_python",
                instruction=f"Flatten `{v}`, a list of lists, into a single list.",
                setup=f"{v} = {_lit(nested)}",
                expected=out,
                solution=f"result = [x for sub in {v} for x in sub]")


FAMILIES: List[Callable] = [
    f_filter_list, f_map_list, f_reduce, f_sort_take, f_dedupe,
    f_string_transform, f_split_join, f_zip_dict, f_dict_invert, f_dict_merge,
    f_group_by_letter, f_count_items, f_label_values, f_dict_default, f_flatten,
]
PREAMBLE = ""


def generate(n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)]
        ex = fam(rng)
        ex["template"] = fam.__name__
        yield ex
