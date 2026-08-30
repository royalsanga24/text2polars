"""Template families that produce polars training examples.

Each family is a function taking a random generator and returning one example:
    instruction, setup, expected, solution, category

TWO RULES, both load-bearing:

1. `expected` is computed in PLAIN PYTHON. `solution` is written in polars.
   Two independent implementations of the same thing. When they agree, that is
   evidence — not me marking my own homework. When they disagree, the example
   is dropped automatically.

2. Templates may be wrong. That is fine. build_sft.py executes every generated
   example and discards anything whose solution does not produce `expected`.
   Generate optimistically, verify ruthlessly.

Diversity comes from three places: randomised data values, randomised column
names, and a pool of phrasings per family. It is still less diverse than real
human instructions — a known limitation, and the reason the held-out test set
comes from real GitHub code rather than from here.
"""

import random
from typing import Callable, Dict, List

NUM_COLS = ["score", "amount", "qty", "price", "age", "value", "total", "n", "count", "weight"]
STR_COLS = ["name", "city", "label", "product", "user", "tag", "item", "title"]
GRP_COLS = ["team", "region", "dept", "category", "group", "status", "kind"]
NAMES = ["alice", "bob", "cara", "dan", "eve", "finn", "gia", "hal", "ivy", "jon", "kim", "leo"]
CITIES = ["delhi", "mumbai", "paris", "tokyo", "lima", "oslo", "cairo", "perth", "rome", "quito"]
GROUPS = ["a", "b", "c", "north", "south", "east", "west", "red", "blue", "green"]


def _lit(v):
    return repr(v)


def _frame(cols: Dict[str, list], name="df") -> str:
    inner = ", ".join(f"{_lit(k)}: {v!r}" for k, v in cols.items())
    return f"{name} = pl.DataFrame({{{inner}}})"


def _pick(rng, pool, n):
    return rng.sample(pool, n)


# ---------------------------------------------------------------- families

def f_filter_select(rng):
    """Keep rows above a threshold, return another column."""
    nc, sc = rng.choice(NUM_COLS), rng.choice(STR_COLS)
    n = rng.randint(3, 6)
    labels = _pick(rng, NAMES + CITIES, n)
    vals = [rng.randint(1, 100) for _ in range(n)]
    thr = rng.choice(sorted(vals)[1:-1]) if len(set(vals)) > 2 else 50
    op, pyop = rng.choice([(">", lambda a, b: a > b), ("<", lambda a, b: a < b),
                           (">=", lambda a, b: a >= b), ("<=", lambda a, b: a <= b)])
    phrasing = rng.choice([
        f"Keep only the rows where {sc!r} is {'greater' if '>' in op else 'less'} than "
        f"{'or equal to ' if '=' in op else ''}{thr} in the {nc!r} column, and return the {sc!r} column as a list.",
        f"Filter the rows to those with {nc!r} {op} {thr}, then return {sc!r} as a plain Python list.",
        f"Return the {sc!r} values for every row whose {nc!r} is {op} {thr}, as a list.",
    ])
    return dict(
        category="pandas_trap",
        instruction=phrasing,
        setup=_frame({sc: labels, nc: vals}),
        expected=[l for l, v in zip(labels, vals) if pyop(v, thr)],
        solution=f"result = df.filter(pl.col({nc!r}) {op} {thr})[{sc!r}].to_list()",
    )


def f_group_agg(rng):
    """Group by a column and aggregate another."""
    gc, nc = rng.choice(GRP_COLS), rng.choice(NUM_COLS)
    keys = _pick(rng, GROUPS, rng.randint(2, 3))
    n = rng.randint(4, 7)
    gvals = [rng.choice(keys) for _ in range(n)]
    gvals += [k for k in keys if k not in gvals]          # every key appears
    vals = [rng.randint(1, 50) for _ in gvals]
    agg, pyagg, expr = rng.choice([
        ("sum", sum, "sum()"),
        ("largest", max, "max()"),
        ("smallest", min, "min()"),
        ("number of rows", len, "len()"),
    ])
    buckets = {}
    for g, v in zip(gvals, vals):
        buckets.setdefault(g, []).append(v)
    if agg == "number of rows":
        expected = {g: len(vs) for g, vs in buckets.items()}
        agg_expr = f"pl.len().alias('r')"
    else:
        expected = {g: pyagg(vs) for g, vs in buckets.items()}
        agg_expr = f"pl.col({nc!r}).{expr}.alias('r')"
    if agg == "number of rows":
        phrasing = rng.choice([
            f"Count how many rows there are for each {gc!r}. Return a dict mapping {gc!r} to that count.",
            f"For each distinct {gc!r}, give the number of rows. Return the result as a dictionary.",
        ])
    else:
        phrasing = rng.choice([
            f"Group by {gc!r} and compute the {agg} of {nc!r} per group. Return a dict mapping {gc!r} to that value.",
            f"For each distinct {gc!r}, give the {agg} of {nc!r}. Return the result as a dictionary.",
            f"Aggregate {nc!r} by {gc!r} using the {agg}, and return a dict from group to result.",
        ])
    return dict(
        category="pandas_trap",
        instruction=phrasing,
        setup=_frame({gc: gvals, nc: vals}),
        expected=expected,
        solution=(f"g = df.group_by({gc!r}).agg({agg_expr})\n"
                  f"result = dict(zip(g[{gc!r}].to_list(), g['r'].to_list()))"),
    )


def f_sort_head(rng):
    """Sort and take the top few."""
    nc, sc = rng.choice(NUM_COLS), rng.choice(STR_COLS)
    n = rng.randint(4, 7)
    labels = _pick(rng, NAMES + CITIES, n)
    vals = rng.sample(range(1, 200), n)
    k = rng.randint(2, min(3, n))
    desc = rng.choice([True, False])
    order = "highest to lowest" if desc else "lowest to highest"
    pairs = sorted(zip(vals, labels), reverse=desc)
    phrasing = rng.choice([
        f"Sort the rows by {nc!r} from {order} and return the first {k} values of {sc!r} as a list.",
        f"Return the {sc!r} of the {k} rows with the {'largest' if desc else 'smallest'} {nc!r}, ordered {order}, as a list.",
    ])
    return dict(
        category="pandas_trap",
        instruction=phrasing,
        setup=_frame({sc: labels, nc: vals}),
        expected=[l for _, l in pairs[:k]],
        solution=f"result = df.sort({nc!r}, descending={desc}).head({k})[{sc!r}].to_list()",
    )


def f_scalar_agg(rng):
    """A single number out."""
    nc = rng.choice(NUM_COLS)
    n = rng.randint(3, 6)
    vals = [rng.randint(1, 100) for _ in range(n)]
    kind, py, expr, cast = rng.choice([
        ("sum", sum, "sum()", "int"),
        ("maximum", max, "max()", "int"),
        ("minimum", min, "min()", "int"),
        ("number of rows", lambda v: len(v), None, "int"),
        ("number of distinct values", lambda v: len(set(v)), "n_unique()", "int"),
    ])
    if kind == "number of rows":
        expected, sol = len(vals), "result = df.height"
    else:
        expected = py(vals)
        sol = f"result = int(df[{nc!r}].{expr})"
    return dict(
        category="output_convention",
        instruction=rng.choice([
            f"Return the {kind} of the {nc!r} column as a plain integer.",
            f"Give back the {kind} of {nc!r}, as a regular Python int.",
        ]),
        setup=_frame({nc: vals}),
        expected=expected,
        solution=sol,
    )


def f_new_column(rng):
    """Compute a derived column."""
    a, b = _pick(rng, NUM_COLS, 2)
    n = rng.randint(3, 5)
    xs = [rng.randint(1, 20) for _ in range(n)]
    ys = [rng.randint(1, 20) for _ in range(n)]
    op, py = rng.choice([("*", lambda p, q: p * q), ("+", lambda p, q: p + q),
                         ("-", lambda p, q: p - q)])
    return dict(
        category="stale_api",
        instruction=rng.choice([
            f"Add a new column called 'result_col' equal to {a!r} {op} {b!r}, and return that column as a list.",
            f"Create a column 'result_col' by computing {a!r} {op} {b!r}, then return it as a plain Python list.",
        ]),
        setup=_frame({a: xs, b: ys}),
        expected=[py(p, q) for p, q in zip(xs, ys)],
        solution=(f"result = df.with_columns((pl.col({a!r}) {op} pl.col({b!r}))"
                  f".alias('result_col'))['result_col'].to_list()"),
    )


def f_fill_nulls(rng):
    nc = rng.choice(NUM_COLS)
    n = rng.randint(4, 6)
    vals = [rng.choice([None, rng.randint(1, 40)]) for _ in range(n)]
    if all(v is None for v in vals):
        vals[0] = 5
    fill = rng.choice([0, -1, 99])
    return dict(
        category="pandas_trap",
        instruction=rng.choice([
            f"Replace every null in the {nc!r} column with {fill}, then return {nc!r} as a list.",
            f"Fill the missing values in {nc!r} with {fill} and give back the column as a plain list.",
        ]),
        setup=_frame({nc: vals}),
        expected=[fill if v is None else v for v in vals],
        solution=f"result = df.with_columns(pl.col({nc!r}).fill_null({fill}))[{nc!r}].to_list()",
    )


def f_string_op(rng):
    sc = rng.choice(STR_COLS)
    n = rng.randint(3, 5)
    words = _pick(rng, NAMES + CITIES, n)
    kind, py, expr, cat = rng.choice([
        ("uppercase", str.upper, "str.to_uppercase()", "pandas_trap"),
        ("length in characters", len, "str.len_chars()", "stale_api"),
    ])
    return dict(
        category=cat,
        instruction=rng.choice([
            f"Convert every value in {sc!r} to its {kind} and return the results as a list.",
            f"Return the {kind} of each value in the {sc!r} column, as a plain list.",
        ]),
        setup=_frame({sc: words}),
        expected=[py(w) for w in words],
        solution=f"result = df.with_columns(pl.col({sc!r}).{expr}.alias('o'))['o'].to_list()",
    )


def f_cum_or_shift(rng):
    nc = rng.choice(NUM_COLS)
    n = rng.randint(3, 6)
    vals = [rng.randint(1, 20) for _ in range(n)]
    kind = rng.choice(["cumulative", "shift"])
    if kind == "cumulative":
        run, tot = [], 0
        for v in vals:
            tot += v
            run.append(tot)
        return dict(
            category="stale_api",
            instruction=f"Return the running cumulative sum of the {nc!r} column as a list.",
            setup=_frame({nc: vals}),
            expected=run,
            solution=f"result = df.with_columns(pl.col({nc!r}).cum_sum().alias('o'))['o'].to_list()",
        )
    fill = rng.choice([0, -1])
    return dict(
        category="stale_api",
        instruction=(f"Shift the {nc!r} column down by one position, filling the first "
                     f"slot with {fill}. Return the result as a list."),
        setup=_frame({nc: vals}),
        expected=[fill] + vals[:-1],
        solution=(f"result = df.with_columns(pl.col({nc!r}).shift(1, fill_value={fill})"
                  f".alias('o'))['o'].to_list()"),
    )


def f_unique(rng):
    nc = rng.choice(GRP_COLS)
    vals = [rng.choice(GROUPS[:4]) for _ in range(rng.randint(4, 7))]
    return dict(
        category="pandas_trap",
        instruction=rng.choice([
            f"Return the distinct values of {nc!r}, sorted alphabetically, as a list.",
            f"Give the unique {nc!r} values in ascending alphabetical order, as a plain list.",
        ]),
        setup=_frame({nc: vals}),
        expected=sorted(set(vals)),
        solution=f"result = sorted(df[{nc!r}].unique().to_list())",
    )


def f_join(rng):
    n1, n2 = rng.randint(3, 4), rng.randint(3, 4)
    ids1 = sorted(rng.sample(range(1, 9), n1))
    ids2 = sorted(rng.sample(range(1, 9), n2))
    right = rng.choice(STR_COLS)
    rvals = _pick(rng, CITIES, n2)
    how = rng.choice(["inner", "anti"])
    if how == "inner":
        keep = [i for i in ids1 if i in ids2]
        expected = [rvals[ids2.index(i)] for i in keep]
        instr = (f"Inner join df and df2 on 'id', sort by 'id' ascending, and return "
                 f"the {right!r} column as a list.")
        sol = f"result = df.join(df2, on='id', how='inner').sort('id')[{right!r}].to_list()"
    else:
        expected = [i for i in ids1 if i not in ids2]
        instr = ("Return the 'id' values present in df but absent from df2, "
                 "sorted ascending, as a list.")
        sol = "result = df.join(df2, on='id', how='anti').sort('id')['id'].to_list()"
    setup = (_frame({"id": ids1}, "df") + "\n" +
             _frame({"id": ids2, right: rvals}, "df2"))
    return dict(category="hard", instruction=instr, setup=setup,
                expected=expected, solution=sol)


def f_when_then(rng):
    nc = rng.choice(NUM_COLS)
    n = rng.randint(3, 6)
    vals = [rng.randint(1, 100) for _ in range(n)]
    thr = rng.choice([10, 25, 50, 75])
    hi, lo = rng.choice([("high", "low"), ("pass", "fail"), ("big", "small")])
    return dict(
        category="hard",
        instruction=(f"Label each value in {nc!r} as {hi!r} when it is {thr} or more, "
                     f"and {lo!r} otherwise. Return the labels as a list."),
        setup=_frame({nc: vals}),
        expected=[hi if v >= thr else lo for v in vals],
        solution=(f"result = df.with_columns(pl.when(pl.col({nc!r}) >= {thr})"
                  f".then(pl.lit({hi!r})).otherwise(pl.lit({lo!r})).alias('o'))['o'].to_list()"),
    )


def f_group_window(rng):
    """Cumulative sum within each group — a window operation."""
    gc, nc = rng.choice(GRP_COLS), rng.choice(NUM_COLS)
    keys = _pick(rng, GROUPS, 2)
    gvals, vals = [], []
    for k in keys:
        for _ in range(rng.randint(2, 3)):
            gvals.append(k)
            vals.append(rng.randint(1, 20))
    run, tot = [], {}
    for g, v in zip(gvals, vals):
        tot[g] = tot.get(g, 0) + v
        run.append(tot[g])
    return dict(
        category="hard",
        instruction=(f"Compute a cumulative sum of {nc!r} within each {gc!r} group. "
                     f"Return the values as a list in the original row order."),
        setup=_frame({gc: gvals, nc: vals}),
        expected=run,
        solution=(f"result = df.with_columns(pl.col({nc!r}).cum_sum().over({gc!r})"
                  f".alias('o'))['o'].to_list()"),
    )


def f_rows_out(rng):
    """Return whole rows, or the schema."""
    sc, nc = rng.choice(STR_COLS), rng.choice(NUM_COLS)
    n = rng.randint(2, 4)
    labels = _pick(rng, NAMES, n)
    vals = [rng.randint(1, 50) for _ in range(n)]
    kind = rng.choice(["rows", "columns", "first_row"])
    if kind == "rows":
        return dict(category="output_convention",
                    instruction="Return every row as a list of dictionaries.",
                    setup=_frame({sc: labels, nc: vals}),
                    expected=[{sc: l, nc: v} for l, v in zip(labels, vals)],
                    solution="result = df.to_dicts()")
    if kind == "columns":
        return dict(category="output_convention",
                    instruction="Return the list of column names.",
                    setup=_frame({sc: labels, nc: vals}),
                    expected=[sc, nc],
                    solution="result = df.columns")
    return dict(category="output_convention",
                instruction="Return the first row as a dictionary.",
                setup=_frame({sc: labels, nc: vals}),
                expected={sc: labels[0], nc: vals[0]},
                solution="result = df.head(1).to_dicts()[0]")


FAMILIES: List[Callable] = [
    f_filter_select, f_group_agg, f_sort_head, f_scalar_agg, f_new_column,
    f_fill_nulls, f_string_op, f_cum_or_shift, f_unique, f_join,
    f_when_then, f_group_window, f_rows_out,
]


# Broadened coverage — see data_gen/generators_v2.py and NOTES O13.
try:
    from data_gen.generators_v2 import EXTRA_FAMILIES
    FAMILIES = FAMILIES + EXTRA_FAMILIES
except ImportError:
    pass


def generate(n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)]
        ex = fam(rng)
        ex["template"] = fam.__name__
        yield ex
