"""Additional template families — broadening operation coverage (NOTES O13).

WHY, and the methodological trap avoided:

F41 showed the training gain was +25pp on operations the training covered and
only +5.6pp on operations it never emitted. The obvious "fix" — write families
for exactly the operations the eval tests — is **benchmark gaming**: the score
would rise and mean nothing.

So these families are chosen by walking the polars API BY AREA (aggregation,
element-wise math, boolean, selection, sorting, windows, strings, lists,
reshaping, nulls, casting), not by reading the eval. Many operations below are
NOT tested by any eval task — that is the evidence the selection was not
eval-driven, and build_sft reports the count.

The real defence remains the held-out test set built from real GitHub polars
code, which does not exist yet (O2). Until it does, treat coverage-driven gains
with suspicion.
"""

import random
from typing import Callable, List

from data_gen.generators import GRP_COLS, NUM_COLS, STR_COLS, _frame, _lit, _pick
from data_gen.generators import CITIES, GROUPS, NAMES


def f_elementwise_math(rng):
    c = rng.choice(NUM_COLS)
    vals = [rng.randint(-40, 40) for _ in range(rng.randint(3, 6))]
    kind, py, expr = rng.choice([
        ("absolute value", abs, "abs()"),
        ("value with any negative replaced by 0", lambda v: max(v, 0), "clip(lower_bound=0)"),
        ("value capped at 10", lambda v: min(v, 10), "clip(upper_bound=10)"),
    ])
    return dict(category="pandas_trap",
                instruction=f"Return the {kind} of each entry in {c!r}, as a list.",
                setup=_frame({c: vals}),
                expected=[py(v) for v in vals],
                solution=f"result = df.with_columns(pl.col({c!r}).{expr}.alias('o'))['o'].to_list()")


def f_round_float(rng):
    c = rng.choice(NUM_COLS)
    vals = [round(rng.uniform(0.5, 90), 4) for _ in range(rng.randint(3, 5))]
    nd = rng.randint(1, 2)
    return dict(category="pandas_trap",
                instruction=f"Round every value in {c!r} to {nd} decimal place{'s' if nd > 1 else ''} and return {c!r} as a list.",
                setup=_frame({c: vals}),
                expected=[round(v, nd) for v in vals],
                solution=f"result = df.with_columns(pl.col({c!r}).round({nd}))[{c!r}].to_list()")


def f_stat_agg(rng):
    c = rng.choice(NUM_COLS)
    vals = [float(rng.randint(1, 60)) for _ in range(rng.randint(4, 7))]
    kind, expr = rng.choice([("mean", "mean"), ("median", "median")])
    srt = sorted(vals)
    if kind == "mean":
        exp = round(sum(vals) / len(vals), 3)
    else:
        n = len(srt)
        exp = round(srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2, 3)
    return dict(category="output_convention",
                instruction=f"Return the {kind} of {c!r} rounded to 3 decimal places, as a plain float.",
                setup=_frame({c: vals}),
                expected=exp,
                solution=f"result = round(float(df[{c!r}].{expr}()), 3)")


def f_boolean_reduce(rng):
    c = rng.choice(NUM_COLS)
    vals = [rng.randint(1, 60) for _ in range(rng.randint(3, 6))]
    thr = rng.randint(10, 50)
    kind = rng.choice(["any", "all"])
    if kind == "any":
        exp, expr, word = any(v > thr for v in vals), "any()", "any value"
    else:
        exp, expr, word = all(v > thr for v in vals), "all()", "every value"
    return dict(category="output_convention",
                instruction=f"Return True if {word} in {c!r} is greater than {thr}, otherwise False, as a plain bool.",
                setup=_frame({c: vals}),
                expected=exp,
                solution=f"result = bool((df[{c!r}] > {thr}).{expr})")


def f_membership(rng):
    c = rng.choice(GRP_COLS)
    pool = _pick(rng, GROUPS, 4)
    vals = [rng.choice(pool) for _ in range(rng.randint(4, 7))]
    want = _pick(rng, pool, 2)
    return dict(category="pandas_trap",
                instruction=f"Keep only the rows where {c!r} is one of {want!r}, and return {c!r} as a list.",
                setup=_frame({c: vals}),
                expected=[v for v in vals if v in want],
                solution=f"result = df.filter(pl.col({c!r}).is_in({want!r}))[{c!r}].to_list()")


def f_range_filter(rng):
    c = rng.choice(NUM_COLS)
    vals = [rng.randint(1, 100) for _ in range(rng.randint(4, 7))]
    lo = rng.randint(10, 40)
    hi = lo + rng.randint(15, 40)
    return dict(category="pandas_trap",
                instruction=f"Keep only the rows where {c!r} lies between {lo} and {hi} inclusive, and return {c!r} as a list.",
                setup=_frame({c: vals}),
                expected=[v for v in vals if lo <= v <= hi],
                solution=f"result = df.filter(pl.col({c!r}).is_between({lo}, {hi}))[{c!r}].to_list()")


def f_position_ops(rng):
    c = rng.choice(NUM_COLS)
    vals = rng.sample(range(1, 200), rng.randint(4, 6))
    kind = rng.choice(["arg_max", "arg_min", "head", "tail", "gather"])
    if kind == "arg_max":
        m = max(vals)
        return dict(category="pandas_trap",
                    instruction=f"Return the row position of the largest value in {c!r}, as a plain integer.",
                    setup=_frame({c: vals}), expected=vals.index(m),
                    solution=f"result = int(df[{c!r}].arg_max())")
    if kind == "arg_min":
        m = min(vals)
        return dict(category="pandas_trap",
                    instruction=f"Return the row position of the smallest value in {c!r}, as a plain integer.",
                    setup=_frame({c: vals}), expected=vals.index(m),
                    solution=f"result = int(df[{c!r}].arg_min())")
    k = rng.randint(2, 3)
    if kind == "head":
        return dict(category="output_convention",
                    instruction=f"Return the first {k} values of {c!r} as a list, in row order.",
                    setup=_frame({c: vals}), expected=vals[:k],
                    solution=f"result = df.head({k})[{c!r}].to_list()")
    if kind == "tail":
        return dict(category="output_convention",
                    instruction=f"Return the last {k} values of {c!r} as a list, in row order.",
                    setup=_frame({c: vals}), expected=vals[-k:],
                    solution=f"result = df.tail({k})[{c!r}].to_list()")
    idx = sorted(rng.sample(range(len(vals)), 2))
    return dict(category="stale_api",
                instruction=f"Return the values of {c!r} at row positions {idx[0]} and {idx[1]}, as a list.",
                setup=_frame({c: vals}), expected=[vals[i] for i in idx],
                solution=f"result = df[{c!r}].gather({idx!r}).to_list()")


def f_rank_family(rng):
    c = rng.choice(NUM_COLS)
    vals = rng.sample(range(1, 90), rng.randint(3, 5))
    kind = rng.choice(["rank_desc", "rank_asc", "arg_sort", "top_k", "bottom_k"])
    srt_desc = sorted(vals, reverse=True)
    srt_asc = sorted(vals)
    if kind == "rank_desc":
        return dict(category="stale_api",
                    instruction=f"Rank the values in {c!r} from highest to lowest, where 1 is the highest. Return the ranks as a list in original row order.",
                    setup=_frame({c: vals}), expected=[srt_desc.index(v) + 1 for v in vals],
                    solution=f"result = df.with_columns(pl.col({c!r}).rank(method='min', descending=True).cast(pl.Int64).alias('r'))['r'].to_list()")
    if kind == "rank_asc":
        return dict(category="stale_api",
                    instruction=f"Rank the values in {c!r} from lowest to highest, where 1 is the lowest. Return the ranks as a list in original row order.",
                    setup=_frame({c: vals}), expected=[srt_asc.index(v) + 1 for v in vals],
                    solution=f"result = df.with_columns(pl.col({c!r}).rank(method='min').cast(pl.Int64).alias('r'))['r'].to_list()")
    if kind == "arg_sort":
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        return dict(category="stale_api",
                    instruction=f"Return the row positions that would sort {c!r} in ascending order, as a list.",
                    setup=_frame({c: vals}), expected=order,
                    solution=f"result = df[{c!r}].arg_sort().cast(pl.Int64).to_list()")
    k = min(2, len(vals))
    if kind == "top_k":
        return dict(category="stale_api",
                    instruction=f"Return the {k} largest values of {c!r}, sorted from highest to lowest, as a list.",
                    setup=_frame({c: vals}), expected=srt_desc[:k],
                    solution=f"result = df[{c!r}].top_k({k}).sort(descending=True).to_list()")
    return dict(category="stale_api",
                instruction=f"Return the {k} smallest values of {c!r}, sorted from lowest to highest, as a list.",
                setup=_frame({c: vals}), expected=srt_asc[:k],
                solution=f"result = df[{c!r}].bottom_k({k}).sort().to_list()")


def f_cumulative_family(rng):
    c = rng.choice(NUM_COLS)
    vals = [rng.randint(1, 12) for _ in range(rng.randint(3, 5))]
    kind = rng.choice(["cum_max", "cum_min", "cum_prod", "diff"])
    out, run = [], None
    if kind == "cum_max":
        for v in vals:
            run = v if run is None else max(run, v)
            out.append(run)
        word = "running maximum"
    elif kind == "cum_min":
        for v in vals:
            run = v if run is None else min(run, v)
            out.append(run)
        word = "running minimum"
    elif kind == "cum_prod":
        run = 1
        for v in vals:
            run *= v
            out.append(run)
        word = "running product"
    else:
        out = [None] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        word = "difference between each value and the previous one (first entry null)"
    return dict(category="stale_api",
                instruction=f"Return the {word} of {c!r} as a list.",
                setup=_frame({c: vals}), expected=out,
                solution=f"result = df.with_columns(pl.col({c!r}).{kind}().alias('o'))['o'].to_list()")


def f_string_family(rng):
    c = rng.choice(STR_COLS)
    words = _pick(rng, NAMES + CITIES, rng.randint(3, 5))
    kind = rng.choice(["lower", "starts", "ends", "slice", "replace", "pad"])
    if kind == "lower":
        return dict(category="pandas_trap",
                    instruction=f"Convert every value in {c!r} to lowercase and return {c!r} as a list.",
                    setup=_frame({c: [w.upper() for w in words]}), expected=words,
                    solution=f"result = df.with_columns(pl.col({c!r}).str.to_lowercase())[{c!r}].to_list()")
    if kind == "starts":
        ch = words[0][0]
        return dict(category="pandas_trap",
                    instruction=f"Keep only rows where {c!r} starts with {ch!r}, and return {c!r} as a list.",
                    setup=_frame({c: words}), expected=[w for w in words if w.startswith(ch)],
                    solution=f"result = df.filter(pl.col({c!r}).str.starts_with({ch!r}))[{c!r}].to_list()")
    if kind == "ends":
        ch = words[0][-1]
        return dict(category="pandas_trap",
                    instruction=f"Keep only rows where {c!r} ends with {ch!r}, and return {c!r} as a list.",
                    setup=_frame({c: words}), expected=[w for w in words if w.endswith(ch)],
                    solution=f"result = df.filter(pl.col({c!r}).str.ends_with({ch!r}))[{c!r}].to_list()")
    if kind == "slice":
        n = rng.randint(1, 2)
        return dict(category="stale_api",
                    instruction=f"Return the first {n} character{'s' if n > 1 else ''} of each value in {c!r}, as a list.",
                    setup=_frame({c: words}), expected=[w[:n] for w in words],
                    solution=f"result = df.with_columns(pl.col({c!r}).str.slice(0, {n}).alias('o'))['o'].to_list()")
    if kind == "replace":
        a, b = "a", "@"
        return dict(category="stale_api",
                    instruction=f"Replace every occurrence of {a!r} with {b!r} in {c!r} and return {c!r} as a list.",
                    setup=_frame({c: words}), expected=[w.replace(a, b) for w in words],
                    solution=f"result = df.with_columns(pl.col({c!r}).str.replace_all({a!r}, {b!r}))[{c!r}].to_list()")
    padded = [f"  {w} " for w in words]
    return dict(category="stale_api",
                instruction=f"Remove leading and trailing whitespace from every value in {c!r} and return {c!r} as a list.",
                setup=_frame({c: padded}), expected=words,
                solution=f"result = df.with_columns(pl.col({c!r}).str.strip_chars())[{c!r}].to_list()")


def f_null_family(rng):
    c = rng.choice(NUM_COLS)
    vals = [rng.choice([None, rng.randint(1, 30)]) for _ in range(rng.randint(4, 6))]
    if all(v is None for v in vals):
        vals[0] = 7
    kind = rng.choice(["drop", "count", "forward"])
    if kind == "drop":
        return dict(category="pandas_trap",
                    instruction=f"Remove the rows where {c!r} is null and return {c!r} as a list.",
                    setup=_frame({c: vals}), expected=[v for v in vals if v is not None],
                    solution=f"result = df.drop_nulls({c!r})[{c!r}].to_list()")
    if kind == "count":
        n = 0
        for v in vals:
            if v is None:
                n += 1
        return dict(category="output_convention",
                    instruction=f"Return how many nulls are in {c!r}, as a plain integer.",
                    setup=_frame({c: vals}), expected=n,
                    solution=f"result = int(df[{c!r}].null_count())")
    if vals[0] is None:
        vals[0] = 3
    out, last = [], None
    for v in vals:
        last = v if v is not None else last
        out.append(last)
    return dict(category="stale_api",
                instruction=f"Fill the nulls in {c!r} by carrying the previous non-null value forward. Return {c!r} as a list.",
                setup=_frame({c: vals}), expected=out,
                solution=f"result = df.with_columns(pl.col({c!r}).fill_null(strategy='forward'))[{c!r}].to_list()")


def f_schema_ops(rng):
    a, b = _pick(rng, NUM_COLS, 2)
    vals = [rng.randint(1, 20) for _ in range(3)]
    other = [rng.randint(1, 20) for _ in range(3)]
    kind = rng.choice(["rename", "row_index", "n_cols", "drop"])
    if kind == "rename":
        return dict(category="pandas_trap",
                    instruction=f"Rename column {a!r} to 'renamed' and return the list of column names.",
                    setup=_frame({a: vals, b: other}), expected=["renamed", b],
                    solution=f"result = df.rename({{{a!r}: 'renamed'}}).columns")
    if kind == "row_index":
        return dict(category="stale_api",
                    instruction="Add a column 'idx' holding each row's position starting at 0, and return 'idx' as a list.",
                    setup=_frame({a: vals}), expected=list(range(len(vals))),
                    solution="result = df.with_row_index('idx')['idx'].cast(pl.Int64).to_list()")
    if kind == "n_cols":
        return dict(category="output_convention",
                    instruction="Return the number of columns as a plain integer.",
                    setup=_frame({a: vals, b: other}), expected=2,
                    solution="result = len(df.columns)")
    return dict(category="pandas_trap",
                instruction=f"Remove the column {b!r} and return the list of remaining column names.",
                setup=_frame({a: vals, b: other}), expected=[a],
                solution=f"result = df.drop({b!r}).columns")


def f_concat_frames(rng):
    c = rng.choice(NUM_COLS)
    a1 = [rng.randint(1, 20) for _ in range(rng.randint(2, 3))]
    a2 = [rng.randint(1, 20) for _ in range(rng.randint(1, 2))]
    return dict(category="hard",
                instruction=f"Stack df on top of df2 into a single table and return {c!r} as a list.",
                setup=_frame({c: a1}, "df") + "\n" + _frame({c: a2}, "df2"),
                expected=a1 + a2,
                solution="result = pl.concat([df, df2])['a'].to_list()".replace("'a'", repr(c)))


def f_join_family(rng):
    n1 = rng.randint(3, 4)
    ids1 = sorted(rng.sample(range(1, 9), n1))
    ids2 = sorted(rng.sample(range(1, 9), rng.randint(2, 4)))
    how = rng.choice(["semi", "left_count", "cross"])
    if how == "semi":
        keep = [i for i in ids1 if i in ids2]
        return dict(category="hard",
                    instruction="Return how many rows of df have an 'id' that also appears in df2, as a plain integer.",
                    setup=_frame({"id": ids1}, "df") + "\n" + _frame({"id": ids2}, "df2"),
                    expected=len(keep),
                    solution="result = df.join(df2, on='id', how='semi').height")
    if how == "cross":
        return dict(category="hard",
                    instruction="Produce every combination of a row from df with a row from df2, and return the number of combinations as a plain integer.",
                    setup=_frame({"id": ids1}, "df") + "\n" + _frame({"code": ids2}, "df2"),
                    expected=len(ids1) * len(ids2),
                    solution="result = df.join(df2, how='cross').height")
    keep = [i for i in ids1 if i not in ids2]
    return dict(category="hard",
                instruction="Return the 'id' values in df that do not appear in df2, sorted ascending, as a list.",
                setup=_frame({"id": ids1}, "df") + "\n" + _frame({"id": ids2}, "df2"),
                expected=keep,
                solution="result = df.join(df2, on='id', how='anti').sort('id')['id'].to_list()")


def f_group_window_family(rng):
    g, c = rng.choice(GRP_COLS), rng.choice(NUM_COLS)
    keys = _pick(rng, GROUPS, 2)
    gv, vv = [], []
    for k in keys:
        for _ in range(rng.randint(2, 3)):
            gv.append(k)
            vv.append(rng.randint(1, 20))
    kind = rng.choice(["cum_max_over", "diff_over", "n_unique", "group_len_filter"])
    if kind == "cum_max_over":
        out, run = [], {}
        for k, v in zip(gv, vv):
            run[k] = v if k not in run else max(run[k], v)
            out.append(run[k])
        return dict(category="hard",
                    instruction=f"Compute the running maximum of {c!r} within each {g!r} group. Return the values as a list in the original row order.",
                    setup=_frame({g: gv, c: vv}), expected=out,
                    solution=f"result = df.with_columns(pl.col({c!r}).cum_max().over({g!r}).alias('o'))['o'].to_list()")
    if kind == "diff_over":
        out, prev = [], {}
        for k, v in zip(gv, vv):
            out.append(None if k not in prev else v - prev[k])
            prev[k] = v
        return dict(category="hard",
                    instruction=f"Within each {g!r} group, return the difference between each {c!r} value and the previous one in that group. The first row of each group is null. Return as a list in row order.",
                    setup=_frame({g: gv, c: vv}), expected=out,
                    solution=f"result = df.with_columns(pl.col({c!r}).diff().over({g!r}).alias('o'))['o'].to_list()")
    if kind == "n_unique":
        seen = {}
        for k, v in zip(gv, vv):
            seen.setdefault(k, set()).add(v)
        return dict(category="hard",
                    instruction=f"For each {g!r} group, count how many distinct {c!r} values it contains. Return a dict from group to that count.",
                    setup=_frame({g: gv, c: vv}),
                    expected={k: len(s) for k, s in seen.items()},
                    solution=(f"x = df.group_by({g!r}).agg(pl.col({c!r}).n_unique().alias('n'))\n"
                              f"result = dict(zip(x[{g!r}].to_list(), x['n'].to_list()))"))
    counts = {}
    for k in gv:
        counts[k] = counts.get(k, 0) + 1
    keep = sorted(k for k, n in counts.items() if n > 2)
    return dict(category="hard",
                instruction=f"Return the {g!r} groups that contain more than 2 rows, sorted alphabetically, as a list.",
                setup=_frame({g: gv, c: vv}), expected=keep,
                solution=(f"x = df.group_by({g!r}).agg(pl.len().alias('n')).filter(pl.col('n') > 2)\n"
                          f"result = sorted(x[{g!r}].to_list())"))


EXTRA_FAMILIES: List[Callable] = [
    f_elementwise_math, f_round_float, f_stat_agg, f_boolean_reduce,
    f_membership, f_range_filter, f_position_ops, f_rank_family,
    f_cumulative_family, f_string_family, f_null_family, f_schema_ops,
    f_concat_frames, f_join_family, f_group_window_family,
]
