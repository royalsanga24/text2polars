"""Prompt variants under test.

The baseline is the BEST simple approach, not the first one tried. If a good
prompt closes the gap, fine-tuning is not justified as a capability fix (it may
still be justified on cost/latency — see NOTES).

Every variant takes {preamble}, {setup}, {instruction}.
"""

_BASE = """\
{preamble}

You are given this data:

{setup}

Task: {instruction}

Write Python code that assigns the answer to a variable named `result`.
{extra}The data is already defined — do not redefine it. Do not print anything.
Reply with ONLY the code, no explanation, no markdown fences.\
"""

# Few-shot examples. Deliberately NOT drawn from the dev set — reusing dev tasks
# here would leak the answers and inflate the score.
_SHOTS = """\
Examples of the expected style:

# data: df = pl.DataFrame({{'city': ['delhi','mumbai'], 'temp': [41, 33]}})
# task: Return the cities where temp is above 35, as a list.
result = df.filter(pl.col('temp') > 35)['city'].to_list()

# data: df = pl.DataFrame({{'dept': ['hr','eng','hr'], 'cost': [10, 40, 5]}})
# task: Group by 'dept' and sum 'cost'. Return a dict mapping dept to total.
g = df.group_by('dept').agg(pl.col('cost').sum())
result = dict(zip(g['dept'].to_list(), g['cost'].to_list()))

# data: df = pl.DataFrame({{'n': [2, 8, 5]}})
# task: Return the largest value in 'n' as a plain integer.
result = int(df['n'].max())

Now your turn.

"""

# A cheatsheet a real engineer would write after reading the failures.
# NOTE: it was derived FROM dev-set failures, so its dev score is optimistic.
# See NOTES.md — this is prompt-overfitting, the same disease as training on
# your test set, and it must be checked against the held-out set.
_CHEATSHEET = """\
polars is NOT pandas. The most common mistakes:

  pandas (wrong here)        polars (correct)
  df.sort_values('c')        df.sort('c')
  df.groupby('c')            df.group_by('c')
  s.fillna(0)                s.fill_null(0)
  s.tolist()                 s.to_list()
  s.apply(f)                 s.map_elements(f, return_dtype=pl.Int64)
  df.merge(o, on='k')        df.join(o, on='k')
  df.drop_duplicates()       df.unique()
  s.astype(int)              s.cast(pl.Int64)
  df.rename(columns={{...}})   df.rename({{...}})
  s.cumsum()                 s.cum_sum()
  s.str.len()                s.str.len_chars()
  pl.count()                 pl.len()
  .rank(desc=True)           .rank(descending=True)
  df.to_dict('records')      df.to_dicts()
  s.take([0, 2])             s.gather([0, 2])
  df.iloc[0]                 df.row(0) / df['c'][0]

`result` must be a plain Python object — list, dict, int, float, str or bool.
Never a DataFrame or Series. Convert it.

"""

PROMPTS = {
    # v1 — the naive prompt. Our starting baseline.
    "v1": _BASE.replace("{extra}", ""),

    # v2 — states the output convention. Tested: no significant effect (NOTES F9).
    "v2": _BASE.replace("{extra}", (
        "\n`result` must be a plain Python object — a list, dict, int, float, str "
        "or bool.\nIt must NOT be a DataFrame, a Series, or any other library "
        "object. Convert it.\n\n")),

    # v3 — three worked examples, teaching the answer style by demonstration.
    "v3": _SHOTS + _BASE.replace("{extra}", ""),

    # v4 — examples plus the pandas->polars cheatsheet. The strongest prompt
    #      a competent engineer would reach for without training anything.
    "v4": _CHEATSHEET + _SHOTS + _BASE.replace("{extra}", ""),
}
