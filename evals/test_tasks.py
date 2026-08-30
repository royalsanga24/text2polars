"""HELD-OUT TEST SET — the honest measurement (NOTES O2).

Every task here uses polars operations that are:
  - among the 80 most frequently used in real, third-party polars code
    (measured from 196 MIT-licensed files in pola-rs/polars — user guide and
    test suite; see data/real/operation_stats.json), AND
  - never emitted by our training generators, AND
  - never tested by our dev set.

The operation list was fixed from real-world frequency BEFORE any score was
looked at. Choosing it afterwards would mean picking the questions the model
happens to fail, which measures nothing.

**RULE: do not add these operations to the SFT training generators.** The
moment SFT trains on them this set stops being held out for SFT.

**IMPORTANT — what this set means changes after CPT.**
100% of the operations tested here appear in the CPT corpus, because they were
*selected from* that corpus by real-world frequency. That was correct for
measuring SFT transfer (and it did: F50, base 43.3% vs SFT 40.0%, p = 1.000 —
a permanent, banked result). But once CPT trains on the corpus, this stops
being a held-out set and becomes a direct test of "did CPT teach these common
operations that synthetic SFT could not?" — which is the question F51 asks.

Report it accordingly after any CPT run. A genuine post-CPT transfer set would
have to be built from operations absent from the CPT corpus (97 exist, but they
are the obscure tail — trig, bitwise — where the base model scores near zero,
so the measurement would carry little information). Parked, not forgotten.
"""

CATEGORIES = {"held_out": "Real-world operations never seen in training"}
PREAMBLE = "import polars as pl"


def _t(tid, instruction, setup, expected, solution):
    return dict(id=tid, category="held_out", instruction=instruction,
                setup=setup, expected=expected, solution=solution)


TASKS = [
    _t("ho_to_dict", "Return the table as a dict mapping each column name to a list of its values.",
       "df = pl.DataFrame({'a': [1, 2], 'b': ['x', 'y']})", {"a": [1, 2], "b": ["x", "y"]},
       "result = {k: v.to_list() for k, v in df.to_dict().items()}"),
    _t("ho_rows", "Return every row as a list of lists, in row order.",
       "df = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})", [[1, 3], [2, 4]],
       "result = [list(r) for r in df.rows()]"),
    _t("ho_row_single", "Return row index 1 as a list of its values.",
       "df = pl.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})", [2, "y"],
       "result = list(df.row(1))"),
    _t("ho_to_series", "The table has a single column. Return it as a plain Python list.",
       "df = pl.DataFrame({'only': [4, 5, 6]})", [4, 5, 6],
       "result = df.to_series().to_list()"),
    _t("ho_get_column", "Return the column named 'b' as a plain Python list.",
       "df = pl.DataFrame({'a': [1, 2], 'b': [9, 8]})", [9, 8],
       "result = df.get_column('b').to_list()"),
    _t("ho_dtype_name", "Return the name of the data type of column 'a', as a string.",
       "df = pl.DataFrame({'a': [1, 2]})", "Int64",
       "result = str(df['a'].dtype)"),
    _t("ho_equals", "Return True if df and df2 hold exactly the same data, otherwise False.",
       "df = pl.DataFrame({'a': [1, 2]})\ndf2 = pl.DataFrame({'a': [1, 2]})", True,
       "result = df.equals(df2)"),
    _t("ho_equals_false", "Return True if df and df2 hold exactly the same data, otherwise False.",
       "df = pl.DataFrame({'a': [1, 2]})\ndf2 = pl.DataFrame({'a': [1, 3]})", False,
       "result = df.equals(df2)"),
    _t("ho_is_sorted", "Return True if the values in 'v' are already in ascending order, otherwise False.",
       "df = pl.DataFrame({'v': [1, 3, 7]})", True,
       "result = df['v'].is_sorted()"),
    _t("ho_is_sorted_false", "Return True if the values in 'v' are already in ascending order, otherwise False.",
       "df = pl.DataFrame({'v': [5, 2, 9]})", False,
       "result = df['v'].is_sorted()"),
    _t("ho_reverse", "Return the values of 'v' in reverse row order, as a list.",
       "df = pl.DataFrame({'v': [1, 2, 3]})", [3, 2, 1],
       "result = df['v'].reverse().to_list()"),
    _t("ho_last_value", "Return the final value of 'v' as a plain integer.",
       "df = pl.DataFrame({'v': [7, 8, 9]})", 9,
       "result = int(df['v'].last())"),
    _t("ho_first_value_expr", "Return the first value of 'v' as a plain integer, using a polars expression.",
       "df = pl.DataFrame({'v': [7, 8, 9]})", 7,
       "result = int(df.select(pl.col('v').first()).item())"),
    _t("ho_exclude", "Return the names of every column except 'b', as a list.",
       "df = pl.DataFrame({'a': [1], 'b': [2], 'c': [3]})", ["a", "c"],
       "result = df.select(pl.exclude('b')).columns"),
    _t("ho_sort_by", "Return the values of 'name' ordered by the corresponding 'score' ascending, as a list.",
       "df = pl.DataFrame({'name': ['p', 'q', 'r'], 'score': [30, 10, 20]})", ["q", "r", "p"],
       "result = df.select(pl.col('name').sort_by('score'))['name'].to_list()"),
    _t("ho_is_null_mask", "Return a list of booleans saying, for each row, whether 'v' is null.",
       "df = pl.DataFrame({'v': [1, None, 3]})", [False, True, False],
       "result = df['v'].is_null().to_list()"),
    _t("ho_count_non_null", "Return how many non-null values are in 'v', as a plain integer.",
       "df = pl.DataFrame({'v': [1, None, 3, None]})", 2,
       "result = int(df.select(pl.col('v').count()).item())"),
    _t("ho_vstack", "Stack df2 underneath df and return column 'a' as a list.",
       "df = pl.DataFrame({'a': [1, 2]})\ndf2 = pl.DataFrame({'a': [3]})", [1, 2, 3],
       "result = df.vstack(df2)['a'].to_list()"),
    _t("ho_extend_series", "Append the values of s2 onto s1 and return the combined values as a list.",
       "s1 = pl.Series('x', [1, 2])\ns2 = pl.Series('x', [3, 4])", [1, 2, 3, 4],
       "result = s1.append(s2).to_list()"),
    _t("ho_to_frame_columns", "Turn the Series `s` into a one-column table and return the list of its column names.",
       "s = pl.Series('speed', [1, 2, 3])", ["speed"],
       "result = s.to_frame().columns"),
    _t("ho_struct_unnest", "Combine columns 'a' and 'b' into a single struct column named 'pair', then expand it back into separate columns. Return the resulting column names as a list.",
       "df = pl.DataFrame({'a': [1], 'b': [2]})", ["a", "b"],
       "result = df.select(pl.struct(['a', 'b']).alias('pair')).unnest('pair').columns"),
    _t("ho_implode", "Collapse the whole 'v' column into a single list, and return that list.",
       "df = pl.DataFrame({'v': [1, 2, 3]})", [1, 2, 3],
       "result = df.select(pl.col('v').implode()).item().to_list()"),
    _t("ho_repeat_by", "Repeat each value of 'v' as many times as the matching entry in 'n' says, returning a list of lists.",
       "df = pl.DataFrame({'v': ['a', 'b'], 'n': [2, 3]})", [["a", "a"], ["b", "b", "b"]],
       "result = df.select(pl.col('v').repeat_by('n').alias('o'))['o'].to_list()"),
    # `get` appears in real code as the LIST-namespace accessor, not a Series
    # method — corrected after validate.py rejected the first version.
    _t("ho_list_get", "Column 'xs' holds lists. Return the element at index 1 of each list, as a list.",
       "df = pl.DataFrame({'xs': [[10, 20, 30], [40, 50, 60]]})", [20, 50],
       "result = df.with_columns(pl.col('xs').list.get(1).alias('o'))['o'].to_list()"),
    _t("ho_schema_names", "Return the column names as reported by the table's schema, as a list.",
       "df = pl.DataFrame({'p': [1], 'q': ['z']})", ["p", "q"],
       "result = list(df.schema.keys())"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # _t("ho_str_truncate", "Keep only the first 3 characters of each value in 's', and return the results as a list.",
    # "df = pl.DataFrame({'s': ['abcdef', 'xyz']})", ["abc", "xyz"],
    # "result = df.with_columns(pl.col('s').str.head(3).alias('o'))['o'].to_list()"),
    _t("ho_reshape", "Reshape the 6 values of 'v' into 3 rows of 2, returning a list of lists.",
       "df = pl.DataFrame({'v': [1, 2, 3, 4, 5, 6]})", [[1, 2], [3, 4], [5, 6]],
       "result = [list(x) for x in df['v'].reshape((3, 2)).to_list()]"),
    _t("ho_fold_sum", "Add columns 'a', 'b' and 'c' together row by row and return the totals as a list.",
       "df = pl.DataFrame({'a': [1, 2], 'b': [10, 20], 'c': [100, 200]})", [111, 222],
       "result = df.select(pl.fold(acc=pl.lit(0), function=lambda a, b: a + b, exprs=pl.col(['a','b','c'])).alias('o'))['o'].to_list()"),
    _t("ho_sum_horizontal", "Add columns 'a' and 'b' together row by row and return the totals as a list.",
       "df = pl.DataFrame({'a': [1, 2], 'b': [5, 6]})", [6, 8],
       "result = df.select(pl.sum_horizontal('a', 'b').alias('o'))['o'].to_list()"),
    _t("ho_item_scalar", "The table has one row and one column. Return that single value as a plain integer.",
       "df = pl.DataFrame({'only': [42]})", 42,
       "result = int(df.item())"),
]


# Tasks 31-100, added because n=30 could not resolve the CPT-vs-SFT difference
# (NOTES F59 / O15). Same construction rules.
from evals.test_tasks_extra import EXTRA_TASKS  # noqa: E402

TASKS = TASKS + EXTRA_TASKS
