"""Held-out test tasks 31-100 (NOTES O15).

Same construction rules as evals/test_tasks.py:
  - operations among the 150 most used in real third-party polars code
    (752 MIT files, 32,701 operation uses, 313 distinct)
  - never emitted by our SFT training generators
  - never tested by the dev set
  - the operation list was fixed from frequency BEFORE any score was seen

Grown from 30 to 100 because at n=30 the CPT-vs-SFT difference (+13.3pp) sat at
p = 0.388 — the most interesting result in the project was below the resolution
of the instrument built to measure it (F59).
"""


def _t(tid, instruction, setup, expected, solution):
    return dict(id=tid, category="held_out", instruction=instruction,
                setup=setup, expected=expected, solution=solution)


EXTRA_TASKS = [
    # --- statistics -------------------------------------------------------
    _t("h2_quantile", "Return the value at the 50th percentile of 'v', as a plain float.",
       "df = pl.DataFrame({'v': [1.0, 2.0, 3.0, 4.0, 5.0]})", 3.0,
       "result = float(df['v'].quantile(0.5))"),
    _t("h2_quantile_25", "Return the value at the 25th percentile of 'v', as a plain float.",
       "df = pl.DataFrame({'v': [1.0, 2.0, 3.0, 4.0, 5.0]})", 2.0,
       "result = float(df['v'].quantile(0.25))"),
    _t("h2_variance", "Return the variance of 'v' rounded to 3 decimal places, as a plain float.",
       "df = pl.DataFrame({'v': [2.0, 4.0, 6.0]})", 4.0,
       "result = round(float(df['v'].var()), 3)"),
    _t("h2_product", "Return the product of every value in 'v', as a plain integer.",
       "df = pl.DataFrame({'v': [2, 3, 4]})", 24,
       "result = int(df.select(pl.col('v').product()).item())"),
    _t("h2_sqrt", "Return the square root of each value in 'v', as a list.",
       "df = pl.DataFrame({'v': [4.0, 9.0, 16.0]})", [2.0, 3.0, 4.0],
       "result = df.with_columns(pl.col('v').sqrt().alias('o'))['o'].to_list()"),
    _t("h2_corr", "Return the correlation between 'a' and 'b' rounded to 2 decimal places, as a plain float.",
       "df = pl.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 4.0, 6.0]})", 1.0,
       "result = round(float(df.select(pl.corr('a', 'b')).item()), 2)"),
    _t("h2_dot", "Return the dot product of columns 'a' and 'b', as a plain integer.",
       "df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})", 32,
       "result = int(df.select(pl.col('a').dot(pl.col('b'))).item())"),

    # --- horizontal / multi-column ----------------------------------------
    _t("h2_min_horizontal", "For each row return the smaller of 'a' and 'b', as a list.",
       "df = pl.DataFrame({'a': [1, 8], 'b': [5, 3]})", [1, 3],
       "result = df.select(pl.min_horizontal('a', 'b').alias('o'))['o'].to_list()"),
    _t("h2_max_horizontal", "For each row return the larger of 'a' and 'b', as a list.",
       "df = pl.DataFrame({'a': [1, 8], 'b': [5, 3]})", [5, 8],
       "result = df.select(pl.max_horizontal('a', 'b').alias('o'))['o'].to_list()"),

    # --- nulls and NaNs ----------------------------------------------------
    _t("h2_is_not_null", "Return a list of booleans saying, for each row, whether 'v' is NOT null.",
       "df = pl.DataFrame({'v': [1, None, 3]})", [True, False, True],
       "result = df['v'].is_not_null().to_list()"),
    _t("h2_has_nulls", "Return True if column 'v' contains any null, otherwise False.",
       "df = pl.DataFrame({'v': [1, None, 3]})", True,
       "result = df['v'].has_nulls()"),
    _t("h2_has_nulls_false", "Return True if column 'v' contains any null, otherwise False.",
       "df = pl.DataFrame({'v': [1, 2, 3]})", False,
       "result = df['v'].has_nulls()"),
    _t("h2_fill_nan", "Replace any NaN in 'v' with 0.0 and return 'v' as a list.",
       "df = pl.DataFrame({'v': [1.0, float('nan'), 3.0]})", [1.0, 0.0, 3.0],
       "result = df.with_columns(pl.col('v').fill_nan(0.0))['v'].to_list()"),
    _t("h2_drop_nans", "Remove rows where 'v' is NaN and return 'v' as a list.",
       "df = pl.DataFrame({'v': [1.0, float('nan'), 3.0]})", [1.0, 3.0],
       "result = df.drop_nans('v')['v'].to_list()"),
    _t("h2_interpolate", "Fill the null in 'v' by linear interpolation between its neighbours. Return 'v' as a list.",
       "df = pl.DataFrame({'v': [1.0, None, 3.0]})", [1.0, 2.0, 3.0],
       "result = df.with_columns(pl.col('v').interpolate())['v'].to_list()"),

    # --- uniqueness --------------------------------------------------------
    _t("h2_is_unique", "Return a list of booleans saying, for each row, whether its 'v' value occurs exactly once.",
       "df = pl.DataFrame({'v': [1, 2, 1]})", [False, True, False],
       "result = df['v'].is_unique().to_list()"),
    _t("h2_value_counts_dict", "Count how many times each value appears in 'c' and return a dict from value to count.",
       "df = pl.DataFrame({'c': ['a', 'b', 'a']})", {"a": 2, "b": 1},
       "vc = df['c'].value_counts()\nresult = dict(zip(vc['c'].to_list(), vc['count'].to_list()))"),

    # --- shape and structure -----------------------------------------------
    _t("h2_is_empty", "Return True if the table has no rows, otherwise False.",
       "df = pl.DataFrame({'a': []})", True,
       "result = df.is_empty()"),
    _t("h2_is_empty_false", "Return True if the table has no rows, otherwise False.",
       "df = pl.DataFrame({'a': [1]})", False,
       "result = df.is_empty()"),
    _t("h2_clear_height", "Produce an empty copy of the table with the same columns, and return its row count as a plain integer.",
       "df = pl.DataFrame({'a': [1, 2], 'b': ['x', 'y']})", 0,
       "result = df.clear().height"),
    _t("h2_clone_equal", "Make an independent copy of the table and return True if it holds the same data as the original.",
       "df = pl.DataFrame({'a': [1, 2]})", True,
       "result = df.clone().equals(df)"),
    _t("h2_dtypes_count", "Return how many columns hold 64-bit integers, as a plain integer.",
       "df = pl.DataFrame({'a': [1], 'b': ['x'], 'c': [2]})", 2,
       "result = sum(1 for d in df.dtypes if d == pl.Int64)"),
    _t("h2_transpose_shape", "Transpose the table and return its shape as a two-element list [rows, columns].",
       "df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})", [2, 3],
       "t = df.transpose()\nresult = [t.height, t.width]"),
    _t("h2_to_struct", "Combine columns 'a' and 'b' into one struct column and return the number of columns in the result, as a plain integer.",
       "df = pl.DataFrame({'a': [1], 'b': [2]})", 1,
       "result = df.select(pl.struct(['a', 'b']).alias('s')).width"),

    # --- row selection -----------------------------------------------------
    _t("h2_limit", "Return the first 2 values of 'v' as a list, using the row-limiting operation.",
       "df = pl.DataFrame({'v': [10, 20, 30, 40]})", [10, 20],
       "result = df.limit(2)['v'].to_list()"),
    _t("h2_gather_every", "Return every 2nd value of 'v' starting from the first, as a list.",
       "df = pl.DataFrame({'v': [1, 2, 3, 4, 5, 6]})", [1, 3, 5],
       "result = df.gather_every(2)['v'].to_list()"),
    _t("h2_index_of", "Return the row position of the first occurrence of the value 30 in 'v', as a plain integer.",
       "df = pl.DataFrame({'v': [10, 20, 30, 40]})", 2,
       "result = int(df['v'].index_of(30))"),
    _t("h2_search_sorted", "Column 'v' is already sorted ascending. Return the position where the value 25 would be inserted to keep it sorted, as a plain integer.",
       "df = pl.DataFrame({'v': [10, 20, 30, 40]})", 2,
       "result = int(df['v'].search_sorted(25))"),
    _t("h2_remove_rows", "Remove the rows where 'v' is greater than 5 and return 'v' as a list.",
       "df = pl.DataFrame({'v': [3, 8, 4, 9]})", [3, 4],
       "result = df.remove(pl.col('v') > 5)['v'].to_list()"),
    _t("h2_new_from_index", "Return a table made of 3 copies of row index 1, and give back its 'v' column as a list.",
       # new_from_index lives on Series, not DataFrame — corrected after
       # validate.py rejected the first version.
       "df = pl.DataFrame({'v': [10, 20, 30]})", [20, 20, 20],
       "result = df['v'].new_from_index(1, 3).to_list()"),

    # --- combining tables --------------------------------------------------
    _t("h2_extend", "Append the rows of df2 onto df in place and return 'a' as a list.",
       "df = pl.DataFrame({'a': [1, 2]})\ndf2 = pl.DataFrame({'a': [3, 4]})", [1, 2, 3, 4],
       "result = df.extend(df2)['a'].to_list()"),
    _t("h2_merge_sorted", "Both tables are sorted by 'v'. Merge them keeping the order, and return 'v' as a list.",
       "df = pl.DataFrame({'v': [1, 5, 9]})\ndf2 = pl.DataFrame({'v': [2, 6]})", [1, 2, 5, 6, 9],
       "result = df.merge_sorted(df2, key='v')['v'].to_list()"),
    _t("h2_update", "Update df with the matching rows from df2 on key 'id', and return 'v' as a list ordered by 'id'.",
       "df = pl.DataFrame({'id': [1, 2, 3], 'v': [10, 20, 30]})\n"
       "df2 = pl.DataFrame({'id': [2], 'v': [99]})", [10, 99, 30],
       "result = df.update(df2, on='id').sort('id')['v'].to_list()"),
    _t("h2_unpivot", "Turn columns 'a' and 'b' into rows (long format) and return the number of resulting rows, as a plain integer.",
       "df = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})", 4,
       "result = df.unpivot(['a', 'b']).height"),

    # --- value transforms --------------------------------------------------
    _t("h2_replace_strict", "Map every value in 'c' using the mapping a->1 and b->2. Return the results as a list.",
       "df = pl.DataFrame({'c': ['a', 'b', 'a']})", [1, 2, 1],
       "result = df.with_columns(pl.col('c').replace_strict({'a': 1, 'b': 2}).alias('o'))['o'].to_list()"),
    _t("h2_cut_labels", "Split the values of 'v' at the boundary 5 into two labelled bands, 'low' and 'high'. Return the labels as a list.",
       "df = pl.DataFrame({'v': [1, 7, 3]})", ["low", "high", "low"],
       "result = df.with_columns(pl.col('v').cut([5], labels=['low', 'high']).alias('o'))['o'].cast(pl.String).to_list()"),
    _t("h2_ewm_mean", "Return the exponentially weighted moving mean of 'v' with alpha 0.5, rounded to 3 decimals, as a list.",
       "df = pl.DataFrame({'v': [1.0, 2.0, 3.0]})", [1.0, 1.667, 2.429],
       "result = df.with_columns(pl.col('v').ewm_mean(alpha=0.5).round(3).alias('o'))['o'].to_list()"),
    _t("h2_log", "Return the natural logarithm of each value in 'v', rounded to 3 decimals, as a list.",
       "df = pl.DataFrame({'v': [1.0, 2.718281828]})", [0.0, 1.0],
       "result = df.with_columns(pl.col('v').log().round(3).alias('o'))['o'].to_list()"),
    _t("h2_scatter", "Set the value at row index 1 of 'v' to 99 and return 'v' as a list.",
       "df = pl.DataFrame({'v': [10, 20, 30]})", [10, 99, 30],
       "s = df['v']\ns[1] = 99\nresult = s.to_list()"),
    _t("h2_to_physical", "Return the underlying integer codes of the categorical column 'c', as a list.",
       "df = pl.DataFrame({'c': ['a', 'b', 'a']}, schema={'c': pl.Categorical})", [0, 1, 0],
       "result = df.with_columns(pl.col('c').to_physical().cast(pl.Int64).alias('o'))['o'].to_list()"),

    # --- comparison with nulls --------------------------------------------
    _t("h2_eq_missing", "Compare 'a' and 'b' element by element, treating two nulls as equal. Return the booleans as a list.",
       "df = pl.DataFrame({'a': [1, None], 'b': [1, None]})", [True, True],
       "result = df.select(pl.col('a').eq_missing(pl.col('b')).alias('o'))['o'].to_list()"),
    _t("h2_ne_missing", "Compare 'a' and 'b' element by element for INEQUALITY, treating two nulls as equal. Return the booleans as a list.",
       "df = pl.DataFrame({'a': [1, None], 'b': [2, None]})", [True, False],
       "result = df.select(pl.col('a').ne_missing(pl.col('b')).alias('o'))['o'].to_list()"),

    # --- iteration and schema ---------------------------------------------
    _t("h2_iter_rows", "Iterate the rows and return the total of column 'v', as a plain integer.",
       "df = pl.DataFrame({'v': [1, 2, 3], 'w': ['a', 'b', 'c']})", 6,
       "result = sum(r[0] for r in df.iter_rows())"),
    _t("h2_collect_schema_names", "Return the column names taken from the table's schema, as a list.",
       "df = pl.DataFrame({'p': [1], 'q': [2]})", ["p", "q"],
       "result = list(df.collect_schema().names())"),
    _t("h2_set_sorted_max", "Column 'v' is known to be sorted ascending. Mark it as sorted, then return its largest value as a plain integer.",
       "df = pl.DataFrame({'v': [1, 4, 9]})", 9,
       "result = int(df['v'].set_sorted().max())"),

    # --- lazy --------------------------------------------------------------
    _t("h2_lazy_filter", "Using the lazy API, keep rows where 'v' exceeds 2 and return 'v' as a list.",
       "df = pl.DataFrame({'v': [1, 3, 5]})", [3, 5],
       "result = df.lazy().filter(pl.col('v') > 2).collect()['v'].to_list()"),
    _t("h2_lazy_select", "Using the lazy API, select only column 'a' and return the resulting column names as a list.",
       "df = pl.DataFrame({'a': [1], 'b': [2]})", ["a"],
       "result = df.lazy().select('a').collect().columns"),

    # --- rolling / windows -------------------------------------------------
    _t("h2_rolling_sum", "Return the rolling sum of 'v' over a window of 2 rows. The first entry should be null.",
       "df = pl.DataFrame({'v': [1, 2, 3, 4]})", [None, 3, 5, 7],
       "result = df.with_columns(pl.col('v').rolling_sum(window_size=2).alias('o'))['o'].to_list()"),
    _t("h2_rolling_min", "Return the rolling minimum of 'v' over a window of 2 rows. The first entry should be null.",
       "df = pl.DataFrame({'v': [5, 2, 8, 1]})", [None, 2, 2, 1],
       "result = df.with_columns(pl.col('v').rolling_min(window_size=2).alias('o'))['o'].to_list()"),

    # --- string operations not used in training ----------------------------
    _t("h2_str_zfill", "Pad each value in 's' on the left with zeros to a total width of 4. Return as a list.",
       "df = pl.DataFrame({'s': ['7', '42']})", ["0007", "0042"],
       "result = df.with_columns(pl.col('s').str.zfill(4).alias('o'))['o'].to_list()"),
    _t("h2_str_count_matches", "Count how many times the letter 'a' appears in each value of 's'. Return as a list.",
       "df = pl.DataFrame({'s': ['banana', 'kiwi']})", [3, 0],
       "result = df.with_columns(pl.col('s').str.count_matches('a').alias('o'))['o'].to_list()"),
    _t("h2_str_reverse", "Reverse each string in 's' and return the results as a list.",
       "df = pl.DataFrame({'s': ['abc', 'xy']})", ["cba", "yx"],
       "result = df.with_columns(pl.col('s').str.reverse().alias('o'))['o'].to_list()"),
    _t("h2_str_to_titlecase", "Convert each value in 's' to title case and return as a list.",
       "df = pl.DataFrame({'s': ['hello world']})", ["Hello World"],
       "result = df.with_columns(pl.col('s').str.to_titlecase().alias('o'))['o'].to_list()"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # # DROPPED (O15): its non-scaffold operations all appear in training
    # # _t("h2_str_tail", "Return the last 2 characters of each value in 's', as a list.",
    # # "df = pl.DataFrame({'s': ['abcdef', 'xy']})", ["ef", "xy"],
    # # "result = df.with_columns(pl.col('s').str.tail(2).alias('o'))['o'].to_list()"),
    # # --- list namespace ----------------------------------------------------
    # _t("h2_list_max", "Column 'xs' holds lists. Return the largest element of each list, as a list.",
    # "df = pl.DataFrame({'xs': [[3, 9, 1], [4, 2]]})", [9, 4],
    # "result = df.with_columns(pl.col('xs').list.max().alias('o'))['o'].to_list()"),
    _t("h2_list_contains", "Column 'xs' holds lists. Return a list of booleans saying whether each list contains the value 2.",
       "df = pl.DataFrame({'xs': [[1, 2], [3, 4]]})", [True, False],
       "result = df.with_columns(pl.col('xs').list.contains(2).alias('o'))['o'].to_list()"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # # DROPPED (O15): its non-scaffold operations all appear in training
    # # # DROPPED (O15): its non-scaffold operations all appear in training
    # # # _t("h2_list_sort", "Column 'xs' holds lists. Sort each list ascending and return the result as a list of lists.",
    # # # "df = pl.DataFrame({'xs': [[3, 1, 2], [9, 4]]})", [[1, 2, 3], [4, 9]],
    # # # "result = df.with_columns(pl.col('xs').list.sort().alias('o'))['o'].to_list()"),
    # # # DROPPED (O15): its non-scaffold operations all appear in training
    # # # _t("h2_list_unique_len", "Column 'xs' holds lists. Return how many distinct elements each list has, as a list.",
    # # # "df = pl.DataFrame({'xs': [[1, 1, 2], [3, 3, 3]]})", [2, 1],
    # # # "result = df.with_columns(pl.col('xs').list.unique().list.len().alias('o'))['o'].to_list()"),
    # # _t("h2_list_join", "Column 'xs' holds lists of strings. Join each list with '-' and return the strings as a list.",
    # # "df = pl.DataFrame({'xs': [['a', 'b'], ['c']]})", ["a-b", "c"],
    # # "result = df.with_columns(pl.col('xs').list.join('-').alias('o'))['o'].to_list()"),
    # # --- aggregation shapes not in training --------------------------------
    # _t("h2_agg_list", "Group by 'g' and collect the 'v' values of each group into a list. Return a dict from group to list.",
    # "df = pl.DataFrame({'g': ['a', 'b', 'a'], 'v': [1, 2, 3]})", {"a": [1, 3], "b": [2]},
    # "x = df.group_by('g', maintain_order=True).agg(pl.col('v'))\n"
    # "result = dict(zip(x['g'].to_list(), x['v'].to_list()))"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # _t("h2_agg_median", "Group by 'g' and compute the median of 'v' per group. Return a dict from group to median.",
    # "df = pl.DataFrame({'g': ['a', 'a', 'a', 'b'], 'v': [1.0, 3.0, 5.0, 8.0]})",
    # {"a": 3.0, "b": 8.0},
    # "x = df.group_by('g').agg(pl.col('v').median())\n"
    # "result = dict(zip(x['g'].to_list(), x['v'].to_list()))"),
    _t("h2_agg_quantile", "Group by 'g' and compute the 50th percentile of 'v' per group. Return a dict from group to that value.",
       "df = pl.DataFrame({'g': ['a', 'a', 'b'], 'v': [1.0, 3.0, 7.0]})", {"a": 3.0, "b": 7.0},
       "x = df.group_by('g').agg(pl.col('v').quantile(0.5))\n"
       "result = dict(zip(x['g'].to_list(), x['v'].to_list()))"),
    _t("h2_agg_first_last", "Group by 'g' and return a dict from group to a two-element list [first v, last v] in row order.",
       "df = pl.DataFrame({'g': ['a', 'a', 'b'], 'v': [1, 2, 9]})", {"a": [1, 2], "b": [9, 9]},
       "x = df.group_by('g', maintain_order=True).agg([pl.col('v').first().alias('f'), pl.col('v').last().alias('l')])\n"
       "result = {g: [f, l] for g, f, l in zip(x['g'].to_list(), x['f'].to_list(), x['l'].to_list())}"),

    # DROPPED (O15): its non-scaffold operations all appear in training
    # # --- misc frequently used ----------------------------------------------
    # _t("h2_sort_nulls_last", "Sort by 'v' ascending with nulls placed last, and return 'v' as a list.",
    # "df = pl.DataFrame({'v': [3, None, 1]})", [1, 3, None],
    # "result = df.sort('v', nulls_last=True)['v'].to_list()"),
    _t("h2_cast_string", "Convert column 'n' to strings and return it as a list.",
       "df = pl.DataFrame({'n': [1, 22]})", ["1", "22"],
       "result = df.with_columns(pl.col('n').cast(pl.String))['n'].to_list()"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # _t("h2_null_count_frame", "Return a dict mapping each column name to how many nulls it contains.",
    # "df = pl.DataFrame({'a': [1, None], 'b': [None, None]})", {"a": 1, "b": 2},
    # "nc = df.null_count()\nresult = {c: int(nc[c][0]) for c in df.columns}"),
    _t("h2_estimated_rows", "Return the number of rows using the table's shape attribute, as a plain integer.",
       "df = pl.DataFrame({'a': [1, 2, 3]})", 3,
       "result = df.shape[0]"),
    _t("h2_select_all_exclude", "Return the names of all columns except 'drop_me', as a list.",
       "df = pl.DataFrame({'keep1': [1], 'drop_me': [2], 'keep2': [3]})", ["keep1", "keep2"],
       "result = df.select(pl.all().exclude('drop_me')).columns"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # _t("h2_when_otherwise_null", "Where 'v' is below 5 return null, otherwise keep the value. Return 'v' as a list.",
    # "df = pl.DataFrame({'v': [3, 7, 1]})", [None, 7, None],
    # "result = df.with_columns(pl.when(pl.col('v') < 5).then(None).otherwise(pl.col('v')).alias('v'))['v'].to_list()"),
    # DROPPED (O15): its non-scaffold operations all appear in training
    # _t("h2_group_by_len_only", "Return a dict mapping each 'g' value to how many rows it has.",
    # "df = pl.DataFrame({'g': ['a', 'b', 'a', 'a']})", {"a": 3, "b": 1},
    # "x = df.group_by('g').len()\nresult = dict(zip(x['g'].to_list(), x['len'].to_list()))"),
]
