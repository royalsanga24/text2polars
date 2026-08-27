"""Probe tasks for polars.

Read one of these and check it yourself: you get a table in, and an answer out.
You do not need to know polars to say whether the answer is right. That is the
property we are protecting — an eval you cannot personally verify is worthless.

Each task gives the model a DataFrame `df` (and sometimes `df2`) and asks it to
compute a variable called `result`.
"""

TASKS = [
    dict(
        id="filter_select",
        instruction="Keep only the rows where age is greater than 30, then return the name column as a plain Python list.",
        setup="df = pl.DataFrame({'name': ['alice','bob','cara'], 'age': [35, 20, 41]})",
        expected=["alice", "cara"],
    ),
    dict(
        id="group_sum",
        instruction="Group by the 'team' column and sum 'points'. Return a dict mapping team -> total points.",
        setup="df = pl.DataFrame({'team': ['a','b','a','b'], 'points': [3, 5, 2, 1]})",
        expected={"a": 5, "b": 6},
    ),
    dict(
        id="sort_head",
        instruction="Sort by 'score' from highest to lowest and return the top 2 names as a list.",
        setup="df = pl.DataFrame({'name': ['x','y','z'], 'score': [7, 12, 9]})",
        expected=["y", "z"],
    ),
    dict(
        id="new_column",
        instruction="Add a column 'total' equal to price times quantity, then return the 'total' column as a list.",
        setup="df = pl.DataFrame({'price': [2.0, 3.0], 'quantity': [4, 5]})",
        expected=[8.0, 15.0],
    ),
    dict(
        id="fill_nulls",
        instruction="Replace null values in the 'v' column with 0, then return 'v' as a list.",
        setup="df = pl.DataFrame({'v': [1, None, 3]})",
        expected=[1, 0, 3],
    ),
    dict(
        id="string_filter",
        instruction="Keep only rows where 'email' contains '@corp.com', and return the 'email' column as a list.",
        setup="df = pl.DataFrame({'email': ['a@corp.com', 'b@other.io', 'c@corp.com']})",
        expected=["a@corp.com", "c@corp.com"],
    ),
    dict(
        id="unique_sorted",
        instruction="Return the unique values in the 'c' column, sorted ascending, as a list.",
        setup="df = pl.DataFrame({'c': [3, 1, 3, 2, 1]})",
        expected=[1, 2, 3],
    ),
    dict(
        id="join",
        instruction="Join df and df2 on the 'id' column, keeping only ids present in both, and return the 'city' column as a list ordered by id ascending.",
        setup=("df = pl.DataFrame({'id': [1, 2, 3], 'name': ['a','b','c']})\n"
               "df2 = pl.DataFrame({'id': [2, 3, 4], 'city': ['x','y','z']})"),
        expected=["x", "y"],
    ),
    dict(
        id="multi_agg",
        instruction="Group by 'k' and compute both the mean of 'v' and the row count per group. Return a dict mapping k -> [mean, count].",
        setup="df = pl.DataFrame({'k': ['a','a','b'], 'v': [1.0, 3.0, 10.0]})",
        expected={"a": [2.0, 2], "b": [10.0, 1]},
    ),
    dict(
        id="elementwise_map",
        instruction="Create a column 'label' that is 'high' when n is 10 or more and 'low' otherwise. Return 'label' as a list.",
        setup="df = pl.DataFrame({'n': [4, 10, 25]})",
        expected=["low", "high", "high"],
    ),
    dict(
        id="rename_cast",
        instruction="Rename column 'a' to 'b', cast it to a 64-bit integer, and return 'b' as a list.",
        setup="df = pl.DataFrame({'a': ['1', '2', '3']})",
        expected=[1, 2, 3],
    ),
    dict(
        id="rank_window",
        instruction="Within each 'grp', rank rows by 'val' descending (1 = highest), and return the ranks as a list in the original row order.",
        setup="df = pl.DataFrame({'grp': ['a','a','b','b'], 'val': [5, 9, 2, 7]})",
        expected=[2, 1, 2, 1],
    ),
]

PREAMBLE = "import polars as pl"
