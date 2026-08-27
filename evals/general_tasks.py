"""General Python tasks — the guard against catastrophic forgetting.

Nothing here touches polars. The point is to detect the failure mode where a
model gets better at our narrow task and worse at ordinary programming. Without
this, "polars score went up" and "the model got better" are indistinguishable.

Measured before training and after every stage, with the SAME prompt each time
(v1 — the few-shot prompt v3 shows polars examples, which would be noise here).
Comparability to the polars numbers is not the point; comparability to ITSELF
before and after training is.
"""

CATEGORIES = {
    "general_python": "Ordinary Python: strings, lists, dicts, comprehensions",
}

PREAMBLE = ""   # no imports needed

def _t(tid, instruction, setup, expected, solution):
    return dict(id=tid, category="general_python", instruction=instruction,
                setup=setup, expected=expected, solution=solution)

TASKS = [
    _t("g_reverse_string", "Return the string `s` reversed.",
       "s = 'polars'", "sralop", "result = s[::-1]"),
    _t("g_sum_list", "Return the sum of all numbers in `nums` as an integer.",
       "nums = [3, 1, 4, 1, 5]", 14, "result = sum(nums)"),
    _t("g_word_count", "Return a dict mapping each word in `words` to how many times it appears.",
       "words = ['a', 'b', 'a', 'c', 'a']", {"a": 3, "b": 1, "c": 1},
       "result = {w: words.count(w) for w in set(words)}"),
    _t("g_filter_even", "Return only the even numbers from `nums`, as a list, in the original order.",
       "nums = [1, 2, 3, 4, 5, 6]", [2, 4, 6], "result = [n for n in nums if n % 2 == 0]"),
    _t("g_sort_dicts", "Sort `people` by the 'age' key ascending and return the list of names.",
       "people = [{'name': 'a', 'age': 30}, {'name': 'b', 'age': 20}]", ["b", "a"],
       "result = [p['name'] for p in sorted(people, key=lambda p: p['age'])]"),
    _t("g_flatten", "Flatten `nested` into a single list.",
       "nested = [[1, 2], [3], [4, 5]]", [1, 2, 3, 4, 5],
       "result = [x for sub in nested for x in sub]"),
    _t("g_dedupe_order", "Remove duplicates from `items` while preserving the original order.",
       "items = ['a', 'b', 'a', 'c', 'b']", ["a", "b", "c"],
       "result = list(dict.fromkeys(items))"),
    _t("g_second_largest", "Return the second largest distinct value in `nums`.",
       "nums = [5, 9, 2, 9, 7]", 7, "result = sorted(set(nums))[-2]"),
    _t("g_invert_dict", "Return `d` with its keys and values swapped.",
       "d = {'a': 1, 'b': 2}", {"1": "a", "2": "b"},
       "result = {str(v): k for k, v in d.items()}"),
    _t("g_zip_to_dict", "Combine `keys` and `vals` into a dict.",
       "keys = ['x', 'y']\nvals = [10, 20]", {"x": 10, "y": 20},
       "result = dict(zip(keys, vals))"),
    _t("g_running_total", "Return the running cumulative total of `nums` as a list.",
       "nums = [1, 2, 3]", [1, 3, 6],
       "t = 0\nresult = []\nfor n in nums:\n    t += n\n    result.append(t)"),
    _t("g_group_by_letter", "Group `names` into a dict keyed by first letter, values are lists in original order.",
       "names = ['ant', 'bee', 'ape']", {"a": ["ant", "ape"], "b": ["bee"]},
       "result = {}\nfor n in names:\n    result.setdefault(n[0], []).append(n)"),
    _t("g_palindrome", "Return True if `s` reads the same forwards and backwards, else False.",
       "s = 'level'", True, "result = s == s[::-1]"),
    _t("g_join_words", "Join `parts` into one string separated by ' - '.",
       "parts = ['a', 'b', 'c']", "a - b - c", "result = ' - '.join(parts)"),
    _t("g_strip_split", "Split `line` on commas and strip whitespace from each piece. Return the list.",
       "line = ' a , b ,c '", ["a", "b", "c"],
       "result = [p.strip() for p in line.split(',')]"),
    _t("g_max_by_key", "Return the name of the entry in `scores` with the highest 'points'.",
       "scores = [{'name': 'a', 'points': 3}, {'name': 'b', 'points': 9}]", "b",
       "result = max(scores, key=lambda s: s['points'])['name']"),
    _t("g_any_over", "Return True if any value in `nums` exceeds 100, else False.",
       "nums = [4, 250, 6]", True, "result = any(n > 100 for n in nums)"),
    _t("g_dict_get_default", "Return the value for key 'missing' in `d`, or 0 if absent.",
       "d = {'present': 5}", 0, "result = d.get('missing', 0)"),
    _t("g_sort_two_keys", "Sort `rows` by 'grp' ascending then 'v' descending. Return the list of 'v' values.",
       "rows = [{'grp': 'a', 'v': 1}, {'grp': 'a', 'v': 9}, {'grp': 'b', 'v': 4}]", [9, 1, 4],
       "result = [r['v'] for r in sorted(rows, key=lambda r: (r['grp'], -r['v']))]"),
    _t("g_char_count", "Return the number of times the letter 'a' appears in `s`.",
       "s = 'banana'", 3, "result = s.count('a')"),
    _t("g_merge_dicts", "Merge `d1` and `d2`; on conflict `d2` wins. Return the merged dict.",
       "d1 = {'a': 1, 'b': 2}\nd2 = {'b': 9, 'c': 3}", {"a": 1, "b": 9, "c": 3},
       "result = {**d1, **d2}"),
    _t("g_label_numbers", "Label each number in `nums` as 'neg', 'zero' or 'pos'. Return the list of labels.",
       "nums = [-4, 0, 7]", ["neg", "zero", "pos"],
       "result = ['neg' if n < 0 else ('zero' if n == 0 else 'pos') for n in nums]"),
    _t("g_slice_step", "Return every second element of `items`, starting from the first.",
       "items = [1, 2, 3, 4, 5]", [1, 3, 5], "result = items[::2]"),
    _t("g_nested_get", "Return the value at `cfg['db']['port']`, or 5432 if either key is missing.",
       "cfg = {'db': {'host': 'x'}}", 5432, "result = cfg.get('db', {}).get('port', 5432)"),
]


# Tasks 25-60, added because O12: 24 tasks could not detect the residual
# deficit after replay (F38).
from evals.general_tasks_extra import EXTRA_TASKS  # noqa: E402

TASKS = TASKS + EXTRA_TASKS
