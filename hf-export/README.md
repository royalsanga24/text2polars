---
license: mit
language:
- en
task_categories:
- text2text-generation
tags:
- code
- code-generation
- polars
- dataframes
- benchmark
- execution-based-evaluation
size_categories:
- n<1K
configs:
- config_name: dev
  data_files: dev.jsonl
- config_name: general
  data_files: general.jsonl
- config_name: held_out
  data_files: held_out.jsonl
---

# text2polars-bench

An execution-based benchmark for **polars** code generation, built to measure
small language models. 268 tasks across three sets that answer three different
questions.

Small models are not merely weak at polars — they are *confidently wrong*. They
reach for pandas (`sort_values`, `fillna`, `groupby`, `tolist`), methods that
are real, familiar, and absent from the library being asked about. This
benchmark was built to measure that, and then to measure whether fine-tuning
actually fixes it.

## The three sets

| Set | Tasks | Question it answers |
|---|---|---|
| `dev` | 120 | How good is the model at polars? |
| `general` | 60 | Did training it damage ordinary Python ability? |
| `held_out` | 88 | Does any improvement generalise past what was trained? |

**`general` and `held_out` are the point.** A single polars score is easy to
improve and easy to misread. On our own runs, a model went from 20% to 60% on
`dev` while `general` collapsed from 83% to 10%, and `held_out` did not move
at all.

## Task format

```json
{
  "id": "pt_sort_desc",
  "category": "pandas_trap",
  "instruction": "Sort rows by 'score' from highest to lowest and return the 'name' column as a list.",
  "setup": "df = pl.DataFrame({'name': ['x','y','z'], 'score': [7, 12, 9]})",
  "expected": ["y", "z", "x"],
  "solution": "result = df.sort('score', descending=True)['name'].to_list()"
}
```

The model is given `setup` and `instruction`, and must write code assigning
`result`. Score by **executing** it and comparing to `expected` — not by
comparing code as text, so a correct answer written differently still counts.

`solution` is a reference implementation, used to validate the task. It is not
shown to the model being evaluated.

## Categories in `dev`

| Category | Tasks | What it isolates |
|---|---|---|
| `output_convention` | 30 | Returning plain Python rather than a DataFrame |
| `pandas_trap` | 30 | Where the natural pandas idiom differs from polars |
| `stale_api` | 30 | polars renamed it; models know the old name |
| `hard` | 30 | Windows, multi-key joins, nested aggregation |

Report per category. A single average hides everything: in our runs `stale_api`
went 0% → 58% while `hard` did not move at all, and the overall number showed
neither.

## How `held_out` was built

This is the set worth explaining, because it is what a self-authored benchmark
usually lacks.

1. Downloaded **752 MIT-licensed polars source files** (the `pola-rs/polars`
   repository — user guide, docs, test suite).
2. Counted which operations real code actually uses: **313 distinct**, across
   32,701 usages.
3. Kept the operations that were **common in real code**, **never emitted by
   our training-data generators**, and **never tested by `dev`**.
4. **Fixed that list before looking at any model score.** Choosing it
   afterwards would mean selecting the questions models happen to fail.
5. Wrote 100 tasks, then **dropped 12** after checking that their non-scaffold
   operations all appeared in training anyway.

The result measures generalisation to operations a model was not shown.

## Validation

Every task has been executed. `expected` was written **independently** of
`solution` — one derived from the other would make agreement meaningless
rather than evidential. Two authoring errors were caught this way.

Answers are polars-version-specific. Validated against **polars 1.43.2**.

## Limitations

- **Small.** 120 / 60 / 88 tasks resolves large effects, not small ones. At
  n=88 a 4.5-point difference sits at p ≈ 0.6. Report significance, not just
  percentages.
- **Frontier models saturate `dev`.** Claude Opus 5 scored 100% on an earlier
  48-task version. This is a diagnostic for small models, not a frontier
  benchmark.
- **Synthetic data.** Tasks use small illustrative DataFrames, not real
  analytical workloads.
- **One library, one language.** Nothing here supports claims about code
  generation generally.
- **`held_out` is held out with respect to a specific training run.** If you
  train on these operations it stops measuring anything. Build a new one.

## Provenance and licence

Tasks were authored for this benchmark and are released under **MIT**.

No third-party code is reproduced. `pola-rs/polars` (MIT) was used only for
**frequency analysis** — counting which operations appear, to select what
`held_out` should test. `operation_frequency.json` contains those counts.

## Harness

Scoring code, contamination screening, and paired significance testing:
**https://github.com/royalsanga24/text2polars**

Includes per-task records for 16 evaluation runs, so published numbers can be
checked without retraining anything.
