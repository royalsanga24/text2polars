# text2polars — project notes

Living document. Update at the end of every session.
Findings and decisions go here so we never re-derive the same thing twice.

---

## 1. What we're building

A small open-weight model that writes **polars** code from a plain-English
instruction, plus the benchmark that proves whether it works.

Four artifacts at the end:

| Artifact | Where | Why it matters |
|---|---|---|
| Benchmark + dataset | Hugging Face | Most reusable piece. Outlives the model. |
| Fine-tuned model | Hugging Face | Open weights + model card with real numbers. |
| Reproducible pipeline | GitHub | One command reruns everything. |
| Writeup | Blog / arXiv | Where the contribution gets read. |

Headline result is one honest sentence, e.g. *"CPT+SFT on a 1.5B cut
pandas-interference errors by N% and raised execution accuracy from 8% to X%,
running locally on a laptop."* A negative result still counts, if measured.

---

## 2. The plan

| # | Phase | What gets built | Status |
|---|---|---|---|
| 1 | Pick the target, prove the gap | `probe/` | **done** |
| 2 | Eval set + scorer | 48 validated tasks, `evals/run_eval.py` | **done** |
| 3 | Baseline | v3 prompt, 1.5B target, ceiling = 100% | **done** |
| 4 | Training data | SFT set done (2,850); CPT corpus outstanding | **in progress** |
| 5 | Data curation stack | dedup (minhash), quality filter, PII strip, decontamination, provenance | |
| 6 | Post-training | arms 1 & 1b done — **polars 50.0%, general 66.7%**; CPT arms next | **in progress** |
| 7 | Serving | vLLM, prefix caching, AWQ quantization, p95 latency, no quality regression | |
| 8 | Release | Dataset, weights, model card, writeup | |

---

## 3. What you learn, by phase

- **2 — Eval.** Metric design, stratification, leakage, held-out discipline.
  The highest-value skill on the list.
- **3 — Baseline.** Prompting properly, cost/latency as first-class metrics,
  and why you always measure the simple approach first.
- **4–5 — Data.** Curation, synthetic generation, dedup, decontamination,
  provenance. ~40% of the real job.
- **6 — Post-training.** What CPT vs SFT vs DPO are each actually *for*, LoRA,
  loss masking, and how to tell whether a stage helped.
- **7 — Serving.** Quantization, batching, caching, measuring p95 rather than
  average, proving no quality regression after compression.
- **8 — Release.** Model cards, reproducibility, writing up honestly.

---

## 4. Findings

**F1 — The gap is real.** 12 validated polars tasks:
Claude Opus 5 **11/12 (92%)**, qwen2.5-coder:1.5b **1/12 (8%)**.
A clearer prompt (v2) scored **0/12** — prompting does not close it.
All 12 tasks were verified solvable by hand first, so the failures belong to
the models, not the tasks.
Caveat: n=12 detects a 90-point gap and nothing finer. 1/12 vs 0/12 is noise.

**F2 — The failure is pandas interference, not ignorance.** *(the interesting one)*
The small model confidently writes real *pandas* methods into polars code:
`sort_values`, `fillna`, `tolist`, `apply`, `groupby`, `to_dict('records')`.
It has a strong wrong prior from the dominant sibling library.
**Testable prediction for Phase 6:** if this is a pandas prior rather than
missing knowledge, CPT on raw polars text should suppress pandas-method errors
specifically, and faster than other error types. So measure per-error-type,
not just overall accuracy.

**F3 — Error types are unevenly distributed.** Of 11 failures: 6 were the same
output-conversion mistake, 3 pandas syntax, 3 stale polars API
(`groupby`→`group_by`, `rank(desc=)`→`descending`). One repeated mistake can
dominate a score and make a trivial fix look like a breakthrough.

**F4 — Unsloth runs on Mac.** Docs confirm Desktop/Studio train on macOS.
CPT/SFT/DPO may run **locally on the M3 Pro** — no GPU rental. Binding
constraint is 18 GB unified memory: 1.5–3B at 4-bit fits, 7B is tight.
Whether the scriptable `unsloth` package trains on Metal is untested (O5).

---

**F5 — The four categories discriminate sharply.** qwen2.5-coder:1.5b on the
48-task dev set (prompt v1):

| category | score |
|---|---|
| output_convention | 8/12 — 67% |
| pandas_trap | 1/12 — 8% |
| stale_api | **0/12 — 0%** |
| hard | 1/12 — 8% |
| **overall** | **10/48 — 21%** |

`stale_api` at zero is the sharpest CPT target we have. Top error kinds:
`to_dict` (8), `groupby` (8), `apply` (4), `sort_values` — all pandas.

**F6 — The headline number is partly a design choice.** Same model, same
prompt: 8% on the 12-task probe, 21% on the 48-task dev set. Nothing improved —
only the *mix* of task categories changed. A set weighted toward output
conventions flatters the model; one weighted toward stale APIs buries it.
Defence: always report per category, and state the mix. This is how benchmarks
get gamed, usually by accident.

**F7 — Validation cannot check the instruction.** `evals/validate.py` proves the
reference solution agrees with `expected` (written independently, so agreement
is real evidence). It cannot prove the *instruction* describes the task — if the
instruction says "ascending" and both solution and expected are descending, it
passes. Only a human catches that. Hence `--review`.

---

**F8 — Most failures are crashes, not wrong answers.** On 48 tasks the 1.5B
model produced 36 that *failed to run* and only 2 that ran and gave a wrong
answer. The bottleneck is syntactic/API validity, not reasoning. Implications:
(a) the first thing training must fix is producing runnable polars at all,
(b) "does it run" is a strong, free reward signal for DPO in Phase 6.

**F9 — The v1/v2 prompt difference is noise, and I over-read it earlier.**
48 tasks: v1 20.8%, v2 25.0%. Paired comparison: 8 tasks changed outcome,
5 one way and 3 the other; two-sided exact binomial **p = 0.73**. No effect.
This *reverses* the 12-task probe reading (where v2 scored 0/12) — that was
noise too. Per-category swings at n=12 are meaningless.

**F10 — How big the eval needs to be, measured not guessed.**
At 48 tasks and ~22% accuracy, 1 SD ≈ 2.9 tasks ≈ 6 percentage points, so
changes under ~12 points are unreliable. Detecting a 10-point improvement
dependably needs roughly **120–150 tasks**.
=> 48 suffices for the headline claim (21% -> 60%+ would be unmissable) but NOT
for comparing training variants. Grow the set before hyperparameter work,
not before Phase 3.

---

**F11 — Prompting improves the 1.5B but does not close the gap.**
qwen2.5-coder:1.5b on the 48-task dev set:

| prompt | overall | note |
|---|---|---|
| v1 naive | 20.8% | |
| v2 output convention stated | 25.0% | vs v1: p=0.73, noise |
| **v3 three few-shot examples** | **35.4%** | vs v1: +14.6pp, **p=0.065**, suggestive |
| v4 v3 + pandas->polars cheatsheet | 29.2% | vs v3: p=0.375, noise |

Few-shot is positive in all four categories (+1/+3/+2/+1), which is mild extra
evidence beyond the p-value. The cheatsheet has no detectable effect — do not
explain a result that has not been established.
Conclusion: best prompting reaches ~35% against Claude's ~92%. **Fine-tuning
stays justified.**

**F12 — We are already at the limit of our instrument.** The improvement we most
care about sits at p=0.065, i.e. not firmly establishable on 48 tasks. This is
F10 arriving on schedule. Optimising against a measurement you cannot read is
how weeks get wasted, so the prompt search stops here.

---

**F13 — Doubling model size buys nothing.** *(the key result so far)*
Same prompt (v3), same 48 tasks: qwen2.5-coder **1.5b = 35.4%**,
**3b = 33.3%**. Paired: 4 fixed, 5 broken, exact two-sided **p = 1.000**.
Interpretation: the failure is not capacity, it is **knowledge**. A bigger model
with the same stale training data has the same pandas prior. This is precisely
the diagnosis that justifies CPT rather than just scaling.
Sharpens the paper claim: *"2x parameters buys nothing; N tokens of domain text
buys X"* beats any raw accuracy number.
Caveat: p=1.0 means no *detectable* difference at n=48, not proven identical.

---

**F14 — Claude Opus 5 scores 48/48 (100%) with prompt v3.**
Three consequences:
1. **The eval is independently validated.** Claude reproduced every `expected`
   value without seeing our reference solutions. `validate.py` only showed our
   own solution agreed with our own answer — same author, correlated errors.
   This is real triangulation. The numbers can be trusted.
2. **The ceiling is 100%**, so the entire 35%->100% gap is closeable, and
   **Claude is qualified to generate training data** (Phase 4 de-risked).
3. **The benchmark is saturated at the top** — see F15.

**F15 — "hard" is not hard, it is hard-for-a-1.5B.** Claude scored 12/12 on it.
The benchmark has no headroom above small models. Fine as a training target and
a small-model diagnostic; weak as a published benchmark ("saturated" will be the
first reviewer comment). Options: frame honestly as a small-model diagnostic
(chosen for now, D17), or add a genuinely compositional tier later where a
frontier model scores ~60%.

**F16 — The value proposition is cost and latency, not quality.**
Claude 100% at 2.7 s/task and paid; qwen2.5-coder:1.5b 35.4% at 0.5 s/task,
free, local. **5.9x faster.** We are not trying to beat Claude — we are trying
to get close enough far cheaper. This is the distillation pattern and the most
common real-world reason to fine-tune.
Working thesis: *can a 1.5B running locally reach ~85% of a frontier model's
accuracy on a library neither was trained on?*

---

**F17 — Contamination is about (data, answer), not about wording.**
First attempt screened instruction text with n-gram Jaccard. It failed
calibration badly: best setting still missed a near-duplicate and flagged
561/1128 unrelated pairs. Two causes — these instructions are short and
templated so unrelated tasks share phrasing, and more importantly **I was
measuring the wrong thing**. Renaming columns and changing numbers produces a
task over *different data with a different answer*, which the model cannot
memorise; it has to compute, which is the skill under test.

| level | situation | verdict |
|---|---|---|
| 1 | same data, same answer | fatal — memorisable |
| 2 | same data, reworded instruction | fatal — still memorisable |
| 3 | same skill, different data | fine — this is what training data *is* |

Rebuilt on a `(normalized setup, normalized expected)` fingerprint plus
near-identical-data and verbatim-solution checks. Result: **0/96 contamination
missed, 0/8 novel tasks falsely flagged, 2/56 total false alarms.**

**F18 — When a test fails, suspect the test.** The rebuilt filter first showed
96 "false alarms". Both groups were my test's fault: the "unrelated" tasks were
built from *other dev tasks* (which really are contamination), and the
"different data" mutation only bumped digits, leaving all-string tasks
identical. Same lesson as "when the frontier model fails, suspect the task".

---

**F19 — One dev task could not tell a right rule from a wrong one.**
Mutating the numeric literals in each reference solution showed
`h_when_then_chain` still scored correct with thresholds of 7 and 107 instead
of 10 and 100 — its data `[5, 15, 150]` had nothing near either boundary.
Fixed to `[5, 9, 12, 98, 103]`; all four mutations now caught.
(Two other flagged tasks were artifacts of the naive check: lowering a
threshold can never flip `any(v > 10) == True`, and the `1` in `head(1)` is not
a threshold.) Task-set hash changed, so all baselines were re-run — identical
at 20.8% / 35.4%. **Claude's 100% needs re-running on the new hash.**
This is the PromQL discriminating-data lesson arriving in a new costume.

**F20 — SFT set built: 2,850 train / 150 val, 100% verification yield.**
Diversity: **100% unique data**, 40% unique instruction phrasings, 34% unique
solution patterns. Repetition of solution patterns is arguably desirable (learn
`group_by`, not `groupby`); repetition of phrasings is a genuine risk of
template-generated data.
**The dev set is an honest test of this** — its instructions were hand-written
before the generator existed, so if the model only learns template phrasings,
the eval will catch it. Measure before adding phrasing variety.
Mix, declared: pandas_trap 42%, hard 23%, stale_api 19%, output_convention 15%.

**F21 — Decontamination blocked 0 of 3,000 real candidates.** Correct: randomly
generated data essentially never reproduces a dev task's exact data *and*
answer. An earlier, wrong version blocked 134 — all false alarms, all triggered
by generated code sharing a generic idiom with a reference solution, which is
the pattern we are *trying to teach*.
**A safety check that catches nothing is doing its job.** You validate it with
planted tests (0/96 missed), not by waiting for a real incident.

**F22 — My test was wrong three times in this phase, my code once.**
"Unrelated" examples built from real dev tasks; a mutation that didn't apply to
half the cases; a sabotage that no-oped on `<`. Test code gets written faster
and reviewed less than real code, so it is *more* likely to be wrong, not less.
When a test fails, check the ruler before the thing measured.

---

**F23 — General-ability baseline: 19/24 = 79.2%.** `evals/general_tasks.py`,
24 ordinary Python tasks (strings, lists, dicts, comprehensions), no polars.
The 1.5B is genuinely competent at normal Python. **This is the number that
detects catastrophic forgetting** — if it drops after training, we traded
general ability for polars ability and must say so.
Run with prompt v1 always: v3's few-shot examples are polars-specific and would
be noise here. Comparability to the polars score is not the point;
comparability to itself before and after training is.

**F24 — Metal works, and the training stack installs.** `torch.backends.mps`
available and ~1.8x faster than CPU on a 2000x2000 matmul. `pip install unsloth`
resolves on macOS/arm64: torch 2.11, transformers 5.5, trl 0.24, peft 0.20,
bitsandbytes 0.50 (1.7 GB). Notably **no triton and no xformers** — those are
gated to Linux/Windows, so unsloth's fast kernels are absent. Whether it
actually *runs* is still open (O5).

---

**F25 — O5 ANSWERED: unsloth works on the Mac, by delegating to MLX-LM.**
`from_pretrained` prints *"Loading ... via mlx-lm (runtime 4-bit affine
quantization)"*, returns an `mlx_lm.models.qwen2.Model` on `Device(gpu, 0)`,
generates correct code, and `get_peft_model` attaches LoRA
(540,672 trainable params, 0.28% of 192M). bitsandbytes is irrelevant here —
MLX does its own affine quantization.
`mlx_lm lora` provides a full training CLI including **`--mask-prompt`**, which
is the loss masking discussed in §5b. No GPU rental needed for the early arms.

**F26 — My probe reported the exact opposite, and it was wrong.**
The first test printed "4-bit: NO / 16-bit: NO". Both failures were
`next(model.parameters())` in MY code — MLX returns a dict where PyTorch
returns a generator. The model had loaded fine in both cases.
**Fourth time this session the test was broken rather than the code.**
A probe that reports failure deserves the same scrutiny as one that reports
success; I nearly concluded "unsloth does not work on Mac" from my own bug.

---

**F27 — O10 was real: a 12.5-point gap between two runtimes of the SAME model.**
Cause found by elimination:

| runtime / precision | score |
|---|---|
| MLX 4-bit affine | **25.0%** |
| MLX 8-bit | 35.4% |
| MLX bf16 | 35.4% |
| ollama Q4_K_M | 35.4% |

**Not all 4-bit is equal.** ollama's Q4_K_M (mixed-precision k-quant) matches
full precision; MLX's default affine 4-bit loses **10.4 points**. Both are
labelled "4-bit" and both labels are honest.
8-bit is lossless here — useful for serving.
The vendor system prompt ollama injects silently is worth ~1-2 tasks, i.e.
noise. Once quantization is controlled, ollama vs MLX bf16 is **p = 0.754** —
indistinguishable. **The original 35.4% baseline was correct.**
Consequence: Phase 7's "validate no quality regression after quantization"
arrived early and uninvited — we had been measuring quantized models all along
without saying so.

**F28 — Reference baselines, all on one runtime (MLX bf16, our own prompt):**

| eval | prompt | score |
|---|---|---|
| polars dev | v1 (the training prompt) | **22.9%** |
| polars dev | v3 (best prompting) | **31.2%** |
| general python | v1 | see below |

The trained model must beat **22.9%** to show training did anything, and
**31.2%** to be worth more than a better prompt.

---

**F29 — 20-iteration smoke run: no real gain, real damage.**
Base 22.9% -> smoke adapter 29.2% on prompt v1. Paired: 9 fixed, 6 broken,
**p = 0.607 — noise.** But the model got measurably *worse* in a way accuracy
does not show:

| | base | 20-iter adapter |
|---|---|---|
| median chars generated | 53 | 86 |
| longest generation | 115 | **1816** |
| eval wall-clock | 54s | **753s (14x)** |

It loops instead of stopping:
`!['k']  # This is a placeholder for the actual key values` repeated to the
512-token limit. 20 steps at lr 1e-4 disturbed the stopping behaviour before
teaching anything.
**A single accuracy number can hide a model that is broken in ways that
matter.** Latency is half this project's thesis; a model that will not stop is
useless regardless of its score.

**F30 — The task-set hash guard earned its keep.** `compare.py` matched on
prompt but not task set, so it grabbed a 24-task *general* run as the
counterpart to a 48-task *dev* run. The hash check refused instead of printing
a confident nonsense number. Now fixed to filter on task set too — but a guard
you rely on tripping is a guard you should not need.

---

## ARM 1 RESULT — SFT only

**F31 — SFT works: 22.9% -> 47.9% on the dev set (prompt v1).**
Paired vs its own baseline: 17 fixed, 5 broken, **p = 0.017 — REAL**.
Paired vs best prompting (v3 base, 31.2%): 17 fixed, 9 broken, **p = 0.169 —
suggestive but NOT established.** That second one is F10/F12 arriving exactly
as predicted: 48 tasks cannot resolve a ~17-point difference. Needs 120-150.

**F32 — The improvement lands precisely where the training data pointed.**

| category | before | after | change |
|---|---|---|---|
| stale_api | 0.0% | 58.3% | **+7 tasks** |
| pandas_trap | 16.7% | 58.3% | **+5 tasks** |
| output_convention | 58.3% | 58.3% | 0 |
| hard | 16.7% | 16.7% | 0 |

The two knowledge categories moved; the two that need reasoning or format
generalisation did not. Direct support for F2: the failure was a stale-knowledge
and pandas-prior problem, and teaching current polars fixed exactly that.

**F33 — The failure mode moved, which a flat metric would hide.**
`failed to run` 31 -> 9; `ran but wrong` 2 -> 16. The model went from "cannot
write valid polars" to "writes valid polars, sometimes computes the wrong
thing". Different problems, different fixes; only the second is reasoning.

**F34 — Our training data taught a bad habit.** `oc_max_float` regressed:
the model wrote `int(df['v'].max())` -> 9, expected 9.25. The `f_scalar_agg`
template only ever produced integer answers (sum/max/min/count), so the model
learned "scalar means int()". Template narrowness leaking into the weights.
Fix: generate float cases in that family.

**F35 — A tokenizer config bug cost a 22x slowdown, and nearly killed the
thesis.** The trained model ran at 22.5 s/task vs 1.0 s/task for the base.
Diagnosis: token throughput was actually *higher* with the adapter (33.6 vs
25.7 tok/s) — it simply generated **491 tokens instead of 15**, running to the
cap. Cause:

```
tokenizer eos_token_id      : 151645  (<|im_end|>)   <- what the model emits
tokenizer eos_token_ids set : {151643} (<|endoftext|>) <- what generate halts on
```

The model correctly emitted `<|im_end|>` at token 9 and generation sailed past
it, restarting the answer 24 times. **Training was fine; the model was fine;
the generation config disagreed with itself.**
Fixed by unioning `eos_token_id` into the stop set: **1080s -> 51s, accuracy
unchanged at 47.9%.** Base is 49s — i.e. same speed, double the accuracy.

**F36 — The metric that should have caught F35 measured the wrong string.**
`runaway` counted characters of the *extracted* code block, not the *raw*
generation. The model wrote clean code then rambled for 400 more tokens, so the
snippet looked short and runaway reported 0. Now records `raw_chars`.
A metric that cannot detect the failure it was added for is worse than none —
it provides false assurance.

---

**F37 — CATASTROPHIC FORGETTING. Arm 1 is a trade, not a win.**

| eval | before | after (700 steps) | ckpt 200 |
|---|---|---|---|
| polars dev | 22.9% | **47.9%** | — |
| general Python | **83.3%** | **20.8%** | 25.0% |

Not a measurement artifact: the general prompt never mentions polars (verified)
and the base model scored 83.3% on that exact prompt.

The model stopped being a Python model and became a polars-only model:
- `g_word_count` -> `df['words'].value_counts()` — **invents a DataFrame that
  does not exist** in a plain-list task
- `g_sum_list` -> `int(nums[0])` — the F34 `int()` habit leaking into general code
- 4-5 tasks generate degenerate loops

**By any honest accounting this is a worse model overall**: +25 points on a
narrow skill, -62.5 on a broad one.

Cause is visible in the training log: **train loss hit 0.000 at iteration 140**
and stayed there. Iterations 140-700 were 560 steps of grinding on data already
memorised perfectly. What gets ground away is everything else the model knew.

**Early stopping alone does not fix it** — checkpoint 200 is 25.0%, barely
better than 700. The damage happens fast because the data is homogeneous: every
single example is "here is a df, write polars".

**This is the entire justification for O9.** Without the general eval we would
have published "22.9% -> 47.9%, more than double!" and been badly wrong.

---

## ARM 1b RESULT — SFT + 25% replay

**F38 — Replay fixes catastrophic forgetting at no cost to the target skill.**
Only one variable changed from arm 1: 25% of the training set is ordinary
Python. Same model, LR, rank, iterations.

| | polars dev | general Python |
|---|---|---|
| base | 22.9% | 83.3% |
| arm 1 (polars only) | 47.9% | **20.8%** |
| **arm 1b (+25% replay)** | **50.0%** | **66.7%** |

Significance:
- general 20.8% -> 66.7%: **p = 0.003, REAL**
- polars 47.9% -> 50.0%: **p = 1.000, no cost**
- remaining gap vs base (66.7% vs 83.3%): p = 0.289, not distinguishable —
  but n=24 cannot detect a 16-point deficit, so this is a limitation, not a
  clean bill of health. Grow the general eval before claiming full recovery.

**Qualitative evidence is cleaner than the score.** Tasks where the model
reaches for a DataFrame that does not exist:

| | score | tasks using df/pl | runaway |
|---|---|---|---|
| base | 83.3% | 0/24 | 0 |
| arm 1 | 20.8% | **5/24** | 5 |
| arm 1b | 66.7% | **0/24** | 0 |

Arm 1 wrote `df['words'].value_counts()` for a plain list of words. Arm 1b
never does.

**F39 — Replay also cured the F34 `int()` habit for free.** `oc_max_float`
regressed in arm 1 (`int(df['v'].max())` -> 9 instead of 9.25); it passes in
arm 1b, and `output_convention` went 58.3% -> 75.0%. Narrow data caused the
habit; more varied data removed it. Data diversity is not only about coverage —
it prevents spurious rules being learned from accidental regularities.

**Headline so far:** polars 22.9% -> 50.0% with no measurable loss of general
ability, from a 20 MB adapter, at unchanged inference speed, trained on a
laptop in 13 minutes.

---

## REVISION — everything above was measured on too small an eval

**F40 — On 120 polars / 60 general tasks, the earlier conclusions weaken sharply.**

| | polars (120) | general (60) |
|---|---|---|
| base, prompt v1 | 20.0% | **83.3%** |
| base, prompt v3 (best) | 25.0% | — |
| SFT only | 33.3% | **10.0%** |
| SFT + 25% replay | **38.3%** | **41.7%** |

Significance on the larger sets:
- SFT vs best prompting: +8.3pp, **p = 0.203 — NOT established**
- replay vs no replay (polars): +5.0pp, **p = 0.307 — NOT established**
- replay vs no replay (general): +31.7pp, **p = 0.000 — REAL**
- base vs replay (general): -41.7pp, **p = 0.000 — REAL, ability is NOT restored**

**F41 — The earlier gains were inflated by operation coverage, not contamination.**
Splitting the 120 by whether a task predates the training-data design:

| model | original 48 | new 72 |
|---|---|---|
| base v1 | 22.9% | 18.1% |
| SFT | 47.9% | 23.6% |
| SFT+replay | 50.0% | 30.6% |

Gain over base, measured on each half separately:

| | original 48 | new 72 |
|---|---|---|
| SFT | **+25.0%** | **+5.6%** |
| SFT+replay | **+27.1%** | **+12.5%** |

No eval example leaked — decontamination did its job. The overlap is at the
level of *which operations exist*: the 13 generator families were written to
cover the same kinds of operations the original 48 tested. On operations the
generators never emit (`is_between`, `arg_sort`, `cum_prod`, `with_row_index`,
`concat_str`, cross/semi joins, `fill_null(strategy=)`), the model gains little.

**The honest claim is now:** SFT teaches the specific operations it is shown and
generalises weakly beyond them. Coverage of the training distribution, not
volume, is the binding constraint.

**F42 — Replay helps forgetting a lot but does not fix it.** 10.0% -> 41.7% is
real and large; 41.7% vs the base's 83.3% is also real. The earlier
"not distinguishable" (p = 0.289) was purely the 24-task instrument, exactly as
O12 warned. 25% replay is not enough.

---

**F43 — Generator coverage broadened 13 -> 28 families, 18 -> 47 operations.**
Eval-operation coverage 28% -> 81% (84% on the new 72 tasks).
Selection was made by walking the polars API by area (aggregation, element-wise
math, boolean, selection, ranking, windows, strings, nulls, schema, joins),
not by reading the eval.

**F44 — ...but the honesty check is WEAK, and this must be stated.**
Only **3 of 47** emitted operations are ones the eval never tests. The intent
was to demonstrate the selection sampled the library rather than the answer
key; 44/47 overlap does not demonstrate that. Some convergence is inevitable
(both target common polars operations), **but the same person wrote the
training generators and the benchmark, so influence cannot be ruled out.**

=> **Any gain from this broadening is PROVISIONAL** until measured on the
held-out test set built from independent real-world polars code (O2), which
still does not exist. Elevated: that test set is now the highest-value
outstanding item, ahead of the CPT arms.

General lesson worth keeping: when one author writes both the training data and
the benchmark, coverage claims cannot be trusted however principled the process
felt. Only genuinely independent held-out data settles it.

**F45 — My coverage measurement was wrong the first time (5th such case).**
It scanned generator *source* for `.method(`, but methods appear there inside
f-string variables (`expr = "abs()"`), so `abs`/`clip`/`mean` read as uncovered
when they are emitted constantly. Fixed by measuring the generated *output*.
Measure the artefact, not the code that produces it.

---

## COVERAGE + REPLAY SWEEP

**F46 — Operation coverage was the binding constraint. Confirmed, and large.**

| run | polars (120) | general (60) |
|---|---|---|
| base, prompt v1 | 20.0% | 83.3% |
| base, prompt v3 (best) | 25.0% | — |
| narrow data + 25% replay | 38.3% | 41.7% |
| **broad data + 25% replay** | **60.0%** | **53.3%** |
| broad data + 50% replay | 58.3% | 56.7% |

- coverage (narrow -> broad), polars: **+21.7pp, p = 0.000 REAL**
- coverage, general: +11.7pp, p = 0.092 suggestive
- **training now beats the best prompt: 25.0% -> 60.0%, p = 0.000 REAL**
  (unprovable at n=48 and at narrow coverage)
- forgetting reduced but NOT solved: 83.3% -> 53.3%, **p = 0.000 REAL**

**F47 — Replay beyond 25% does nothing.** 25% -> 50%: polars p = 0.754,
general p = 0.804, both noise. Replay is a threshold, not a dial: some is
essential (10.0% -> 41.7% earlier), more is wasted. O14 answered.

**F48 — The coverage gain is real but partly benchmark alignment (see F44).**

| run | original 48 | new 72 |
|---|---|---|
| narrow + 25% | 50.0% | 30.6% |
| broad + 25% | **64.6%** (+14.6) | **56.9%** (+26.3) |

Broadening lifted the new 72 more (+26.3) than the original 48 (+14.6) — what
alignment would predict, since those are the operations targeted. **But it also
lifted the original 48 by 14.6pp, whose operations the narrow generators
already covered.** That gain cannot be alignment; most likely more varied data
reduces over-memorisation. Both effects present. Only the independent test set
(O2) can separate them.

**F49 — Data diversity protects general ability too.** Broadening polars
coverage raised the GENERAL score (41.7% -> 53.3%) with no gaming explanation
available: the general eval was never targeted. Diversity is not only about
covering the target skill; it reduces the collapse of everything else.

---

## THE HELD-OUT RESULT — O2 ANSWERED

**F50 — On operations never trained, fine-tuning bought NOTHING.**

`evals/test_tasks.py`: 30 tasks using polars operations that are (a) in the
top-80 most used in real third-party code, (b) never emitted by our training
generators, (c) never tested by our dev set. Operation list fixed from
real-world frequency **before any score was seen**.

| model | dev (120) | general (60) | **held-out (30)** |
|---|---|---|---|
| base | 20.0% | 83.3% | **43.3%** |
| narrow SFT + 25% replay | 38.3% | 41.7% | **13.3%** |
| broad SFT + 25% replay | **60.0%** | 53.3% | **40.0%** |

- base vs best trained: 43.3% vs 40.0%, **p = 1.000 — no difference**
- base vs narrow trained: 43.3% vs 13.3%, **p = 0.012 — training made it WORSE**
- narrow vs broad: 13.3% vs 40.0%, **p = 0.021 — broadening repaired the damage
  but never beat the untrained baseline**

**The 60% dev score was the model learning the specific operations we showed it.
It did not generalise to polars at large.** F44's warning was correct and this
is the measurement that proves it. The dev-set number is an in-distribution
number and must always be reported next to the held-out one.

**Where the real corpus came from:** 196 MIT-licensed files from pola-rs/polars
(user guide + test suite), 1.3 MB, 229 distinct operations observed. Our
training emits 47; our dev set tests 54; of the real top-80, training covers
30 and the dev set 37. Stats in `data/real/operation_stats.json`. (O4: MIT,
frequency analysis only, no redistribution.)

**F51 — This makes the CPT arm the sharpest remaining question.** SFT on
synthetic templates teaches exactly what it shows and no more. Continued
pretraining on real polars text is the intervention that could plausibly
produce transfer, because it exposes the model to operations no template author
would think to write. That is now a genuine open hypothesis with a clean way to
test it: the held-out set.

---

**F52 — O11 ANSWERED: measured inference speed.** Warm-up first, then prompt
processing and generation timed separately on identical prompts:

| config | prompt tok/s | gen tok/s | tokens out | s/task |
|---|---|---|---|---|
| base bf16 | 840 | 44.4 | 17 | 0.61 |
| base + adapter | 710 | 39.2 | 10 | 0.54 |
| **fused bf16** | 824 | 46.4 | 10 | **0.45** |

- **keeping the adapter separate costs ~12% generation / ~15% prompt speed** —
  the extra B(Ax) matmuls per layer per token.
- **fusing recovers all of it** (`mlx_lm fuse`); the result is an ordinary
  2.9 GB model with no adapter. Adapter on disk: 20 MB.
- the trained model emits **10 tokens vs the base's 17** — it answers directly
  instead of padding, so it is faster per task even before fusing.
- merging is mathematically exact in bf16 (`W' = W + BA`). **Merging into a
  QUANTIZED base is not** — requires dequantize/requantize, which given F27
  (MLX 4-bit costs 10 points) must be measured after merging, not assumed.

**F53 — Correcting a plausible-sounding wrong theory.** It is tempting to
explain the no-transfer result (F50) by saying SFT only teaches "how to talk"
and does not modify attention. **Our adapter demonstrably modifies q_proj,
k_proj, v_proj and o_proj** (plus up/down/gate) across 16 of 28 layers — all
four attention projections. CPT and SFT do not differ in *which weights* they
touch; they differ in (a) what data is seen and (b) which tokens carry loss.
The transfer failure is a **data** fact: the model never saw the held-out
operations. No choice of weight location creates knowledge absent from the data.

---

**F54 — The held-out set cannot stay held-out through CPT. Caught before
training, not after.** All 31 operations tested by `evals/test_tasks.py` appear
in the CPT corpus — necessarily, since they were selected *from* it by
real-world frequency. Correct for measuring SFT transfer; invalid for measuring
CPT transfer.

Options considered:
- **(A) relabel and proceed** — post-CPT the set measures "did CPT teach these
  common operations that SFT could not?", which is exactly F51's question. The
  SFT transfer result (F50) is already banked and unaffected. **CHOSEN.**
- (B) split the corpus by file — common operations appear in nearly every file,
  so the residue is again the rare tail.
- (C) build a set from the 97 operations absent from the corpus — genuine
  transfer test, but they are the obscure tail (trig, bitwise) where the base
  scores near zero, so it would carry little information. Parked.

General lesson: **a held-out set is held out with respect to a specific
training run, not forever.** Changing the training data can silently invalidate
it. Check the overlap before every new training stage, not once.

---

**F55 — O3 ANSWERED: the CPT corpus is clean.** `curation/decontaminate_text.py`
screens raw source against all 210 eval tasks (dev + general + held-out).
Result: **0 of 196 documents flagged.** Real polars code does not contain our
specific toy-data-plus-answer pairs.

The first implementation flagged **50 of 196 (26%), every one innocent** — it
matched on the data literal alone, and `pl.DataFrame({'a': [1,2,3]})` appears
in nearly every polars test file. Narrowed to require the task's data **and**
its solution within 600 characters of each other.
**This is D20 relearned in a second context.** Contamination is data AND
answer; either alone flags the world. Worth stating as a standing rule rather
than a per-case discovery.

Corpus after screening: 196 files, 1.30 MB, **~0.32M tokens** — small. Taking
all 734 `.py` files in the repo would reach roughly 1.2M tokens. Real CPT runs
use 50M-1B. A null result at this scale would be ambiguous between "CPT does
not help" and "1M tokens is not enough", and that ambiguity must be stated
before the run, not discovered after it.

---

## CPT SETUP (arms 2 and 3)

**Corpus:** 752 files from pola-rs/polars (all `.py` plus `docs/source` `.md`),
MIT, **9.57 MB / ~2.39M tokens**. Screened against all 210 eval tasks —
**0 flagged** (F55). Chunked on line boundaries into 3,332 pieces of ~800
tokens, because mlx_lm *truncates* rather than splits anything over
`max_seq_length` and a 40 KB file would otherwise contribute its first 1000
tokens and silently drop the rest.

**Stated before the run, so it cannot be a post-hoc excuse:**
- **2.4M tokens is small for CPT.** Real runs use 50M-1B. A null result is
  ambiguous between "CPT does not help here" and "2.4M tokens is not enough".
  A *positive* result is unambiguous, which is why the run is still worth doing.
- **Learning rate 5e-5 was chosen by convention, not swept.** If CPT
  underperforms, LR is an unexcluded explanation and must be reported as such.
- **The corpus is the library's own repo**, not third-party application code.
  It over-represents API-definition and test style, under-represents how a data
  analyst actually strings operations together.
- **`evals/test_tasks.py` is no longer a transfer test after CPT** (F54). It
  becomes "did CPT teach these operations that SFT could not?" — F51's
  question. Report it under that name.

**Arm 3 layout:** fuse the CPT adapter into the base, then train a *fresh* SFT
adapter on the fused model. Keeps the stages cleanly separated, which the
ablation needs. `train/run_sft.sh` takes a base-model override as its 4th arg.

---

## ARM 2 — CPT ONLY

**F56 — A parser bug nearly produced a false result about CPT.**
First scoring gave dev 8.3%, general 30.0%, held-out 6.7% — apparently
catastrophic. **81 of 110 dev failures were our `extract_code`**, not the model:
CPT trained on documentation full of code blocks, so the model emits
```` ```python ```` **and never closes the fence**. Our regex required a closing
fence and handed correct polars to Python as a syntax error.
Fixed to accept unclosed fences. Corrected scores: **dev 28.3%, general 58.3%,
held-out 53.3%.**
Checked which past runs were affected rather than re-running all 17: **only the
3 CPT runs**; every other model emitted zero unclosed fences, because CPT is
what taught this model to write markdown.
**Lesson: a measurement tool is only validated against behaviours it has
already seen.** It worked for three model families and broke on the fourth.

**F57 — Arm 2 results, with significance.**

| eval | base | CPT only | Δ | p |
|---|---|---|---|---|
| dev (120) | 20.0% | 28.3% | +8.3 | 0.164 |
| general (60) | 83.3% | **58.3%** | **-25.0** | **0.000 REAL** |
| held-out (30) | 43.3% | **53.3%** | +10.0 | 0.549 |
| held-out vs best SFT (40.0%) | — | 53.3% | **+13.3** | 0.388 |

**Only the forgetting is established.** CPT costs 25 points of general Python,
solidly. Everything positive is suggestive and unproven.

**F58 — The §5b prediction was correct in mechanism.** CPT partly overwrote
instruction-following: 22 dev tasks came back as a *bare value* (`10`, `3`,
`[1, 3, 2]`) instead of code. Asked to write code assigning `result`, the model
computed the answer and wrote it down — continuing text rather than answering,
which is exactly what next-token training on raw code rewards.
A benchmark measures a **behaviour**, not a capability. The model may know more
polars and still score lower because it stopped doing the graded thing.

**F59 — The most interesting result in the project is currently unprovable.**
CPT beats the best SFT model by **+13.3pp on held-out operations — where SFT
was flat against base (F50, p = 1.000)**. That is exactly the F51 hypothesis,
and at n=30 it sits at p = 0.388.
This is the third time sample size has been the limiter (F10, F12, O12).
**The held-out set must grow before this question can be answered.**

---

## O15 ANSWERED — THE CENTRAL RESULT

**F60 — Held-out set grown 30 -> 88 clean tasks. No method shows transfer.**

Construction: 70 new tasks from operations chosen by frequency in the 752-file
corpus, list fixed before any score seen. All 100 validated; then **12 were
dropped** because their non-scaffold operations all appeared in training. 88
remain, **0 leaks**.

| model | held-out (88) |
|---|---|
| base | 27.3% |
| best SFT (broad coverage + replay) | 30.7% |
| CPT only | 35.2% |

| comparison | Δ | p |
|---|---|---|
| base -> CPT | +8.0 | 0.265 |
| best SFT -> CPT | **+4.5** | **0.627** |
| base -> best SFT | +3.4 | 0.728 |

**None significant. All three models are statistically indistinguishable on
operations they were never trained on.**

**F61 — The n=30 result was noise, and we would have published it.**
At 30 tasks CPT beat the best SFT model by **+13.3pp**; at 88 the same
comparison is **+4.5pp, p = 0.627**. The effect shrank by two thirds when the
instrument got better. Every comparison shows heavy churn (29-38 tasks changing
outcome) with little net movement — the signature of noise.
This is the **fourth** time sample size has been the binding constraint
(F10, F12, O12, F59) and the second time a promising result evaporated under a
larger eval (see also F40/F41).

**The honest project conclusion:**
- **In-distribution, training works and works well**: dev 20.0% -> 60.0%,
  p = 0.000 vs the best prompt.
- **Neither SFT nor CPT demonstrates transfer** to unseen polars operations at
  this scale.
- **Both cause real forgetting**: SFT+replay 83.3% -> 53.3%; CPT 83.3% -> 58.3%
  (p = 0.000), and replay beyond 25% does not help.

**What this does NOT establish:** that CPT cannot work. 2.4M tokens is small
(real runs use 50M-1B), the run stopped with loss still falling, the learning
rate was not swept, and the corpus is the library's own repo rather than
third-party application code. Any of those could hide a real effect. The claim
is "no transfer demonstrated at this scale", not "CPT does not transfer".

---

## 5. Decisions

**D1 — Target library is polars.** Chosen on measured evidence (F1), not
argument. Real gap, fast CPU verifier (12 tasks in 5s), auditable by reading
tables in/out, and a messy real corpus for the curation stack.

**D2 — Task shape:** instruction + input DataFrame → Python code assigning
`result`. `result` must be a plain Python object.

**D3 — Scoring is execution accuracy.** Run the code, compare `result` to the
expected value. Never compare code as text.

**D4 — Always test whether prompting closes the gap before fine-tuning.**
The baseline is the best *simple* approach, not the first one tried.

**D33 — Re-check held-out validity before EVERY training stage.** A set is
held out relative to one training run. F54 caught this for CPT before any
compute was spent; the same check is due before any future stage.

**D31 — NEVER train on the held-out operations (SFT).** The moment they enter the
generators, `evals/test_tasks.py` stops measuring anything. If they are wanted
in training, a NEW held-out set must be built first, from real-world frequency,
before any score is seen.

**D32 — Every headline reports three numbers**: dev (in-distribution), general
(forgetting), held-out (transfer). Reporting the dev number alone would claim a
3x improvement that does not survive contact with unseen operations.

**D29 — Arm 1 is reported as a TRADE, never as a bare win.** Any headline
carries both numbers: polars +25.0, general -62.5. Reporting only the first
would be the single most dishonest thing this project could do.

**D30 — The fix to try is replay, not early stopping.** Mix general Python
examples into the SFT set so the model is reminded that not everything is a
DataFrame. Needs general Python *training* data distinct from
`evals/general_tasks.py`, and the decontaminator extended to screen against the
general eval too — otherwise we fix forgetting by memorising its test.

**D28 — Every eval reports median output length and runaway count**, not just
accuracy. `>400 chars` flags a generation that never stopped. Added after F29,
where accuracy alone said "better" and the model was looping.

**D26 — MLX bf16 is the single reference runtime for all local measurement.**
Removes the framework confound entirely. ollama numbers are historical.
Quantization becomes a deliberate, measured decision at serving time (Phase 7)
rather than an invisible property of whatever tool we happened to use.

**D27 — Train from bf16, not 4-bit.** Starting from a base that is 10 points
worse would handicap every arm and confuse the ablation. Quantize afterwards
and measure the regression — which is Phase 7 done properly, with a
pre-quantization number to compare against.

**D24 — Train with `mlx_lm lora`.** It is what unsloth delegates to on Apple
Silicon anyway, it is Apple's own supported path, and it has `--mask-prompt`.
Going through it directly removes a layer of indirection when something breaks.

**D25 — Train on prompt v1, not v3.** Baking the answer format into the weights
is precisely how you stop paying for few-shot examples on every call (F16). The
honest comparison becomes *best prompted baseline (v3, 35.4%)* vs *trained model
on the cheap prompt (v1)* — a win there is a win on quality and cost together.
Prompt text comes from `prompts.py` at both training and eval time so the two
cannot drift apart (train/serve skew).

**D22 — Separate venvs for eval and training.** `.venv` (polars + anthropic,
~30 MB) and `.venv-train` (torch stack, 1.7 GB). A broken training install
cannot stop the eval running, and the split is already in place for when
training moves to a rented GPU box.

**D23 — Task sets are pluggable.** `--tasks dev|general` on both `run_eval` and
`validate`; each task module carries its own `PREAMBLE`, and result files record
which set was used so dev and general runs cannot be compared by accident.

**D20 — Contamination requires same data AND same answer.** Identical solution
code is not contamination — generic idioms are the thing being taught. The
verbatim-solution signal survives only behind `use_code_signal=True`, for
screening a scraped corpus where no `expected` exists to compare.

**D21 — Generate optimistically, verify ruthlessly.** Templates may be wrong;
every candidate is executed and dropped unless its solution reproduces the
independently-computed `expected`. Proven to work by deliberate sabotage:
flipped operators, stale APIs and wrong return types are all caught.

**D18 — Decontamination keys on data + answer, never on prose.**
`curation/decontaminate.py`. Thresholds are tuned to catch everything at the
cost of discarding some good data: a missed contamination silently inflates
every published number, a false alarm costs one training example.

**D19 — MinHash is built but not used for the SFT screen.** At 48 dev tasks vs
a few thousand candidates, exact Jaccard is simpler and more accurate. MinHash
earns its place on the CPT corpus in Phase 5, where the corpus is large enough
that comparing 128-integer signatures beats comparing documents. Do not reach
for the scalable tool before you have the scale.

**D16 — Claude generates the SFT data.** Proven competent on every task type
we measure (F14). Outputs still get execution-verified — a 100% score on 48
tasks is not a promise about 5,000.

**D17 — Frame the benchmark as a small-model diagnostic, not a frontier
benchmark.** It saturates at the top (F15) and pretending otherwise would be
dishonest. Revisit adding a harder tier before release.

**D15 — Train the 1.5B, not the 3B.** Evidence-based (F13): no detectable
quality difference, and the 1.5B is faster, cheaper and fits Metal training
more comfortably.

**D13 — v3 (few-shot) is the baseline prompt.** Best point estimate, direction
consistent across categories, and the uncertainty is recorded rather than
hidden. Prompt search stops; returns are now smaller than our resolution.

**D14 — Prompt engineering against the dev set is overfitting too.** The v4
cheatsheet was written by reading dev-set failures. Any gain it showed would be
partly memorisation of our own eval. Same disease as training on the test set —
it must be checked against the held-out set built from real GitHub code.

**D11 — Every result file records a task-set hash.** If tasks change, old
scores stop being comparable and `--list` says so. Without this you see a
number move and believe you caused it.

**D12 — Report a paired comparison, not two accuracies.** Which tasks flipped,
in which direction, and whether that split is distinguishable from chance.
Two percentages side by side invite a conclusion the data does not support.

**D5 — Tasks are generated by Claude, then verified by execution, then
hand-audited.** Hand-writing 120 tasks is not a good use of time; hand-*checking*
them is, and takes about an hour because you read tables, not code.

**D6 — Hand-audit the eval set; derive the training set.** Standard practice.

**D7 — Stratify results by error type** (F3): pandas-interference, stale API,
output convention, genuinely hard. Four numbers, not one.

---

## 5b. The training ablation (Phase 6 plan)

**Why order matters:** training is sequential and later stages partially
overwrite earlier ones, so **whatever runs last dominates the final behaviour**.
CPT grades every token ("predict the next word") and shifts *knowledge*; SFT
grades only the answer half and shifts *behaviour and format*.

Four arms, same eval after each:

| # | Arm | Prediction | What it tests |
|---|---|---|---|
| 0 | baseline (no training) | 35.4% | already measured |
| 1 | **SFT only** | large jump; `output_convention` near 100% | how far format + examples alone get us |
| 2 | **CPT only** | may score *below* baseline | knowledge without format — expect the model to continue text rather than answer |
| 3 | **CPT → SFT** | best, *if F2 (pandas prior) is right* | **the headline claim** |
| 4 | SFT → CPT | worse than arm 3, format degrades | included only to demonstrate why order matters |

**The gap between arm 1 and arm 3 is the paper.** If CPT adds little, that is a
genuine and publishable negative result about when continued pretraining earns
its cost.

Practical notes:
- **Replay** — mixing a little of the previous stage's data into the current
  stage reduces catastrophic forgetting. Worth trying if arm 3 disappoints.
- **LoRA layout** — merge the CPT adapter, then train a *fresh* adapter for
  SFT. Keeps the stages cleanly separated, which an ablation needs.
- **Guard against forgetting** — need a small general coding eval alongside
  ours (O9), or we cannot tell "better at polars" from "worse at everything".

---

## 6. Open questions

- **O1 — Base model.** qwen2.5-coder:1.5b used for the probe; not yet committed
  as the training target. Consider 3B.
- **O2b — CPT corpus VOLUME.** 0.32M tokens now, ~1.2M available from the
  polars repo. Small for CPT; state the ambiguity of a null result up front.
- **O3 — ANSWERED (F55).** Text-mode screening built; corpus clean.
- **O4 — PARTLY ANSWERED.** Corpus is pola-rs/polars, MIT, used for frequency
  analysis and CPT. Provenance recorded per-file in data/real/corpus.jsonl.
  Still to decide: whether to redistribute any of it in the released dataset.
- **O5 — RESOLVED.** Yes, via MLX-LM (F25). Train with `mlx_lm lora` (D24).
- **O10 — RESOLVED.** See F27. Cause was quantization scheme, not runtime.
- **O12 — RESOLVED.** Grown to 60; the deficit is real (F42).
- **O15 — ANSWERED (F60/F61).** Grown to 88; the difference was noise.
- **O2 — ANSWERED (F50).** Held-out set built; fine-tuning shows no transfer.
- **O13 — ANSWERED (F46).** 13 -> 28 families gave +21.7pp; interpretation
  partly confounded by alignment (F48).
- **O14 — ANSWERED (F47).** 25% is enough; 50% adds nothing. For the residual
  30-point general deficit the lever is training steps / learning rate / LoRA
  rank, not more replay.
- **O11 — ANSWERED (F52).** Adapter costs ~12-15%; fusing recovers it.
  Still unmeasured: speed of the *quantized* fused model (needs convert+measure).
- **O9 — RESOLVED.** `evals/general_tasks.py`, baseline 79.2% (F23).
- **O6 — Claude failed `string_filter`** in the probe; cause unknown. Run
  `--show-failures` once to rule out a harness bug.

---

## 7. Lessons from the false starts

We tried job-posting extraction, then PromQL, before landing here. Both dead
ends taught something worth keeping:

- **Build the verifier before the data, and attack it before trusting it.**
  Our first comparison scored *every* wrong answer as correct whenever a query
  returned nothing. The dangerous eval bug is not the one that crashes — it's
  the one that silently reports success.
- **Found data often doesn't fit the task you imagined.** Alert descriptions
  looked like free labels; measuring showed 96% omitted the time window and 78%
  omitted the threshold. Count before assuming.
- **Near-duplicates break random train/test splits.** The same alert repeated
  across exporters with one word changed. Will matter again for a code corpus —
  that's what minhash dedup in Phase 5 is for.
- **A check that answers a nearby question is not evidence.**
  `torch.cuda.is_available() == False` describes torch, not Unsloth (F4).
- **Pick a task you can audit without learning a new language.** PromQL failed
  this and it killed the direction. polars passes: you read the input table and
  the answer, never the code.

---

## 8. Glossary

Moved to [GLOSSARY.md](GLOSSARY.md) — plain-language definitions of every term
used here, grouped by area, with the ones we actually hit marked **[used]**.
