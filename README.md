# text2polars

Teaching a small open-weight model to write **polars** code — and building the
benchmark that proves whether it worked.

Small language models are confidently bad at polars. They write *pandas*
instead: `sort_values`, `fillna`, `tolist`, `groupby`. This project measures
that gap, then tries to close it with continued pretraining, supervised
fine-tuning and preference optimization — measuring after every stage.

**Current status:** gap confirmed, eval built. See [NOTES.md](NOTES.md) for
findings, decisions and the phase plan.

Baselines on the 48-task dev set:

| model | prompt | overall | eval time |
|---|---|---|---|
| claude-opus-5 | v3 few-shot | 100.0% | paid |
| **Qwen2.5-Coder-1.5B + our LoRA** | **v1 plain** | **47.9%** | **51s** |
| Qwen2.5-Coder-1.5B (bf16) | v3 few-shot | 31.2% | 54s |
| Qwen2.5-Coder-1.5B (bf16) | v1 plain | 22.9% | 49s |

Measured on 120 polars tasks and 60 general-Python tasks:

| | polars | general Python |
|---|---|---|
| base model | 20.0% | 83.3% |
| SFT, polars data only | 33.3% | **10.0%** ← forgot Python |
| **SFT + 25% replay** | **38.3%** | **41.7%** |

**Training on polars alone doubled polars accuracy and destroyed general
ability.** The model began writing `df['words'].value_counts()` for a plain
list of words — reaching for a DataFrame that does not exist. It happened in
5 of 24 general tasks; the base model never did it once.

**Mixing 25% ordinary-Python examples back in helped a lot but did not fix it**
— general 10.0% → 41.7% (p = 0.000), still far below the base's 83.3%
(p = 0.000). The polars-in-Python contamination disappeared entirely, but
general ability did not return.

**And the polars gain is smaller than it first appeared.** On an earlier
48-task eval, SFT looked like +25 points. Splitting the 120 tasks by whether
they predate the training-data design:

| gain over base | tasks the training covered | operations it never saw |
|---|---|---|
| SFT | +25.0% | **+5.6%** |
| SFT + replay | +27.1% | **+12.5%** |

Not contamination — no eval example leaked. **Operation coverage**: the model
learns the operations it is shown and generalises weakly past them. Against the
best prompt, the overall gain is not statistically established (p = 0.203).

Two evals — one for the target skill, one for what we were *not* optimising —
and expanding both is what turned an apparent win into an honest result.

**The thesis:** can a 1.5B running locally reach ~85% of a frontier model's
accuracy on a library neither was trained on — at 6x the speed and no cost?

Three findings shape the project:

- **Prompting improves but does not close the gap** (~35% vs Claude's ~92%).
- **Doubling model size buys nothing** — 1.5B and 3B are indistinguishable
  (p = 1.0). The failure is stale knowledge, not capacity, which is what makes
  continued pretraining the right intervention.
- **The ceiling is 100%** — Claude solves every task, so the whole gap is
  closeable. It also means the benchmark saturates at the frontier: this is a
  diagnostic for small models, not a frontier benchmark.

Most failures don't run at all: the model writes pandas methods (`sort_values`,
`fillna`, `groupby`) that simply don't exist in polars.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# a small local model to test against (needs the Ollama app running)
ollama pull qwen2.5-coder:1.5b

# for the frontier-model reference
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the eval

```bash
python -m evals.validate                      # prove every task is well-formed
python -m evals.validate --review stale_api   # read tasks as a human
python -m evals.run_eval --model ollama:qwen2.5-coder:1.5b
python -m evals.run_eval --list               # every past run, side by side
```

## Train

```bash
bash train/run_sft.sh              # arm 1: SFT only, ~700 iters, ~13 min on an M3 Pro
bash train/run_sft.sh 20 smoke     # quick smoke test

# score the adapter
.venv-train/bin/python -m evals.run_eval \
  --model "mlx:mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16@adapters/sft-v1" --prompt v1
```

Training runs locally on Apple Silicon via `mlx_lm lora` — no GPU rental needed
for a 1.5B with LoRA (peak memory ~6.5 GB, 0.34% of weights trained).

**Watch the eval, not the loss.** The validation split comes from the same
template generator as training, so falling val loss mostly measures "did it
memorise my templates". The dev set is the real test.

## Layout

```
evals/
  dev_tasks.py      48 polars tasks, 12 per category, independently validated
  general_tasks.py  24 ordinary-Python tasks — the forgetting guard
  validate.py       prove tasks are solvable and self-consistent; --review for humans
  run_eval.py       score a model, report per category, save comparable results
  compare.py        paired comparison of two runs, with a significance test
execution.py        run generated code in a subprocess, compare the result
prompts.py          prompt variants v1-v4
curation/           decontamination (data+answer fingerprints, MinHash)
data_gen/           template families + the generate->verify->screen pipeline
train/run_sft.sh    LoRA fine-tune via mlx_lm, with prompt masking
adapters/           trained LoRA adapters (small, shareable)
data/sft/           2,850 verified training examples
probe/              the original 12-task gap probe (kept for reference)
results/            one JSON per run, stamped with a task-set hash
NOTES.md            findings, decisions, open questions
GLOSSARY.md         plain-language definitions of the jargon
```

## Reading a task

Every task is checkable without knowing polars — you read the input table and
the expected answer:

```python
instruction: "Keep only rows where age is greater than 30, return the name column as a list."
setup:       df = pl.DataFrame({'name': ['alice','bob','cara'], 'age': [35, 20, 41]})
expected:    ['alice', 'cara']
```

That property is deliberate. An eval you cannot personally verify is worthless.

## Safety note

The harness executes code written by a language model, in a subprocess with a
timeout. That is fine for a personal experiment on your own machine with tasks
you can read. It is **not** fine for anything public or automated at scale —
production systems run generated code in a container with no network and no
filesystem access.
