---
license: mit
base_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16
library_name: mlx
tags:
- lora
- mlx
- polars
- code-generation
- apple-silicon
datasets:
- royalsanga/text2polars-bench
language:
- en
---

# text2polars-lora-1.5b

A 20 MB LoRA adapter that roughly triples a 1.5B model's accuracy at writing
**polars** — on the operations it was trained on, and **only** those.

> ### Read this before using it
>
> This adapter **damages general Python ability**. On 60 ordinary Python tasks
> with no polars involved, the base model scores **83.3%** and this adapter
> scores **53.3%**. That is a real, statistically significant regression
> (p = 0.000), not measurement noise.
>
> It also shows **no measurable transfer**. On 88 polars operations it was
> never trained on, it scores 30.7% against the base model's 27.3%
> (p = 0.728 — indistinguishable).
>
> Use it if you want better polars on common operations and can accept both.
> Do not use it as a general coding model.

## Results

Measured on [text2polars-bench](https://huggingface.co/datasets/royalsanga/text2polars-bench).
Every comparison is paired, with an exact McNemar test.

| Eval | Base | This adapter | Δ | p |
|---|---|---|---|---|
| polars, in-distribution (120) | 20.0% | **60.0%** | +40.0 | **0.000** |
| polars, held-out operations (88) | 27.3% | 30.7% | +3.4 | 0.728 |
| general Python (60) | **83.3%** | 53.3% | **−30.0** | **0.000** |

Against the best prompt we could write for the base model (few-shot, 25.0%),
the trained model on a *plain* prompt reaches 60.0%, p = 0.000 — so the gain is
not something a better prompt recovers.

Per category on the in-distribution set:

| Category | Base | Adapter |
|---|---|---|
| output convention | 58.3% | 70.0% |
| pandas interference | 16.7% | 70.0% |
| stale API names | 0.0% | 63.3% |
| genuinely hard | 16.7% | 36.7% |

The gain concentrates where the training data pointed. `stale_api` went from
zero; `hard` barely moved. Teaching API names does not teach compositional
reasoning.

## Usage

Needs [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon.

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tok = load(
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
    adapter_path="text2polars-lora-1.5b",   # this repo, downloaded locally
)

# The tokenizer config disagrees with itself: the model emits <|im_end|>
# (151645) but generation halts only on <|endoftext|>. Without this the model
# writes a correct answer, then restarts and runs to max_tokens — a 22x
# slowdown with identical output.
tok.eos_token_ids = set(getattr(tok, "eos_token_ids", None) or ()) | {tok.eos_token_id}

prompt = tok.apply_chat_template([{"role": "user", "content": """import polars as pl

You are given this data:

df = pl.DataFrame({'name': ['x','y','z'], 'score': [7, 12, 9]})

Task: Sort rows by 'score' from highest to lowest and return the 'name' column as a list.

Write Python code that assigns the answer to a variable named `result`.
The data is already defined — do not redefine it. Do not print anything.
Reply with ONLY the code, no explanation, no markdown fences."""}],
    add_generation_prompt=True, tokenize=False)

print(generate(model, tok, prompt=prompt, max_tokens=256,
               sampler=make_sampler(temp=0.0)))
```

**Use the prompt format above.** The adapter was trained with it; a different
framing degrades output. Greedy decoding (`temp=0.0`) keeps results
reproducible.

To remove the ~12% throughput cost of applying an adapter at runtime, fuse it:

```bash
mlx_lm.fuse --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 \
            --adapter-path text2polars-lora-1.5b --save-path fused/
```

Fusing is exact in bf16. **Fusing into a quantized base is not** — it requires
dequantise/requantise, and MLX's default 4-bit costs ~10 points on this task.
Measure after fusing, not before.

## Training

| | |
|---|---|
| Base | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16` |
| Method | LoRA, rank 8, 16 of 28 layers, q/k/v/o + MLP |
| Trainable | 5.28M params (0.34%) |
| Data | 3,605 examples — 75% synthetic polars, **25% ordinary Python replay** |
| Steps | 700, batch 4, lr 1e-4, prompt-masked loss |
| Hardware | Apple M3 Pro, 13 minutes, 6.5 GB peak |

**The 25% replay is not optional.** Without it, general Python collapses to
**10.0%** rather than 53.3%. Raising replay to 50% changes nothing further
(p ≈ 0.8) — it is a threshold, not a dial.

Training data was generated from 28 template families, every example executed
and discarded unless it produced its stated answer, then screened against all
268 benchmark tasks. Zero contamination.

## What this adapter demonstrates

It is published mainly so the claims above can be checked. The interesting
result is the one that did not happen:

**Fine-tuning taught the 47 operations it was shown, taught them well, and
generalised to none beyond them.** In-distribution accuracy tripled; held-out
accuracy did not move. An earlier 30-task held-out set suggested otherwise;
growing it to 88 removed the effect.

Full method, the results that did not survive better measurement, and the three
measurement bugs that nearly produced false numbers:
**https://github.com/royalsanga24/text2polars**

## Limitations

- Apple Silicon / MLX only. Not converted for `transformers` or vLLM.
- Answers are polars-version-specific; trained and validated against 1.43.2.
- Degrades general Python (see above).
- No transfer to unseen operations (see above).
- Trained on synthetic template data, not real analytical workloads.
