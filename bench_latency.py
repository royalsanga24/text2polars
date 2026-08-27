"""Measure inference speed properly (NOTES O11).

The earlier 4bit/8bit/bf16 timings were polluted by multi-GB downloads and said
nothing about throughput. This warms up first, then measures prompt processing
and token generation separately, on identical prompts.

    .venv-train/bin/python bench_latency.py
"""
import statistics as st
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "probe")
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from evals.dev_tasks import TASKS
from execution import PREAMBLE
from prompts import PROMPTS

BASE = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"
CONFIGS = [
    ("base bf16",            BASE, None),
    ("base bf16 + adapter",  BASE, "adapters/broad25"),
    ("fused bf16",           "models/broad25-fused", None),
]
N = 12
sampler = make_sampler(temp=0.0)
prompts = [PROMPTS["v1"].format(preamble=PREAMBLE, setup=t["setup"],
                                instruction=t["instruction"]) for t in TASKS[:N]]

print(f"{'config':<24}{'prompt tok/s':>14}{'gen tok/s':>12}{'gen toks':>10}{'s/task':>9}")
print("-" * 70)
for label, repo, adapter in CONFIGS:
    try:
        model, tok = load(repo, adapter_path=adapter)
    except Exception as e:
        print(f"{label:<24} LOAD FAILED: {type(e).__name__}: {str(e)[:40]}")
        continue
    if tok.eos_token_id is not None:
        tok.eos_token_ids = set(getattr(tok, "eos_token_ids", None) or ()) | {tok.eos_token_id}

    chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False) for p in prompts]
    # warm up — the first call pays lazy-load and kernel-compile costs
    for _ in stream_generate(model, tok, prompt=chat[0], max_tokens=8, sampler=sampler):
        pass

    pps, gps, ntoks, walls = [], [], [], []
    for c in chat:
        t0 = time.time()
        last = None
        n = 0
        for resp in stream_generate(model, tok, prompt=c, max_tokens=256, sampler=sampler):
            last = resp
            n += 1
        walls.append(time.time() - t0)
        ntoks.append(n)
        if last is not None:
            pps.append(last.prompt_tps)
            gps.append(last.generation_tps)
    print(f"{label:<24}{st.median(pps):>14.0f}{st.median(gps):>12.1f}"
          f"{st.median(ntoks):>10.0f}{st.median(walls):>9.2f}")
