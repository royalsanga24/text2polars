"""Ask models to solve the probe tasks, run their code, count what's correct.

    python -m probe.run_probe --model ollama:qwen2.5-coder:1.5b
    python -m probe.run_probe --model claude:claude-opus-5
    python -m probe.run_probe --model ollama:qwen2.5-coder:1.5b --show-failures

This is a miniature version of the eval we build in Phase 3. Same shape:
  ask a model -> run its answer -> compare to the truth -> count.

SAFETY NOTE, and it matters beyond this project:
We execute code a language model wrote. That is genuinely dangerous. Here it
runs in a separate process with a timeout, on your own machine, with tasks you
can read. That is acceptable for a personal experiment and NOT acceptable for
anything public or automated at scale — production systems run generated code
in a container or VM with no network and no filesystem access.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution import execute, matches  # noqa: E402  shared with evals/

from prompts import PROMPTS  # noqa: E402


# ---------------------------------------------------------------- model calls

def ask_ollama(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},   # deterministic: reruns must be comparable
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["message"]["content"]


_MLX_CACHE = {}


def ask_mlx(spec: str, prompt: str) -> str:
    """Run a local MLX model, optionally with a LoRA adapter.

        mlx:mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
        mlx:mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit@adapters/sft-1

    The model is cached across calls — reloading it per task would dominate
    the runtime. Sampling is greedy (temp=0) so reruns are byte-identical,
    exactly as for the ollama and claude backends.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    if spec not in _MLX_CACHE:
        repo, _, adapter = spec.partition("@")
        model, tok = load(repo, adapter_path=adapter or None)
        # The tokenizer config disagrees with itself: `eos_token_id` is 151645
        # (<|im_end|>, what the model actually emits) but the `eos_token_ids`
        # set that generation halts on contains only 151643 (<|endoftext|>).
        # Without this, generation never stops: the model writes the answer,
        # emits <|im_end|>, and starts over until it hits max_tokens — 491
        # tokens instead of 15, a 22x slowdown. See NOTES F35.
        if tok.eos_token_id is not None:
            tok.eos_token_ids = set(getattr(tok, "eos_token_ids", None) or ()) | {tok.eos_token_id}
        _MLX_CACHE[spec] = (model, tok)
    model, tok = _MLX_CACHE[spec]

    # Apply the model's chat template: an instruct model trained with one
    # expects it, and skipping it silently degrades quality.
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)

    out = generate(model, tok, prompt=prompt, max_tokens=512,
                   sampler=make_sampler(temp=0.0), verbose=False)
    # The chat template's end token comes back as literal text
    # ("...to_list()<|im_end|>") and is a syntax error inside the snippet.
    # Cut at the first special token rather than string-replacing a fixed list,
    # so this holds for any model family's template.
    return re.split(r"<\|[^|]*\|>", out)[0].strip()


def ask_claude(model: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_code(text: str) -> str:
    """Pull code out of a model response.

    Models add markdown fences no matter how firmly you ask them not to — and
    some never CLOSE them. The original version required a closing fence, so an
    unclosed block was handed to Python verbatim and failed as a syntax error.
    That mis-scored 81 of 110 CPT failures as model errors when the polars
    inside was correct (NOTES F56). Handle both shapes.
    """
    text = text.strip()
    closed = re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", text, re.S)
    if closed:
        return closed[0].strip()
    # unclosed fence: take everything after the opening one
    m = re.match(r"```(?:[a-zA-Z]*)\n(.*)", text, re.S)
    if m:
        return m.group(1).strip()
    return text


# ------------------------------------------------------------------ execution

# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="ollama:<name> or claude:<model-id>")
    ap.add_argument("--tasks", default="polars")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()

    mod = __import__(f"tasks_{args.tasks}")
    tasks, preamble = mod.TASKS, mod.PREAMBLE

    kind, _, name = args.model.partition(":")
    ask = {"ollama": ask_ollama, "claude": ask_claude}[kind]

    print(f"model: {args.model}   tasks: {args.tasks} ({len(tasks)})   prompt: {args.prompt}\n")
    correct, failures = 0, []
    t0 = time.time()

    for i, t in enumerate(tasks, 1):
        prompt = PROMPTS[args.prompt].format(preamble=preamble, setup=t["setup"],
                               instruction=t["instruction"])
        try:
            code = extract_code(ask(name, prompt))
        except Exception as e:
            code, ok, val = "", False, f"model call failed: {e}"
        else:
            ok, val = execute(t["setup"], code, preamble)

        good = ok and matches(val, t["expected"])
        correct += good
        mark = "PASS" if good else ("WRONG" if ok else "ERROR")
        print(f"  [{i:>2}/{len(tasks)}] {t['id']:<18} {mark}")
        if not good:
            failures.append((t, code, val))

    pct = 100 * correct / len(tasks)
    print(f"\n{'='*54}\n  {args.model}  [prompt {args.prompt}]\n  {correct}/{len(tasks)} correct  ({pct:.0f}%)"
          f"   {time.time()-t0:.0f}s\n{'='*54}")

    if args.show_failures and failures:
        print("\nFAILURES — read these, they are the actual information:\n")
        for t, code, val in failures:
            print(f"--- {t['id']}: {t['instruction']}")
            print(f"    model wrote:  {code.replace(chr(10), ' ; ')[:150]}")
            print(f"    produced:     {val}")
            print(f"    expected:     {t['expected']}\n")


if __name__ == "__main__":
    main()
