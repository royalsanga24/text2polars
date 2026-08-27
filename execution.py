"""Run generated polars code and compare the result. Used by the probe, the
task validator, and (in Phase 3) the eval harness.

SAFETY: this executes code written by a language model. It runs in a separate
process with a timeout, on your machine, with tasks you can read. Fine for a
personal experiment; NOT fine for anything public — production systems run
generated code in a container with no network and no filesystem.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

PREAMBLE = "import polars as pl"

RUNNER = """\
{preamble}
import json, sys
{setup}

{code}

def _coerce(x):
    m = getattr(x, "to_list", None) or getattr(x, "to_dicts", None)
    if m: return m()
    return x

sys.stdout.write("<<<RESULT>>>" + json.dumps(_coerce(result), default=str))
"""


def execute(setup, code, preamble=PREAMBLE, timeout=20):
    """Run code in a fresh process. Returns (ok, value_or_error_string)."""
    script = RUNNER.format(preamble=preamble, setup=textwrap.dedent(setup), code=code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)

    if p.returncode != 0:
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        return False, (lines[-1] if lines else "unknown error")
    if "<<<RESULT>>>" not in p.stdout:
        return False, "no result produced"
    return True, json.loads(p.stdout.split("<<<RESULT>>>", 1)[1])


def matches(got, expected) -> bool:
    """Structural comparison with float tolerance."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return got == expected
    if isinstance(expected, float) or isinstance(got, float):
        try:
            return abs(float(got) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list) and isinstance(got, list):
        return len(got) == len(expected) and all(matches(g, e) for g, e in zip(got, expected))
    if isinstance(expected, dict) and isinstance(got, dict):
        return set(got) == set(expected) and all(matches(got[k], expected[k]) for k in expected)
    return got == expected


# ------------------------------------------------------------ batch execution

BATCH_RUNNER = """\
import json, sys

payload = json.load(sys.stdin)
BASE = {}
exec(payload["preamble"], BASE)          # preamble may be empty (general tasks)

def _coerce(x):
    m = getattr(x, "to_list", None) or getattr(x, "to_dicts", None)
    return m() if m else x

out = []
for item in payload["items"]:
    ns = dict(BASE)                      # exec() with explicit globals sees nothing else
    try:
        exec(item["setup"], ns)
        exec(item["code"], ns)
        out.append([True, _coerce(ns["result"])])
    except Exception as e:
        out.append([False, f"{{type(e).__name__}}: {{e}}"])
sys.stdout.write("<<<RESULT>>>" + json.dumps(out, default=str))
"""


def execute_batch(items, preamble=PREAMBLE, timeout=600):
    """Run many (setup, code) pairs in ONE process. Returns [(ok, value), ...].

    Launching a subprocess per example costs ~0.4s in interpreter start and
    polars import; at 2000 examples that is 15 minutes of pure overhead.
    Amortising it over a batch takes the same work to a few seconds.

    NOTE on safety: this runs the batch in one shared interpreter, so an
    example that corrupts global state can affect later ones (each gets its own
    namespace, which handles the common cases). That trade is acceptable for
    OUR OWN template-generated code. Code written by a MODEL still goes through
    execute() one subprocess at a time — see the safety note at the top.
    """
    if not items:
        return []
    script = BATCH_RUNNER
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path],
                           input=json.dumps({"preamble": preamble, "items": items}),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [(False, "BATCH TIMEOUT")] * len(items)
    finally:
        os.unlink(path)

    if "<<<RESULT>>>" not in p.stdout:
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return [(False, f"batch failed: {err}")] * len(items)
    return [tuple(r) for r in json.loads(p.stdout.split("<<<RESULT>>>", 1)[1])]
