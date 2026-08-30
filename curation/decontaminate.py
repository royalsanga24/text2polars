"""Detect eval examples leaking into training data.

Contamination is the most dangerous bug in a training pipeline because it makes
results look BETTER. A bug that lowers your score gets investigated within the
hour; one that raises it gets published.

Two comparison methods live here, and knowing when to use which is the point:

  exact_jaccard  — compares two texts directly. Simple, precise, and O(n*m).
                   Fine for 48 dev tasks vs a few thousand candidates.

  MinHash        — estimates the same number from a small fixed-size signature.
                   Slower and less accurate on small data. Indispensable when
                   the corpus is millions of documents, because you can compare
                   signatures instead of full texts.

We validate the MinHash implementation against exact Jaccard below. Do not
trust an approximation you have not checked against the thing it approximates.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

# A large prime, for the (a*x + b) mod p hash family used by MinHash.
_PRIME = (1 << 61) - 1


# ------------------------------------------------------------------ normalise

def normalize(text: str) -> str:
    """Strip away differences that don't change meaning.

    Without this, `df['age']` and `df["age"]` look like different text and a
    renamed-quote copy of a dev task sails through undetected.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"['\"`]", "", text)          # quote style is not meaning
    text = re.sub(r"[^\w\s]", " ", text)        # punctuation is not meaning
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, k: int = 5) -> Set[str]:
    """Overlapping k-word windows.

    "keep rows where age is over 30" with k=5 gives
        {"keep rows where age is", "rows where age is over", ...}

    Why windows rather than individual words: word overlap alone would call
    any two polars instructions similar, because they all say "return the
    column as a list". Windows capture phrasing, which is what actually gets
    copied. k is a real knob — smaller catches more paraphrase and more false
    alarms.
    """
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def exact_jaccard(a: Set[str], b: Set[str]) -> float:
    """|intersection| / |union|. 1.0 identical, 0.0 nothing in common."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# -------------------------------------------------------------------- MinHash

def _hash_shingle(s: str) -> int:
    # Python's hash() is randomised per process — signatures would not be
    # reproducible across runs. That matters: you must be able to recompute a
    # corpus signature months later and get the same answer.
    import hashlib
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def _permutations(num_perm: int, seed: int = 0) -> List[Tuple[int, int]]:
    import random
    rng = random.Random(seed)
    return [(rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(num_perm)]


def minhash_signature(sh: Set[str], perms: Sequence[Tuple[int, int]]) -> List[int]:
    """One small fixed-size fingerprint per document.

    The trick: for each of N random hash permutations, keep only the SMALLEST
    hashed shingle. Two documents that share many shingles are likely to share
    the same minimum under any given permutation. So the fraction of matching
    signature positions estimates the Jaccard similarity — without ever
    comparing the documents themselves.

    That's why it scales: 128 integers per document, regardless of length.
    """
    if not sh:
        return [0] * len(perms)
    hashed = [_hash_shingle(s) for s in sh]
    return [min((a * h + b) % _PRIME for h in hashed) for a, b in perms]


def minhash_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Estimate Jaccard from two signatures: the fraction of matching slots."""
    if not sig_a:
        return 0.0
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


# --------------------------------------------------------------- the detector

def task_fingerprint(setup: str, expected) -> str:
    """What actually identifies a task: its data and its answer.

    NOT its prose. Two tasks over the same DataFrame with the same expected
    output are the same task however differently they are worded. Two tasks
    over different data are different tasks however similar the wording —
    the model cannot memorise an answer it never saw, so it has to compute,
    which is the skill we are measuring.

    Calibrating on instruction text failed badly (see curation/calibrate.py):
    these instructions are short and templated, so unrelated tasks share
    phrasing and the distributions do not separate.
    """
    return f"{normalize(setup)} || {normalize(repr(expected))}"


@dataclass
class Hit:
    candidate_index: int
    dev_id: str
    score: float
    reason: str


class Decontaminator:
    """Holds the eval set; screens candidate training examples against it."""

    def __init__(self, dev_tasks, k: int = 4, setup_threshold: float = 0.8,
                 num_perm: int = 128):
        self.k = k
        self.setup_threshold = setup_threshold
        self.perms = _permutations(num_perm)
        self.dev = []
        for t in dev_tasks:
            self.dev.append(dict(
                id=t["id"],
                fingerprint=task_fingerprint(t["setup"], t["expected"]),
                expected=t["expected"],
                setup_shingles=shingles(t["setup"], k),
                setup_norm=normalize(t["setup"]),
                solution_norm=normalize(t.get("solution", "")),
            ))
        self._by_fp = {d["fingerprint"]: d["id"] for d in self.dev}

    def check(self, instruction: str, setup: str, code: str = "",
              expected=None, use_code_signal: bool = False) -> Optional[Hit]:
        """Return the worst contamination hit for one candidate, or None.

        Contamination is SAME DATA **and** SAME ANSWER. Both halves matter:

        - same data, different answer  -> a different question. The model
          cannot memorise an answer it never saw.
        - same answer, different data  -> coincidence. Plenty of tasks answer
          [1, 2, 3].
        - identical solution CODE      -> not contamination at all. Generic
          idioms like `df.join(df2, on='id', how='inner')` are what we are
          trying to teach. An earlier version treated this as a leak and threw
          away 134 perfectly good training examples, every one a false alarm.

        `use_code_signal` re-enables the verbatim-solution check for screening a
        SCRAPED corpus, where there is no `expected` to compare against and a
        long distinctive solution appearing word for word is worth a look.
        """
        # 1. Exact fingerprint: same data, same answer. Fatal, unambiguous.
        if expected is not None:
            fp = task_fingerprint(setup, expected)
            if fp in self._by_fp:
                return Hit(-1, self._by_fp[fp], 1.0, "identical data and answer")

        setup_sh = shingles(setup, self.k)
        setup_norm = normalize(setup)
        code_norm = normalize(code)

        worst = None
        for d in self.dev:
            # 2. Near-identical data AND the same answer. Catches a reworded
            #    or slightly perturbed copy. Both halves required.
            score, reason = 0.0, ""
            same_answer = expected is not None and normalize(repr(expected)) == \
                normalize(repr(d["expected"]))
            if same_answer:
                if setup_norm and setup_norm == d["setup_norm"]:
                    score, reason = 1.0, "identical data and answer"
                else:
                    j = exact_jaccard(setup_sh, d["setup_shingles"])
                    if j >= self.setup_threshold:
                        score, reason = j, "near-identical data, same answer"

            # 3. Corpus screening only — see docstring.
            if use_code_signal and code_norm and len(d["solution_norm"]) > 60 \
                    and d["solution_norm"] in code_norm:
                score, reason = 1.0, "reference solution appears verbatim"

            if reason and (worst is None or score > worst.score):
                worst = Hit(-1, d["id"], score, reason)
        return worst

    def screen(self, candidates: Iterable[dict]) -> Tuple[List[dict], List[Hit]]:
        """Split candidates into (clean, contaminated)."""
        clean, hits = [], []
        for i, c in enumerate(candidates):
            hit = self.check(c.get("instruction", ""), c.get("setup", ""),
                             c.get("solution", ""), c.get("expected"))
            if hit:
                hit.candidate_index = i
                hits.append(hit)
            else:
                clean.append(c)
        return clean, hits
