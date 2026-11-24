"""Pure-Python scoring utilities for Hinglish-LLM-Bench.

This module intentionally has NO heavy dependencies (torch/transformers) so it
can be imported and unit-tested on CPU-only machines. `src/evaluate.py` uses
these functions after generation.
"""

import math
import re
import string

# ---------------------------------------------------------------------------
# Text normalization & exact / normalized match
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower().strip()
    # remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def extract_final_answer(text: str) -> str:
    """Best-effort extraction of a final answer from a model completion.

    Handles common patterns like 'Answer: X', '**X**', or a trailing line.
    Falls back to the last non-empty line.
    """
    if not text:
        return ""
    t = text.strip()
    # explicit answer markers
    for pat in (r"answer\s*[:\-]\s*(.+)", r"final answer\s*[:\-]\s*(.+)",
                r"\*\*(.+?)\*\*", r"`(.+?)`"):
        m = re.findall(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m[-1].strip().split("\n")[0].strip()
    # last non-empty line
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    return lines[-1] if lines else t


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_text(extract_final_answer(prediction)) == normalize_text(reference)


def normalized_match(prediction: str, reference: str) -> bool:
    """Normalized match: containment-aware for short answers."""
    p = normalize_text(extract_final_answer(prediction))
    r = normalize_text(reference)
    if not p or not r:
        return False
    return p == r or r in p or p in r


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def accuracy(scores: list) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s) / len(scores)


def mean(values: list) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Cosine similarity between two vectors (pure python, for tests)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cmi_robustness_drop(per_item_results: list) -> float:
    """Accuracy drop from lowest to highest CMI tier.

    per_item_results: iterable of dicts with keys 'correct' (bool) and
    'cmi_tier' ('low'|'medium'|'high'). Returns low_acc - high_acc
    (positive => model degrades on heavier code-mixing).
    """
    by_tier = {"low": [], "medium": [], "high": []}
    for r in per_item_results:
        t = r.get("cmi_tier")
        if t in by_tier:
            by_tier[t].append(bool(r.get("correct")))
    low = accuracy(by_tier["low"])
    high = accuracy(by_tier["high"])
    return round(low - high, 4)


def ms_per_token(total_ms: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return round(total_ms / total_tokens, 3)
