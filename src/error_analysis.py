#!/usr/bin/env python3
"""Error taxonomy & chart generator for Hinglish-LLM-Bench.

Reads results/eval_results.json (produced by src/evaluate.py on GPU) and:

  1. Classifies each failed item into one of four error categories:
       - Idiom/Colloquial Mistranslation
       - Negation & Scope Confusion
       - Entity/Number Hallucination
       - Language Drift
  2. Generates publication-grade PNG charts:
       assets/cmi_degradation.png   accuracy vs CMI tier per model
       assets/error_taxonomy.png    error-category distribution per model

The classifier is rule-based (keyword/heuristic) so it runs anywhere with
just matplotlib + the results JSON — no GPU or model weights needed.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_JSON = os.path.join(ROOT, "results", "eval_results.json")
ASSETS_DIR = os.path.join(ROOT, "assets")

ERROR_CATEGORIES = [
    "Idiom/Colloquial Mistranslation",
    "Negation & Scope Confusion",
    "Entity/Number Hallucination",
    "Language Drift",
]

# ---------------------------------------------------------------------------
# Rule-based error classifier
# ---------------------------------------------------------------------------

_NEGATION_PAT = re.compile(
    r"\b(not|n't|never|no|nahi|nahin|mat|without|cannot|can't|won't|don't|doesn't|"
    r"isn't|aren't|wasn't|weren't|na|nai)\b", re.IGNORECASE)

_IDIOM_PAT = re.compile(
    r"\b(dimaag|aankh|mann|dil|haath|paanv|dil jeet|shabaash|bakwaas|mast|"
    r"swagat|suno|arre|yaar|bhai|zabardast|kamaal|badhiya|mazaa|saaf|seedha|"
    r"phir se|waqt|din|raat)\b", re.IGNORECASE)

_DRIFT_LATIN_ONLY = re.compile(r"^[\x00-\x7F\s]+$")  # pure ASCII => likely English drift


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def classify_error(item):
    """Return one of ERROR_CATEGORIES for a failed result dict."""
    prompt = item.get("prompt", "")
    pred = item.get("predicted_answer", "") or item.get("raw_completion", "")
    ref = item.get("reference_answer", "") or ""

    # 1) Language drift: model answered in a different script/language regime.
    #    Heuristic: high-CMI prompt but prediction is pure ASCII English, or
    #    prediction is very long narrative when a label/number was expected.
    if item.get("cmi_tier") == "high" and pred and _DRIFT_LATIN_ONLY.match(pred):
        return "Language Drift"
    if len(pred.split()) > 60 and len(ref.split()) <= 5:
        return "Language Drift"

    # 2) Negation & scope: prompt or reference contains negation but the
    #    prediction flips/drops it (e.g. answered 'yes' style opposite).
    if _NEGATION_PAT.search(prompt) or _NEGATION_PAT.search(ref):
        p_norm = pred.lower().strip(" .!")
        r_norm = ref.lower().strip(" .!")
        opposites = {"yes": "no", "no": "yes", "true": "false", "false": "true"}
        if p_norm in opposites and opposites.get(p_norm) == r_norm:
            return "Negation & Scope Confusion"
        if _NEGATION_PAT.search(ref) and not _NEGATION_PAT.search(pred):
            return "Negation & Scope Confusion"

    # 3) Entity/number hallucination: numeric/entity mismatch.
    ref_nums = _numbers(ref)
    pred_nums = _numbers(pred)
    if ref_nums and pred_nums and not (ref_nums & pred_nums):
        return "Entity/Number Hallucination"
    # capitalized entity in reference missing from prediction
    ref_ents = set(re.findall(r"\b[A-Z][a-z]+\b", ref))
    if ref_ents and not any(e.lower() in pred.lower() for e in ref_ents):
        return "Entity/Number Hallucination"

    # 4) Idiom/colloquial mistranslation: high-CMI items with idiomatic
    #    prompts where the prediction misses the idiomatic meaning.
    if item.get("cmi_tier") in ("medium", "high") and _IDIOM_PAT.search(prompt):
        return "Idiom/Colloquial Mistranslation"

    # Fallback ordering: numbers first already handled; default to idiom for
    # high-CMI, else negation if cue present, else entity.
    if item.get("cmi_tier") == "high":
        return "Idiom/Colloquial Mistranslation"
    if _NEGATION_PAT.search(prompt):
        return "Negation & Scope Confusion"
    return "Entity/Number Hallucination"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def analyze(results_path=RESULTS_JSON):
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    per_model = {}
    for model_id, items in data["models"].items():
        failures = [r for r in items if not r.get("nmatch", r.get("em"))]
        cats = Counter(classify_error(r) for r in failures)
        # ensure all categories present
        for c in ERROR_CATEGORIES:
            cats.setdefault(c, 0)
        per_model[model_id] = {
            "n_total": len(items),
            "n_failures": len(failures),
            "error_counts": {c: cats[c] for c in ERROR_CATEGORIES},
            "by_tier_acc": tier_accuracy(items),
        }
    return per_model


def tier_accuracy(items):
    by = defaultdict(list)
    for r in items:
        by[r["cmi_tier"]].append(bool(r.get("nmatch", r.get("em"))))
    return {t: round(sum(v) / len(v), 4) if v else 0.0
            for t, v in sorted(by.items())}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("#FAF9F6")
    ax.grid(True, axis="y", color="#E5E2DA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#D6D3C9")
    ax.tick_params(colors="#111827", labelsize=9)
    ax.set_title(title, color="#111827", fontsize=12, fontweight="bold",
                 pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color="#374151", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#374151", fontsize=9)


def plot_cmi_degradation(per_model, out_dir=ASSETS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor("#FAF9F6")

    tiers = ["low", "medium", "high"]
    palette = ["#2563EB", "#D97706", "#DC2626", "#059669", "#7C3AED"]
    x = range(len(tiers))
    width = 0.8 / max(len(per_model), 1)

    for i, (model, stats) in enumerate(per_model.items()):
        short = model.split("/")[-1]
        accs = [stats["by_tier_acc"].get(t, 0.0) for t in tiers]
        bars = ax.bar([xi + i * width - (len(per_model) - 1) * width / 2
                       for xi in x], accs, width=width,
                      label=short, color=palette[i % len(palette)],
                      edgecolor="white", linewidth=0.5)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#111827")

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Low CMI\n(~15% Hindi)", "Medium CMI\n(~40% Hindi)",
                        "High CMI\n(~70% Hindi)"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized-match accuracy", color="#374151", fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor="#111827")
    _style_ax(ax, "Accuracy vs Code-Mixing Index (CMI) Tier")
    fig.tight_layout()
    path = os.path.join(out_dir, "cmi_degradation.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_error_taxonomy(per_model, out_dir=ASSETS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    models = list(per_model.keys())
    short_names = [m.split("/")[-1] for m in models]
    data = np.array([[per_model[m]["error_counts"][c] for c in ERROR_CATEGORIES]
                     for m in models], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    fig.patch.set_facecolor("#FAF9F6")
    y = np.arange(len(models))
    left = np.zeros(len(models))
    colors = ["#2563EB", "#DC2626", "#D97706", "#059669"]

    for i, cat in enumerate(ERROR_CATEGORIES):
        vals = data[:, i]
        ax.barh(y, vals, left=left, color=colors[i], label=cat,
                edgecolor="white", linewidth=0.5, height=0.55)
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v > 0:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9, color="#111827")
    ax.invert_yaxis()
    ax.set_xlabel("Number of failed items", color="#374151", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor="#111827",
              loc="lower right")
    _style_ax(ax, "Error Taxonomy Distribution", xlabel=None)
    fig.tight_layout()
    path = os.path.join(out_dir, "error_taxonomy.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_JSON)
    ap.add_argument("--out", default=ASSETS_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"No results found at {args.results}. Run src/evaluate.py first.")
        sys.exit(1)

    per_model = analyze(args.results)
    p1 = plot_cmi_degradation(per_model, args.out)
    p2 = plot_error_taxonomy(per_model, args.out)
    print("Saved:", p1)
    print("Saved:", p2)


if __name__ == "__main__":
    main()
