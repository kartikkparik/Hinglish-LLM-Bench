# Hinglish-LLM-Bench

**Evaluating Small Language Models on Code-Mixed Hinglish Reasoning & Instruction Following**

[![Dataset](https://img.shields.io/badge/dataset-150%20items-blue)]()
[![Models](https://img.shields.io/badge/models-SmollM2--1.7B%20%7C%20Qwen2.5--1.5B-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Abstract

Code-mixed language — particularly Romanized Hinglish, the informal blend of
Hindi and English written in Latin script — remains a blind spot for
instruction-tuned small language models. We introduce **Hinglish-LLM-Bench**,
a 150-item evaluation suite spanning three task families (multi-step
arithmetic/logical reasoning, reading comprehension, and intent/sarcasm
classification) crossed with three Code-Mixing Index (CMI) tiers of increasing
Hindi density. Each item ships with a gold answer, human-readable reasoning
steps, and linguistic metadata. We evaluate `SmolLM2-1.7B-Instruct` and
`Qwen2.5-1.5B-Instruct` on Exact/Normalized Match accuracy, semantic cosine
similarity, token-level latency, and a CMI robustness drop metric, and publish
an error taxonomy covering idiom mistranslation, negation/scope confusion,
entity/number hallucination, and language drift.

> **Status:** GPU evaluation complete (NVIDIA T4, Colab). See
> [Results](#results) for final metrics.

---

## Dataset

`data/hinglish_bench_150.jsonl` — one JSON object per line:

| Field | Description |
|---|---|
| `id` | `HLLB-001` … `HLLB-150` |
| `category` | `reasoning` · `comprehension` · `intent_sarcasm` (50 each) |
| `cmi_tier` | `low` (~15% Hindi) · `medium` (~40%) · `high` (~70% + idioms) (50 each) |
| `prompt` | Code-mixed evaluation prompt |
| `reference_answer` | Gold answer |
| `reasoning_steps` | Human-readable solution trace |
| `metadata` | `hindi_token_pct_est`, `script`, `idioms_used`, `difficulty` |

Regenerate deterministically (seed = 42):

```bash
python scripts/generate_dataset.py
```

## Methodology

- **Models:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`
- **Generation:** greedy decoding, chat template, ≤160 new tokens, fp16 on CUDA
- **Metrics:**
  - *Exact Match (EM)* / *Normalized Match* — after text normalization
    (lowercase, punctuation/article stripping)
  - *Semantic Cosine Similarity* — `all-MiniLM-L6-v2` embeddings of
    prediction vs. reference
  - *Latency* — ms per generated token
  - *CMI Robustness Drop* — `acc(low) − acc(high)`; positive ⇒ degradation
    under heavier code-mixing
- **Error taxonomy** — rule-based classifier over failed items into four
  categories; rendered as charts by `src/error_analysis.py`.

## Google Colab — 1-click run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kartikkparik/Hinglish-LLM-Bench/blob/main/notebooks/colab_eval.ipynb)

Or manually:

```python
# In a Colab notebook cell:
!git clone https://github.com/kartikkparik/Hinglish-LLM-Bench.git
%cd Hinglish-LLM-Bench
!pip install -r requirements.txt
!python src/evaluate.py            # ~10–20 min on a T4 for both models
!python src/error_analysis.py      # writes assets/*.png
!git add results assets && git commit -m "gpu results" && git push
```

Then open `docs/index.html` (or enable **GitHub Pages** on `docs/`) for the
interactive dashboard.

## Repository layout

```
├── data/hinglish_bench_150.jsonl   # the benchmark
├── scripts/generate_dataset.py     # deterministic dataset generator
├── src/evaluate.py                 # GPU evaluation pipeline
├── src/metrics.py                  # pure-python scoring utils
├── src/error_analysis.py           # taxonomy + charts
├── docs/                           # GitHub-Pages dashboard
├── tests/test_pipeline.py          # CPU-only unit tests
├── results/                        # eval_results.json + summary_metrics.csv
└── assets/                         # generated PNG charts
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Runs fully on CPU (no model weights required).

## Results

Evaluation completed on a single NVIDIA T4 (Google Colab), fp16, greedy
decoding, 150 items per model.

| Artifact | Path | Status |
|---|---|---|
| Per-item predictions | `results/eval_results.json` | ✅ committed |
| Aggregate metrics | `results/summary_metrics.csv` | ✅ committed |
| CMI degradation chart | `assets/cmi_degradation.png` | ✅ committed |
| Error taxonomy chart | `assets/error_taxonomy.png` | ✅ committed |

### Aggregate metrics

| Model | N | EM Acc | NMatch Acc | Cosine Sim | ms/token | CMI Drop |
|---|---|---|---|---|---|---|
| SmolLM2-1.7B-Instruct | 150 | 0.0533 | 0.0933 | 0.3281 | 13.199 | 0.0000 |
| Qwen2.5-1.5B-Instruct | 150 | 0.1267 | 0.2200 | 0.4036 | 12.631 | 0.1000 |

**Observations.** Both sub-2B models struggle with code-mixed Hinglish:
normalized-match accuracy stays below 22%. `Qwen2.5-1.5B-Instruct` leads on
every metric — roughly **2.4×** the EM accuracy and **+0.076** higher mean
cosine similarity — while maintaining comparable latency (~12.6–13.2 ms/token).
The CMI robustness drop is small for both models (0.00 / 0.10), suggesting
errors are distributed across tiers rather than concentrated at high code-mixing;
per-tier breakdowns are in `results/eval_results.json` and the dashboard.

## Citation

```bibtex
@misc{hinglishllmbench2026,
  title  = {Hinglish-LLM-Bench: Evaluating Small Language Models on
            Code-Mixed Hinglish Reasoning and Instruction Following},
  author = {Parik, Kartik},
  year   = {2026},
  howpublished = {\url{https://github.com/kartikkparik/Hinglish-LLM-Bench}}
}
```

## License

MIT
