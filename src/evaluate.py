#!/usr/bin/env python3
"""GPU-ready evaluation pipeline for Hinglish-LLM-Bench.

Evaluates instruction-tuned LLMs on the 150-item code-mixed Hinglish bench.
Designed for a single GPU (e.g. Google Colab T4, 16 GB):

    pip install -r requirements.txt
    python src/evaluate.py

Models evaluated (override with --models):
    HuggingFaceTB/SmolLM2-1.7B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct

Automatically uses CUDA + fp16 when available; falls back to CPU fp32.
Outputs:
    results/eval_results.json   per-item predictions & scores
    results/summary_metrics.csv per-model aggregate metrics
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA_PATH = os.path.join(ROOT, "data", "hinglish_bench_150.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question directly and "
    "concisely. The user may write in English, Hindi, or code-mixed Hinglish. "
    "For classification tasks output only the label. For QA output only the "
    "answer span. For math/logic problems output only the final number."
)


def load_dataset(path=DATA_PATH):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_messages(item):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]


def generate_for_model(model_id, items, batch_size=4, max_new_tokens=160,
                       device=None):
    """Run generation for one model; returns per-item result dicts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    model.eval()

    results = []
    texts = [build_messages(it) for it in items]

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        gen_tokens = int((out.shape[1] - enc["input_ids"].shape[1])
                         * out.shape[0])

        decoded = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for item, text in zip(items[start:start + batch_size], decoded):
            pred_answer = metrics.extract_final_answer(text)
            results.append({
                "id": item["id"],
                "category": item["category"],
                "cmi_tier": item["cmi_tier"],
                "prompt": item["prompt"],
                "reference_answer": item["reference_answer"],
                "raw_completion": text.strip(),
                "predicted_answer": pred_answer,
                "em": metrics.exact_match(text, item["reference_answer"]),
                "nmatch": metrics.normalized_match(text, item["reference_answer"]),
                "gen_ms": round(gen_ms / len(batch), 3),
                "gen_tokens": gen_tokens // len(batch),
            })
    return results, model_id


def add_semantic_similarity(results, model_id="all-MiniLM-L6-v2"):
    """Cosine similarity between predicted and reference answers."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    encoder = SentenceTransformer(model_id)
    preds = [r["predicted_answer"] for r in results]
    refs = [r["reference_answer"] for r in results]
    pe = encoder.encode(preds, normalize_embeddings=True, show_progress_bar=False)
    re_ = encoder.encode(refs, normalize_embeddings=True, show_progress_bar=False)
    sims = (pe * re_).sum(axis=1)
    for r, s in zip(results, sims):
        r["cosine_sim"] = round(float(s), 4)
    return results


def evaluate(models, data_path=DATA_PATH, out_dir=RESULTS_DIR,
             batch_size=4, max_new_tokens=160):
    os.makedirs(out_dir, exist_ok=True)
    items = load_dataset(data_path)

    all_results = {}
    summary_rows = []

    for model_id in models:
        print(f"\n=== Evaluating {model_id} ===")
        res, _ = generate_for_model(model_id, items, batch_size=batch_size,
                                    max_new_tokens=max_new_tokens)
        res = add_semantic_similarity(res)
        all_results[model_id] = res

        em = metrics.accuracy([r["em"] for r in res])
        nm = metrics.accuracy([r["nmatch"] for r in res])
        cos = metrics.mean([r.get("cosine_sim") for r in res])
        tot_ms = sum(r["gen_ms"] for r in res)
        tot_tok = sum(r["gen_tokens"] for r in res)
        mpt = metrics.ms_per_token(tot_ms, tot_tok)
        # per-item dicts need 'correct' key for cmi drop
        cmi_in = [{"correct": r["nmatch"], "cmi_tier": r["cmi_tier"]} for r in res]
        drop = metrics.cmi_robustness_drop(cmi_in)

        summary_rows.append({
            "model": model_id,
            "n_items": len(res),
            "exact_match_acc": round(em, 4),
            "normalized_match_acc": round(nm, 4),
            "mean_cosine_sim": round(cos, 4),
            "ms_per_token": mpt,
            "cmi_robustness_drop": drop,
        })
        print(f"  EM={em:.3f} NMatch={nm:.3f} Cosine={cos:.3f} "
              f"ms/tok={mpt} CMI_drop={drop}")

    # Save outputs
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"models": all_results, "summary": summary_rows}, f,
                  ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"\nSaved: {json_path}\nSaved: {csv_path}")
    return all_results, summary_rows


def main():
    ap = argparse.ArgumentParser(description="Hinglish-LLM-Bench evaluator")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="HuggingFace model ids to evaluate")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    import torch
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    evaluate(args.models, data_path=args.data, out_dir=args.out,
             batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
