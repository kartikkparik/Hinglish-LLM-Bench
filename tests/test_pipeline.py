#!/usr/bin/env python3
"""Lightweight unit tests for Hinglish-LLM-Bench.

Runs on CPU with only the stdlib — no torch/transformers needed.
Verifies:
  1. The dataset JSONL has exactly 150 items with the required schema and
     balanced category / CMI-tier distribution.
  2. Scoring math (normalize/exact/normalized match, cosine, CMI drop,
     ms/token) behaves correctly on dummy inputs.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import metrics  # noqa: E402

DATA = os.path.join(ROOT, "data", "hinglish_bench_150.jsonl")

REQUIRED_KEYS = {"id", "category", "cmi_tier", "prompt",
                 "reference_answer", "reasoning_steps", "metadata"}
META_KEYS = {"hindi_token_pct_est", "script", "idioms_used", "difficulty"}
CATEGORIES = {"reasoning", "comprehension", "intent_sarcasm"}
TIERS = {"low", "medium", "high"}


def load_rows():
    with open(DATA, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestDatasetSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows()

    def test_row_count(self):
        self.assertEqual(len(self.rows), 150)

    def test_required_fields(self):
        for r in self.rows:
            self.assertTrue(REQUIRED_KEYS <= set(r), r.get("id"))
            self.assertTrue(META_KEYS <= set(r["metadata"]), r.get("id"))
            self.assertIsInstance(r["prompt"], str)
            self.assertTrue(r["prompt"].strip())
            self.assertTrue(r["reference_answer"].strip())
            self.assertIsInstance(r["reasoning_steps"], list)
            self.assertTrue(r["reasoning_steps"])
            self.assertEqual(r["metadata"]["script"], "romanized")

    def test_unique_ids(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(set(ids)), 150)

    def test_category_balance(self):
        from collections import Counter
        c = Counter(r["category"] for r in self.rows)
        self.assertEqual(c, {"reasoning": 50, "comprehension": 50,
                             "intent_sarcasm": 50})

    def test_tier_balance(self):
        from collections import Counter
        c = Counter(r["cmi_tier"] for r in self.rows)
        self.assertEqual(c, {"low": 50, "medium": 50, "high": 50})

    def test_hindi_pct_ranges(self):
        for r in self.rows:
            p = r["metadata"]["hindi_token_pct_est"]
            t = r["cmi_tier"]
            if t == "low":
                self.assertLessEqual(p, 25)
            elif t == "medium":
                self.assertTrue(25 < p < 55)
            else:
                self.assertGreaterEqual(p, 55)

    def test_intent_labels_valid(self):
        valid = {"praise", "complaint", "request", "sarcasm", "neutral_info"}
        for r in self.rows:
            if r["category"] == "intent_sarcasm":
                self.assertIn(r["reference_answer"], valid, r["id"])


class TestScoringMath(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(metrics.normalize_text("The Answer, Is 42!"),
                         "answer is 42")
        self.assertEqual(metrics.normalize_text("  Hello   World  "),
                         "hello world")

    def test_extract_final_answer(self):
        self.assertEqual(metrics.extract_final_answer("blah\nAnswer: 900"), "900")
        self.assertEqual(metrics.extract_final_answer("**sarcasm**"), "sarcasm")
        self.assertEqual(metrics.extract_final_answer("line1\nline2"), "line2")

    def test_exact_match(self):
        self.assertTrue(metrics.exact_match("The answer is 900.", "900") is False)
        self.assertTrue(metrics.exact_match("Answer: 900", "900"))
        self.assertFalse(metrics.exact_match("Answer: 901", "900"))
        self.assertTrue(metrics.exact_match("sarcasm", "Sarcasm"))

    def test_normalized_match_containment(self):
        self.assertTrue(metrics.normalized_match("The answer is 900 rupees",
                                                 "900"))
        self.assertTrue(metrics.normalized_match("900", "900"))
        self.assertFalse(metrics.normalized_match("450", "900"))

    def test_accuracy_and_mean(self):
        self.assertAlmostEqual(metrics.accuracy([True, False, True]), 2 / 3)
        self.assertEqual(metrics.accuracy([]), 0.0)
        self.assertAlmostEqual(metrics.mean([1.0, 3.0]), 2.0)
        self.assertEqual(metrics.mean([]), 0.0)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(
            metrics.cosine_similarity([1, 1], [1, 1]), 1.0)

    def test_cmi_robustness_drop(self):
        rows = ([{"correct": True, "cmi_tier": "low"}] * 4 +
                [{"correct": False, "cmi_tier": "high"}] * 4)
        self.assertAlmostEqual(metrics.cmi_robustness_drop(rows), 1.0)
        # medium items are ignored in the low-vs-high drop
        rows2 = rows + [{"correct": True, "cmi_tier": "medium"}]
        self.assertAlmostEqual(metrics.cmi_robustness_drop(rows2), 1.0)

    def test_ms_per_token(self):
        self.assertAlmostEqual(metrics.ms_per_token(1000.0, 200), 5.0)
        self.assertEqual(metrics.ms_per_token(100.0, 0), 0.0)


class TestErrorClassifier(unittest.TestCase):
    def _item(self, **kw):
        base = {"prompt": "What is 2+2?", "predicted_answer": "4",
                "reference_answer": "4", "cmi_tier": "low", "raw_completion": ""}
        base.update(kw)
        return base

    def test_classifier_importable_and_labels(self):
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import error_analysis  # noqa: E402
        self.assertEqual(len(error_analysis.ERROR_CATEGORIES), 4)
        # entity/number mismatch -> hallucination
        r = self._item(predicted_answer="7", reference_answer="4")
        self.assertEqual(error_analysis.classify_error(r),
                         "Entity/Number Hallucination")
        # high-CMI ascii-only prediction -> language drift
        r2 = self._item(cmi_tier="high", predicted_answer="The answer is four.")
        self.assertEqual(error_analysis.classify_error(r2), "Language Drift")


if __name__ == "__main__":
    unittest.main()
