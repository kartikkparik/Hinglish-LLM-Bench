#!/usr/bin/env bash
# Mirror results/, assets/, and data/ into docs/ so the GitHub Pages site
# (which serves docs/ as web root) can fetch them at ./results/ etc.
# Run after src/evaluate.py + src/error_analysis.py on the GPU machine.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p docs/results docs/assets docs/data
cp results/eval_results.json results/summary_metrics.csv docs/results/
cp assets/cmi_degradation.png assets/error_taxonomy.png docs/assets/
cp data/hinglish_bench_150.jsonl docs/data/

echo "docs/ synced:"
ls docs/results docs/assets docs/data
