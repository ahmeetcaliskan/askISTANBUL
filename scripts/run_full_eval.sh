#!/usr/bin/env bash
# Full evaluation deliverables for AskIstanbul (proposal §3.7 / §3.11).
#   1. Retrieval ablation sweeps (k, encoder, chunk size/overlap) — no LLM.
#   2. Full 4-condition comparison on all 85 QA items (dense/bm25/rerank/no-rag).
#   3. Error-analysis report on the dense-rag condition.
#
# Usage:  bash scripts/run_full_eval.sh [client_type]
#   client_type: ollama (default, local/free) | openrouter (API, faster)
set -uo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
CLIENT="${1:-ollama}"
TS="$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo " AskIstanbul full evaluation — client=$CLIENT  ts=$TS"
echo "================================================================"

echo
echo ">>> [1/3] Retrieval ablation sweeps (k / encoder / chunk) ..."
$PY -m askistanbul.eval.ablation --sweep all \
    --out "results/ablation_full_${TS}.json"

echo
echo ">>> [2/3] Full 4-condition comparison (n=85) ..."
$PY -m askistanbul.eval.evaluator --conditions all --n 85 \
    --client-type "$CLIENT" \
    --out "results/eval_baseline_${TS}.json"

echo
echo ">>> [3/3] Error analysis (dense-rag) ..."
$PY -m askistanbul.eval.error_analysis "results/eval_baseline_${TS}.json" \
    --condition dense-rag \
    --out "results/error_analysis.md"

echo
echo "DONE. Artifacts:"
echo "  results/ablation_full_${TS}.json"
echo "  results/eval_baseline_${TS}.json"
echo "  results/error_analysis.md"
