"""
eval/error_analysis.py
----------------------
Turn an evaluation results JSON (from ``evaluator.py``) into a human-readable
error-analysis report (proposal Week 4 / Expected Outcomes).

It surfaces *where* and *why* the system fails by:
  - breaking every metric down by question ``category`` and ``difficulty``;
  - bucketing each question into failure modes:
      * retrieval miss     — no gold chunk retrieved (mrr == 0)
      * weak retrieval     — precision_at_k below threshold
      * incomplete recall  — recall_at_k below threshold
      * hallucination      — faithfulness below threshold
      * off-topic answer   — answer_relevance below threshold
  - listing the worst concrete examples per failure mode.

Usage:
  python -m askistanbul.eval.error_analysis results/eval_run.json
  python -m askistanbul.eval.error_analysis results/eval_run.json --condition dense-rag
  python -m askistanbul.eval.error_analysis results/eval_run.json --out results/error_analysis.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional

from ..paths import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"

# Failure thresholds — tweak via CLI if needed.
PRECISION_MIN = 0.20
RECALL_MIN = 0.50
FAITH_MIN = 0.70
RELEVANCE_MIN = 0.70

METRICS = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k",
           "faithfulness", "answer_relevance"]


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _safe_mean(values: list) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return mean(nums) if nums else None


def pick_condition(payload: dict, name: Optional[str]) -> dict:
    conds = payload.get("conditions", [])
    if not conds:
        raise SystemExit("No conditions found in results file.")
    if name:
        for c in conds:
            if c.get("condition") == name:
                return c
        raise SystemExit(f"Condition {name!r} not in file. "
                         f"Available: {[c.get('condition') for c in conds]}")
    return conds[0]


def breakdown(rows: list[dict], key: str) -> list[tuple]:
    """Mean of each metric grouped by row[key]."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "unknown"].append(r)
    out = []
    for g, items in sorted(groups.items()):
        means = {m: _safe_mean([it.get(m) for it in items]) for m in METRICS}
        out.append((g, len(items), means))
    return out


def classify(row: dict, thresholds: dict) -> list[str]:
    modes = []
    mrr = row.get("mrr")
    pk = row.get("precision_at_k")
    rk = row.get("recall_at_k")
    faith = row.get("faithfulness")
    rel = row.get("answer_relevance")
    if isinstance(mrr, (int, float)) and mrr == 0.0:
        modes.append("retrieval-miss")
    if isinstance(pk, (int, float)) and pk < thresholds["precision"]:
        modes.append("weak-retrieval")
    if isinstance(rk, (int, float)) and rk < thresholds["recall"]:
        modes.append("incomplete-recall")
    if isinstance(faith, (int, float)) and faith < thresholds["faith"]:
        modes.append("hallucination")
    if isinstance(rel, (int, float)) and rel < thresholds["relevance"]:
        modes.append("off-topic")
    return modes


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report(payload: dict, cond: dict, thresholds: dict, top_n: int) -> str:
    rows = [r for r in cond.get("rows", []) if "error" not in r]
    agg = cond.get("aggregate", {})
    condition = cond.get("condition", "?")
    k = cond.get("k", payload.get("k"))

    lines: list[str] = []
    lines.append(f"# AskIstanbul — Error Analysis")
    lines.append("")
    lines.append(f"- **Source:** evaluation run from `{payload.get('timestamp', '?')}`")
    lines.append(f"- **Condition analysed:** `{condition}`  (k={k}, client="
                 f"`{payload.get('client_type','?')}`)")
    lines.append(f"- **Questions:** {len(rows)} (errored rows excluded)")
    lines.append(f"- **Failure thresholds:** precision<{thresholds['precision']}, "
                 f"recall<{thresholds['recall']}, faithfulness<{thresholds['faith']}, "
                 f"answer_relevance<{thresholds['relevance']}")
    lines.append("")

    # --- Overall aggregate -------------------------------------------------
    lines.append("## Overall metrics")
    lines.append("")
    agg_rows = []
    for m in METRICS:
        stats = agg.get(m, {})
        agg_rows.append([m, _fmt(stats.get("mean")), _fmt(stats.get("min")),
                         _fmt(stats.get("max")), str(stats.get("n", ""))])
    lines.append(md_table(["metric", "mean", "min", "max", "n"], agg_rows))
    lines.append("")

    # --- Breakdown by category --------------------------------------------
    for key, title in (("category", "category"), ("difficulty", "difficulty")):
        lines.append(f"## Metrics by {title}")
        lines.append("")
        brk = breakdown(rows, key)
        table_rows = []
        for g, n, means in brk:
            table_rows.append([str(g), str(n)] + [_fmt(means[m]) for m in METRICS])
        lines.append(md_table([title, "n"] + METRICS, table_rows))
        lines.append("")

    # --- Failure-mode counts ----------------------------------------------
    mode_counts: dict = defaultdict(int)
    mode_examples: dict = defaultdict(list)
    for r in rows:
        for mode in classify(r, thresholds):
            mode_counts[mode] += 1
            mode_examples[mode].append(r)

    lines.append("## Failure modes")
    lines.append("")
    if mode_counts:
        fm_rows = [[mode, str(cnt), f"{cnt/len(rows)*100:.0f}%"]
                   for mode, cnt in sorted(mode_counts.items(), key=lambda x: -x[1])]
        lines.append(md_table(["failure mode", "count", "% of questions"], fm_rows))
    else:
        lines.append("_No questions tripped any failure threshold._")
    lines.append("")

    # --- Worst examples per mode ------------------------------------------
    sort_key = {
        "retrieval-miss":    lambda r: r.get("mrr") or 0,
        "weak-retrieval":    lambda r: r.get("precision_at_k") or 0,
        "incomplete-recall": lambda r: r.get("recall_at_k") or 0,
        "hallucination":     lambda r: r.get("faithfulness") or 0,
        "off-topic":         lambda r: r.get("answer_relevance") or 0,
    }
    lines.append("## Worst concrete cases")
    lines.append("")
    for mode in sorted(mode_examples):
        examples = sorted(mode_examples[mode], key=sort_key[mode])[:top_n]
        if not examples:
            continue
        lines.append(f"### {mode}")
        lines.append("")
        for r in examples:
            lines.append(f"- **Q{r.get('id')}** ({r.get('category')}/{r.get('difficulty')}): "
                         f"{r.get('question')}")
            lines.append(f"  - scores: P={_fmt(r.get('precision_at_k'))} "
                         f"R={_fmt(r.get('recall_at_k'))} MRR={_fmt(r.get('mrr'))} "
                         f"NDCG={_fmt(r.get('ndcg_at_k'))} "
                         f"faith={_fmt(r.get('faithfulness'))} "
                         f"rel={_fmt(r.get('answer_relevance'))}")
            lines.append(f"  - gold: `{r.get('gold_ids')}`")
            lines.append(f"  - retrieved: `{r.get('retrieved_ids')}`")
            ans = (r.get("answer") or "").strip().replace("\n", " ")
            if ans:
                lines.append(f"  - answer: {ans[:300]}{'…' if len(ans) > 300 else ''}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an error-analysis report from eval results.")
    parser.add_argument("results_file", help="Path to an evaluator.py results JSON")
    parser.add_argument("--condition", default=None,
                        help="Which condition to analyse (default: first in file)")
    parser.add_argument("--out", default=None,
                        help="Output markdown path (default: results/error_analysis.md)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many worst examples to list per failure mode")
    parser.add_argument("--precision-min", type=float, default=PRECISION_MIN)
    parser.add_argument("--recall-min", type=float, default=RECALL_MIN)
    parser.add_argument("--faith-min", type=float, default=FAITH_MIN)
    parser.add_argument("--relevance-min", type=float, default=RELEVANCE_MIN)
    args = parser.parse_args()

    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    cond = pick_condition(payload, args.condition)
    thresholds = {
        "precision": args.precision_min,
        "recall": args.recall_min,
        "faith": args.faith_min,
        "relevance": args.relevance_min,
    }
    report = build_report(payload, cond, thresholds, args.top_n)

    out_path = Path(args.out) if args.out else (RESULTS_DIR / "error_analysis.md")
    out_path.write_text(report, encoding="utf-8")
    print(f"Error-analysis report written → {out_path}")


if __name__ == "__main__":
    main()
