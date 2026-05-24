"""Baseline vs orchestrated comparison report (Epic 7 task 7.5).

Reads two predictions JSON files produced by run_eval.py and prints a side-by-side
table covering latency, cost, path distribution, and retrieval_hit_rate.
Quality scoring (the 5-dim Claude judge) is delegated to the existing harness —
this script just bundles the per-system metrics so you can hand both prediction
files to scripts/closed_book_recall_eval.py and get judge scores too.

Usage:

  python -m legal_explainer.agents.eval.compare_baseline \\
      --baseline reports/agent_eval/predictions_baseline.json \\
      --orchestrated reports/agent_eval/predictions_orchestrated.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _per_system_stats(data: dict[str, Any]) -> dict[str, Any]:
    preds = [p for p in data["predictions"] if p.get("prediction")]
    if not preds:
        return {"n": 0}

    durations = [p["duration_ms"] for p in preds if p.get("duration_ms")]
    costs = [p["cost_usd"] for p in preds if p.get("cost_usd") is not None]
    hits = [p["retrieval_hit"] for p in preds if p.get("retrieval_hit") is not None]

    by_language: dict[str, list[float]] = {}
    for p in preds:
        if p.get("retrieval_hit") is not None:
            by_language.setdefault(p["language"], []).append(float(p["retrieval_hit"]))

    return {
        "n": len(preds),
        "latency_ms_p50": int(statistics.median(durations)) if durations else None,
        "latency_ms_p90": int(sorted(durations)[int(0.9 * len(durations)) - 1]) if len(durations) >= 10 else None,
        "cost_usd_mean": round(statistics.mean(costs), 5) if costs else None,
        "cost_usd_total": round(sum(costs), 4) if costs else None,
        "retrieval_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "retrieval_hit_by_language": {
            lang: round(sum(v) / len(v), 3) for lang, v in by_language.items() if v
        },
        "path_distribution": data.get("path_distribution", {}),
    }


def _print_table(baseline: dict[str, Any], orchestrated: dict[str, Any]) -> None:
    metrics = [
        ("n", "n"),
        ("latency_ms_p50", "latency p50 (ms)"),
        ("latency_ms_p90", "latency p90 (ms)"),
        ("cost_usd_mean", "$ / query (mean)"),
        ("cost_usd_total", "$ total"),
        ("retrieval_hit_rate", "retrieval_hit@k"),
    ]
    print(f"\n{'metric':<24} {'baseline':>14} {'orchestrated':>16}")
    print("-" * 56)
    for key, label in metrics:
        b = baseline.get(key)
        o = orchestrated.get(key)
        print(f"{label:<24} {str(b):>14} {str(o):>16}")
    print()

    if baseline.get("retrieval_hit_by_language") or orchestrated.get(
        "retrieval_hit_by_language"
    ):
        print("retrieval_hit by language:")
        for lang in sorted(
            set(baseline.get("retrieval_hit_by_language", {}))
            | set(orchestrated.get("retrieval_hit_by_language", {}))
        ):
            b = baseline.get("retrieval_hit_by_language", {}).get(lang, "-")
            o = orchestrated.get("retrieval_hit_by_language", {}).get(lang, "-")
            print(f"  {lang:>4}: baseline={b}  orchestrated={o}")
        print()

    if orchestrated.get("path_distribution"):
        print(f"orchestrated path distribution: {orchestrated['path_distribution']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--orchestrated", type=Path, required=True)
    args = p.parse_args()

    baseline_data = _load(args.baseline)
    orchestrated_data = _load(args.orchestrated)

    b_stats = _per_system_stats(baseline_data)
    o_stats = _per_system_stats(orchestrated_data)

    _print_table(b_stats, o_stats)

    print(
        "\nNext: feed both prediction files to the existing Claude judge harness "
        "(scripts/closed_book_recall_eval.py) to add quality scores."
    )


if __name__ == "__main__":
    main()