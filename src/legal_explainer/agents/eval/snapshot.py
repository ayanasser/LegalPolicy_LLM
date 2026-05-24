"""Live snapshot of an in-progress eval run.

Reads a streaming `predictions_<system>.jsonl.partial` (or the final `.json`)
and prints aggregate metrics — no LLM calls, no network, safe to run while
the main eval is still going.

Usage (from project root, PYTHONPATH=src):

  # Snapshot the orchestrated run currently in flight
  python -m legal_explainer.agents.eval.snapshot --system orchestrated

  # Or point at a specific file
  python -m legal_explainer.agents.eval.snapshot --file reports/agent_eval/predictions_orchestrated.jsonl.partial
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from legal_explainer.agents.config import EVAL_OUTPUT_DIR


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("predictions", [])
    preds: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))


def _pct(num: int, denom: int) -> str:
    return f"{num}/{denom} ({num / denom:.1%})" if denom else "n/a"


def report(preds: list[dict[str, Any]]) -> None:
    if not preds:
        print("No predictions yet.")
        return

    n = len(preds)
    errors = [p for p in preds if p.get("error")]
    successful = [p for p in preds if not p.get("error") and p.get("prediction")]

    print(f"\n══ SNAPSHOT: {n} cases processed ══")
    print(f"   {len(successful)} succeeded, {len(errors)} errored")

    if errors:
        _section("Errors")
        err_counter: Counter[str] = Counter()
        for e in errors:
            # Keep just the first sentence of the error for grouping.
            msg = (e.get("error") or "").split("\n")[0][:80]
            err_counter[msg] += 1
        for msg, count in err_counter.most_common():
            print(f"  ×{count:>3}  {msg}")

    # ── Retrieval hit rate ────────────────────────────────────────────────────
    hits = [p for p in successful if p.get("retrieval_hit") is not None]
    if hits:
        _section("Retrieval hit rate (cited the gold article?)")
        n_hit = sum(1 for p in hits if p["retrieval_hit"])
        print(f"  overall: {_pct(n_hit, len(hits))}")

        by_lang: dict[str, list[bool]] = defaultdict(list)
        for p in hits:
            by_lang[p["language"]].append(p["retrieval_hit"])
        for lang, vals in sorted(by_lang.items()):
            print(f"  {lang:>4}: {_pct(sum(vals), len(vals))}")

    # ── Path distribution (orchestrated only) ─────────────────────────────────
    paths = Counter(p.get("path") for p in successful if p.get("path"))
    if paths:
        _section("Path distribution")
        for path, count in paths.most_common():
            print(f"  {path:<10} {count:>4} ({count / len(successful):.1%})")

    # ── Latency ──────────────────────────────────────────────────────────────
    durations = [p["duration_ms"] for p in successful if p.get("duration_ms")]
    if durations:
        _section("Latency")
        dur_sorted = sorted(durations)
        print(f"  min:    {dur_sorted[0] / 1000:.1f}s")
        print(f"  p50:    {dur_sorted[len(dur_sorted) // 2] / 1000:.1f}s")
        print(f"  mean:   {statistics.mean(durations) / 1000:.1f}s")
        if len(dur_sorted) >= 10:
            print(f"  p90:    {dur_sorted[int(0.9 * len(dur_sorted))] / 1000:.1f}s")
        print(f"  max:    {dur_sorted[-1] / 1000:.1f}s")

    # ── Cost ─────────────────────────────────────────────────────────────────
    costs = [p["cost_usd"] for p in successful if p.get("cost_usd") is not None]
    if costs:
        _section("Cost")
        print(f"  total so far:  ${sum(costs):.2f}")
        print(f"  mean / query:  ${statistics.mean(costs):.4f}")
        # Project to full set if we know it
        print(
            f"  projection to 424:  ${statistics.mean(costs) * 424:.2f} "
            f"({n}/{424} = {n / 4.24:.0f}% done)"
        )

    # ── Per-kind breakdown ────────────────────────────────────────────────────
    if successful and successful[0].get("kind"):
        _section("Per-kind breakdown")
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in successful:
            by_kind[p["kind"]].append(p)
        for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            group_hits = [p["retrieval_hit"] for p in group if p.get("retrieval_hit") is not None]
            hit_str = _pct(sum(group_hits), len(group_hits)) if group_hits else "no article_key"
            avg_dur = statistics.mean(p["duration_ms"] for p in group if p.get("duration_ms"))
            print(f"  {kind:<14} n={len(group):>3}  hit={hit_str:<18}  avg_dur={avg_dur / 1000:.1f}s")

    # ── Refusals — what triggered them ────────────────────────────────────────
    refusals = [p for p in successful if p.get("path") == "refused"]
    if refusals:
        _section(f"Refusals ({len(refusals)})")
        for p in refusals[:5]:
            print(f"  case {p['case_index']:>3}: {p['user_query'][:80]!r}")

    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--system",
        choices=["orchestrated", "baseline"],
        default="orchestrated",
        help="Which run to snapshot. Used to find the default path.",
    )
    p.add_argument(
        "--file",
        type=Path,
        help="Explicit path to a predictions file (.jsonl.partial or .json). "
        "Overrides --system.",
    )
    args = p.parse_args()

    if args.file:
        path = args.file
    else:
        # Prefer the streaming file if it exists, otherwise the final JSON.
        partial = EVAL_OUTPUT_DIR / f"predictions_{args.system}.jsonl.partial"
        final = EVAL_OUTPUT_DIR / f"predictions_{args.system}.json"
        if partial.exists():
            path = partial
        elif final.exists():
            path = final
        else:
            print(f"No predictions file found for system={args.system}")
            print(f"  looked for: {partial}")
            print(f"  and:        {final}")
            return

    print(f"Reading {path}")
    preds = _load(path)
    report(preds)


if __name__ == "__main__":
    main()