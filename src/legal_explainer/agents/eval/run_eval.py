"""Run the orchestrator (or the baseline) against an existing eval JSONL.

Reads cases from data/qa_pairs_raft_val.jsonl (or any compatible file) and
writes predictions in a shape compatible with the repo's existing
judge_predictions_*.json format, so the rest of the eval harness
(`scripts/closed_book_recall_eval.py`) can score them as-is.

Usage (from project root, after `PYTHONPATH=src`):

  # Smoke test — 21 cases, sequential
  python -m legal_explainer.agents.eval.run_eval --system orchestrated --n 21 --concurrency 1

  # Full sweep — all 424 RAFT cases, 4-way parallel, silent flow logs
  python -m legal_explainer.agents.eval.run_eval --system orchestrated --concurrency 4 --quiet

Also computes retrieval_hit@k against the gold `article_key` field — a metric
the existing harness doesn't have but is critical for a RAG agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from legal_explainer.agents.config import EVAL_DATA_DIR, EVAL_OUTPUT_DIR

DEFAULT_INPUT = EVAL_DATA_DIR / "qa_pairs_raft_val.jsonl"


def _load_cases(
    path: Path, n: int | None, shuffle: bool, seed: int
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            user_msg = next(
                (m for m in d["messages"] if m.get("role") == "user"), None
            )
            gold_msg = next(
                (m for m in d["messages"] if m.get("role") == "assistant"), None
            )
            if not user_msg or not gold_msg:
                continue
            cases.append(
                {
                    "user_query": user_msg["content"],
                    "gold_answer": gold_msg["content"],
                    "language": d.get("language", "en"),
                    "kind": d.get("kind", "explanation"),
                    "article_key": d.get("article_key"),
                }
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(cases)
    if n is not None:
        cases = cases[:n]
    return cases


_ARTICLE_RE = re.compile(
    r"(?ix)(?:article|art\.?|المادة|مادة)\s*(?:no\.?\s*)?(\d+)"
)


def _retrieval_hit(answer: str, gold_article_key: str | None) -> bool | None:
    """Did the answer cite the gold article number anywhere?
    Returns None if the gold lacks an article_key (refusal-style cases)."""
    if not gold_article_key or not answer:
        return None
    gold_num = _ARTICLE_RE.search(gold_article_key)
    if not gold_num:
        return None
    target = gold_num.group(1)
    return any(m.group(1) == target for m in _ARTICLE_RE.finditer(answer))


async def _run_one(
    system: str, engine: str, case_index: int, case: dict[str, Any]
) -> dict[str, Any]:
    """Run a single case. Returns a prediction dict suitable for the output JSON.

    `engine` selects the orchestrator implementation:
      - "langgraph" (v2, default): LangGraph state machine + LLM-based router
      - "legacy" (v1): rule-based router + if/else dispatch in plain Python

    Imports are inside the function so --help / import-time errors don't crash.
    """
    if system == "baseline":
        from legal_explainer.agents.orchestrator import run_baseline
        runner = run_baseline
    elif engine == "langgraph":
        from legal_explainer.agents.orchestrator_langgraph import run_orchestrated_lg
        runner = run_orchestrated_lg
    else:
        from legal_explainer.agents.orchestrator import run_orchestrated
        runner = run_orchestrated

    try:
        result = await runner(case["user_query"])
    except Exception as e:
        return {
            "case_index": case_index,
            "user_query": case["user_query"],
            "gold_answer": case["gold_answer"],
            "language": case["language"],
            "kind": case["kind"],
            "article_key": case["article_key"],
            "prediction": None,
            "error": str(e),
        }

    hit = _retrieval_hit(result.answer, case["article_key"])
    return {
        "case_index": case_index,
        "user_query": case["user_query"],
        "gold_answer": case["gold_answer"],
        "language": case["language"],
        "kind": case["kind"],
        "article_key": case["article_key"],
        "prediction": result.answer,
        "path": result.path,
        "complexity": result.complexity,
        "safety_verdict": result.safety_verdict,
        "duration_ms": result.duration_ms,
        "cost_usd": result.total_cost_usd,
        "retrieval_hit": hit,
        "query_id": result.query_id,
    }


async def _bounded(
    sem: asyncio.Semaphore,
    coro_factory,
    *args,
) -> Any:
    async with sem:
        return await coro_factory(*args)


async def main_async(args: argparse.Namespace) -> None:
    cases = _load_cases(args.input, args.n, args.shuffle, args.seed)
    engine_label = args.engine if args.system == "orchestrated" else "n/a"
    print(
        f"Loaded {len(cases)} cases from {args.input.name}  "
        f"(system={args.system}, engine={engine_label}, "
        f"concurrency={args.concurrency}, shuffle={args.shuffle})"
    )

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_OUTPUT_DIR / f"predictions_{args.system}.json"
    streaming_path = out_path.with_suffix(".jsonl.partial")
    # Wipe the streaming buffer from previous runs.
    if streaming_path.exists():
        streaming_path.unlink()

    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    completed_lock = asyncio.Lock()
    t_total = time.time()

    async def _runner(i: int, case: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        async with sem:
            pred = await _run_one(args.system, args.engine, i, case)
        async with completed_lock:
            completed += 1
            done_n = completed
            # Stream-append for crash safety.
            with open(streaming_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        # Print progress (out-of-order, but tagged with case index).
        hit_str = (
            "hit" if pred.get("retrieval_hit") is True
            else "miss" if pred.get("retrieval_hit") is False
            else "n/a"
        )
        err = pred.get("error")
        status = f"ERROR: {err[:60]}" if err else (
            f"path={pred.get('path')} {hit_str} {pred.get('duration_ms', 0) / 1000:.1f}s"
        )
        print(
            f"  [{done_n:>3}/{len(cases)}] case {i:>3} {case['language']:>2} "
            f"{case['kind']:<14} | {status}"
        )
        return pred

    predictions = await asyncio.gather(
        *[_runner(i, c) for i, c in enumerate(cases)]
    )
    # Re-sort by original case index for stable downstream consumption.
    predictions.sort(key=lambda p: p["case_index"])

    elapsed = time.time() - t_total

    # Aggregate metrics
    path_counts: Counter[str] = Counter(
        p.get("path", "n/a") for p in predictions if p.get("prediction") is not None
    )
    hits = [p["retrieval_hit"] for p in predictions if p.get("retrieval_hit") is not None]
    errors = sum(1 for p in predictions if p.get("error"))

    summary = {
        "system": args.system,
        "n_cases": len(cases),
        "n_errors": errors,
        "elapsed_s": round(elapsed, 1),
        "concurrency": args.concurrency,
        "path_distribution": dict(path_counts),
        "retrieval_hit_rate": (sum(hits) / len(hits)) if hits else None,
        "retrieval_hit_count": f"{sum(hits)}/{len(hits)}" if hits else "n/a",
        "total_cost_usd": round(
            sum(p.get("cost_usd", 0) or 0 for p in predictions), 4
        ),
        "predictions": predictions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Streaming buffer is no longer needed once final file is written.
    if streaming_path.exists():
        streaming_path.unlink()

    print(f"\nWrote {len(predictions)} predictions → {out_path}")
    print(f"  errors:             {errors}")
    print(f"  path distribution:  {dict(path_counts)}")
    if hits:
        print(f"  retrieval_hit:      {sum(hits)}/{len(hits)} ({sum(hits) / len(hits):.1%})")
    print(f"  total cost:         ${summary['total_cost_usd']:.2f}")
    print(f"  elapsed:            {elapsed:.1f}s  ({elapsed / len(cases):.1f}s/case avg)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--system",
        choices=["orchestrated", "baseline"],
        default="orchestrated",
        help="Which pipeline to run.",
    )
    p.add_argument(
        "--engine",
        choices=["langgraph", "legacy"],
        default="langgraph",
        help=(
            "Orchestrator implementation. 'langgraph' (default, v2) uses the "
            "LangGraph state machine with LLM-based routing. 'legacy' (v1) "
            "uses the plain-Python if/else dispatch with rule-based routing. "
            "Only applies when --system=orchestrated."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Eval JSONL with messages[]/language/kind/article_key.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of cases to run (default: all). For smoke tests use --n 21.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help=(
            "How many queries to run in parallel. Default 4. "
            "Each in-flight query spawns ~2-3 Claude Agent SDK subprocesses, "
            "so raise carefully. >8 has been observed to crash the CLI transport."
        ),
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before slicing --n (for representative sub-samples).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (used only with --shuffle).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Silence the per-step flow logger (FLOW_LOG_LEVEL=silent).",
    )
    args = p.parse_args()

    if args.quiet:
        os.environ["FLOW_LOG_LEVEL"] = "silent"

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
