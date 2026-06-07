"""Evaluation runner — simple, flat output.

One run = one system on one gold set, producing exactly **one file**:
    reports/eval/rag/<system>__<dataset>.jsonl     (everything: answers + metrics)

Plus a single shared table that every run updates in place:
    reports/eval/rag/SUMMARY.md   (one row per run — the file you read)
    reports/eval/rag/SUMMARY.json (same, machine-readable)

Flow (judge = ollama, the default): predict → judge → summarise, all in one go.
Judge = claude-code: predict writes the run file with a `judge_prompt` per row;
Claude Code fills each row's `verdict` in-session; then `--phase report` summarises.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import GoldRow
from .judge import JudgeTask, make_task
from .prompts import JUDGE_SYSTEM, LLM_METRICS
from .scores import deterministic_metrics

DET_NUMERIC = ("citation_accuracy", "has_citation", "retrieval_hit@k",
               "retrieval_mrr", "context_precision@k", "token_overlap_gold")


def run_file(out_dir: Path, system: str, dataset: str) -> Path:
    return Path(out_dir) / f"{system}__{Path(dataset).stem}.jsonl"


# ── JSONL helpers (tolerant of partial/corrupt lines) ─────────────────────────

def _read_jsonl_safe(path: Path, progress=print) -> list[dict]:
    if not path.exists():
        return []
    rows, bad = [], 0
    for l in path.read_text(encoding="utf-8", errors="replace").splitlines():
        l = l.strip().replace("\x00", "")
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        progress(f"[warn] skipped {bad} corrupt line(s) in {path}")
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Phase 1: predict ──────────────────────────────────────────────────────────

def predict_phase(system, rows: list[GoldRow], article_texts: dict[int, str],
                  k: int, path: Path, progress=print) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            pred = system.predict(row.question)
            if pred.closed_book:   # graded against the gold article it should recall
                gold_txt = article_texts.get(row.gold_article or -1, "")
                contexts = [gold_txt] if gold_txt else []
            else:
                contexts = pred.contexts
            det = deterministic_metrics(
                pred.answer, pred.retrieved_articles, row.gold_article, row.gold_answer, k)
            task = make_task(row.id, row.question, pred.answer, contexts,
                             row.gold_answer, row.gold_article, pred.closed_book)
            rec = {
                "id": row.id, "question": row.question, "language": row.language,
                "direction": row.direction, "gold_article": row.gold_article,
                "gold_answer": row.gold_answer, "answer": pred.answer,
                "retrieved_articles": pred.retrieved_articles, "contexts": contexts,
                "closed_book": pred.closed_book, "elapsed_ms": pred.elapsed_ms,
                "error": pred.error, "deterministic": det,
                "judge_prompt": task.user_prompt, "verdict": None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            hit = "✓" if det["retrieval_hit@k"] else "·"
            cite = "✓" if det["citation_accuracy"] else "·"
            err = f"  ERROR: {pred.error}" if pred.error else ""
            progress(f"[{i:>3}/{len(rows)}] id={row.id} hit@{k}={hit} cite={cite} "
                     f"{pred.elapsed_ms}ms{err}")
    progress(f"[predict] wrote {path}")


# ── Phase 2: judge (ollama) ───────────────────────────────────────────────────

def judge_phase_ollama(judge, path: Path, progress=print) -> None:
    rows = _read_jsonl_safe(path, progress)
    for i, r in enumerate(rows, 1):
        task = JudgeTask(row_id=r["id"], system_prompt=JUDGE_SYSTEM,
                         user_prompt=r["judge_prompt"], closed_book=r.get("closed_book", False))
        r["verdict"] = judge.score(task)
        scores = " ".join(f"{m[:4]}={r['verdict'].get(m)}" for m in LLM_METRICS)
        progress(f"[judge {i:>3}/{len(rows)}] id={r['id']} {scores}")
    _write_rows(path, rows)
    progress(f"[judge] updated {path}")


# ── Phase 3: summarise ────────────────────────────────────────────────────────

def _mean(vals: list) -> float | None:
    nums = [float(v) for v in vals if isinstance(v, (int, float, bool))]
    return round(sum(nums) / len(nums), 4) if nums else None


def aggregate(rows: list[dict]) -> dict:
    def over(subset: list[dict]) -> dict:
        det = {m: _mean([r["deterministic"].get(m) for r in subset]) for m in DET_NUMERIC}
        llm = {m: _mean([(r.get("verdict") or {}).get(m) for r in subset]) for m in LLM_METRICS}
        return {**det, **llm}

    return {
        "n_rows": len(rows),
        "n_judged": sum(1 for r in rows if r.get("verdict")),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "mean_elapsed_ms": _mean([r.get("elapsed_ms") for r in rows]),
        "overall": over(rows),
        "by_language": {lg: over([r for r in rows if r.get("language") == lg])
                        for lg in ("en", "ar")
                        if any(r.get("language") == lg for r in rows)},
        "by_direction": {d: over([r for r in rows if r.get("direction") == d])
                         for d in ("forward", "reverse")
                         if any(r.get("direction") == d for r in rows)},
    }


# Columns shown in the consolidated SUMMARY table.
_SUMMARY_COLS = [
    ("faithfulness", "Faith"), ("answer_relevance", "Relev"),
    ("context_precision", "CtxP"), ("context_recall", "CtxR"),
    ("answer_correctness", "Correct"), ("retrieval_hit@k", "Hit@k"),
    ("citation_accuracy", "Cite"),
]


def update_summary(out_dir: Path, system: str, dataset: str, k: int,
                   judge_backend: str, summary: dict) -> Path:
    """Upsert this run's row into the single shared SUMMARY.{json,md}."""
    out_dir = Path(out_dir)
    sj = out_dir / "SUMMARY.json"
    data = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    key = f"{system} · {Path(dataset).stem}"
    o = summary["overall"]
    data[key] = {
        "system": system, "dataset": Path(dataset).stem, "k": k,
        "judge": judge_backend, "n_rows": summary["n_rows"],
        "n_judged": summary["n_judged"], "n_errors": summary["n_errors"],
        **{m: o.get(m) for m, _ in _SUMMARY_COLS},
    }
    sj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    header = "| System | Dataset | N | " + " | ".join(lbl for _, lbl in _SUMMARY_COLS) + " |"
    sep = "|---|---|---:|" + "---:|" * len(_SUMMARY_COLS)
    lines = ["# RAG / answer evaluation — summary", "",
             "Scores are 0–1. `Hit@k` & `Cite` are retrieval/citation; the rest are "
             "LLM-judged (faithfulness, relevance, context precision/recall, correctness).", "",
             header, sep]
    for v in data.values():
        cells = " | ".join(fmt(v.get(m)) for m, _ in _SUMMARY_COLS)
        lines.append(f"| {v['system']} | {v['dataset']} | {v['n_rows']} | {cells} |")
    lines += ["", "_Per-question detail: `reports/eval/rag/<system>__<dataset>.jsonl`._", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "SUMMARY.md"
