"""Suggested questions for the Unified UI.

Pulls real, in-distribution questions from three sources:
  - data/general_user_legal_questions.csv   (layperson, colloquial Arabic)
  - data/lawyer_llm_solution_questions.csv   (lawyer-framed Arabic)
  - data/qa_pairs_knowledge.jsonl            (knowledge-recall prompts, EN + AR)

Sampling is deterministic (evenly spaced) so the example set is stable across
launches.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from . import config


def _evenly(items: list[str], n: int) -> list[str]:
    items = [i for i in items if i and i.strip()]
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _from_csv(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    rows: list[str] = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").strip()
            if q:
                rows.append(q)
    return _evenly(rows, n)


def _strip_lang_tag(text: str) -> str:
    text = text.strip()
    for tag in ("[AR]", "[EN]", "[ar]", "[en]"):
        if text.startswith(tag):
            return text[len(tag):].strip()
    return text


def _from_knowledge(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    qs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                msgs = rec.get("messages")
                if isinstance(msgs, str):
                    # stored as a python-repr / json string
                    try:
                        msgs = json.loads(msgs)
                    except Exception:
                        msgs = ast.literal_eval(msgs)
                user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                if user:
                    qs.append(_strip_lang_tag(user))
            except Exception:
                continue
    return _evenly(qs, n)


def load_suggestions() -> dict[str, list[str]]:
    """Return grouped suggested questions."""
    return {
        "General public (AR)": _from_csv(config.GENERAL_CSV, 8),
        "Lawyer-framed (AR)": _from_csv(config.LAWYER_CSV, 8),
        "Knowledge recall (EN/AR)": _from_knowledge(config.KNOWLEDGE_JSONL, 8),
    }


def flat_examples() -> list[str]:
    """A flat, de-duplicated list for gr.Examples."""
    out: list[str] = []
    seen: set[str] = set()
    for group in load_suggestions().values():
        for q in group:
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out
