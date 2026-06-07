"""Dataset loaders for RAG evaluation.

Gold sets are the bilingual question CSVs (general-public + lawyer-framed); each
row carries a `legal_reference` we parse into the gold article number. The corpus
JSON (`data/orig_data.json`) supplies the gold article *text* used as the
closed-book context for the fine-tuned model.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .text_utils import detect_language, parse_reference_number

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "orig_data.json"


@dataclass
class GoldRow:
    id: str
    question: str
    gold_answer: str
    gold_article: int | None
    language: str = ""
    direction: str = ""  # "forward" | "reverse" | "" (from article_lookup_golden.csv)

    def __post_init__(self) -> None:
        if not self.language:
            self.language = detect_language(self.question)


def load_gold_csv(path: str | Path, limit: int = 0) -> list[GoldRow]:
    """Read a gold CSV into GoldRow records.

    Handles both schemas:
      * general/lawyer sets:   id, question, answer, legal_reference
      * article_lookup_golden: id, question, answer, legal_reference, direction, language
    Extra columns are ignored; the `language` column (if present) overrides
    auto-detection."""
    rows: list[GoldRow] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            q = (r.get("question") or "").strip()
            if not q:
                continue
            rows.append(GoldRow(
                id=str(r.get("id") or i),
                question=q,
                gold_answer=(r.get("answer") or "").strip(),
                gold_article=parse_reference_number(r.get("legal_reference")),
                language=(r.get("language") or "").strip().lower(),
                direction=(r.get("direction") or "").strip().lower(),
            ))
    return rows[:limit] if limit else rows


def load_article_texts(corpus_path: str | Path = DEFAULT_CORPUS) -> dict[int, str]:
    """Map article number → bilingual text (AR + EN) from the corpus JSON."""
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, str] = {}
    import re
    for key, value in data.items():
        if not key.startswith("Article") or not isinstance(value, dict):
            continue
        m = re.search(r"(\d+)", key)
        if not m:
            continue
        num = int(m.group(1))
        ar = (value.get("arabic") or "").strip()
        en = (value.get("english") or "").strip()
        parts = [p for p in (en, ar) if p]
        out[num] = "\n".join(parts)
    return out
