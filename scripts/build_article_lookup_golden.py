#!/usr/bin/env python3
"""Build a bilingual golden dataset for pure article-number lookup.

Two directions per article, in Arabic and English:
  forward  : article number -> what the article states
  reverse  : a distinctive quote from the article -> which article number

Source: data/orig_data.json (Egyptian Civil Code, AR/EN parallel).
Output: data/article_lookup_golden.csv  (CSV sample, ~50 articles).

Schema matches the existing golden CSVs (id, question, answer, legal_reference)
plus two extra columns (direction, language) for filtering during eval.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "orig_data.json"
OUT = ROOT / "data" / "article_lookup_golden.csv"

N_ARTICLES = 50          # how many articles to sample
SNIPPET_CHARS = 140      # length of the quoted snippet for reverse questions


def clean(text: str) -> str:
    """Collapse the PDF-extracted whitespace/newlines into a single line."""
    return re.sub(r"\s+", " ", (text or "").strip())


def snippet(text: str, n: int = SNIPPET_CHARS) -> str:
    """First n chars, trimmed back to a word boundary so we don't cut mid-word."""
    t = clean(text)
    if len(t) <= n:
        return t
    cut = t[:n]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " …"


def article_num(key: str) -> int:
    return int(key.split()[-1])


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    arts = [k for k in data if k.startswith("Article")]
    # keep only articles that have BOTH arabic and english so every sampled
    # article yields a full bilingual set of rows
    arts = [k for k in arts if clean(data[k].get("arabic")) and clean(data[k].get("english"))]

    # even, deterministic spread across the whole code
    step = max(1, len(arts) // N_ARTICLES)
    sample = arts[::step][:N_ARTICLES]

    rows = []
    rid = 0
    for key in sample:
        n = article_num(key)
        ar = clean(data[key]["arabic"])
        en = clean(data[key]["english"])

        # --- forward: number -> text ---
        rid += 1
        rows.append({
            "id": rid,
            "question": f"ما نص المادة {n} من القانون المدني المصري؟",
            "answer": ar,
            "legal_reference": f"مادة {n}",
            "direction": "forward",
            "language": "ar",
        })
        rid += 1
        rows.append({
            "id": rid,
            "question": f"What does Article {n} of the Egyptian Civil Code state?",
            "answer": en,
            "legal_reference": f"Article {n}",
            "direction": "forward",
            "language": "en",
        })

        # --- reverse: quote -> number ---
        rid += 1
        rows.append({
            "id": rid,
            "question": f"في أي مادة من القانون المدني المصري ورد هذا النص: «{snippet(ar)}»؟",
            "answer": f"المادة {n}",
            "legal_reference": f"مادة {n}",
            "direction": "reverse",
            "language": "ar",
        })
        rid += 1
        rows.append({
            "id": rid,
            "question": f'In which article of the Egyptian Civil Code does the following text appear: "{snippet(en)}"?',
            "answer": f"Article {n}",
            "legal_reference": f"Article {n}",
            "direction": "reverse",
            "language": "en",
        })

    fields = ["id", "question", "answer", "legal_reference", "direction", "language"]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows from {len(sample)} articles -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
