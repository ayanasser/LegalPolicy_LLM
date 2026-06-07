"""Convert article-number references in the knowledge dataset from digits to words.

  "Article 775"     -> "Article seven hundred and seventy five"
  "المادة 23"       -> "المادة ثلاثة و عشرون"
  "### Article Number\\n775"  -> "### Article Number\\nseven hundred and seventy five"

Scope (deliberately narrow):
  - Converts NATURAL-LANGUAGE article references in prompts AND assistant answers.
  - LEAVES the verbatim Civil-Code text untouched (paragraph markers like (١), and
    cross-references embedded inside an article's quoted text — converting those
    would corrupt the statute. The dataset's whole point is teaching the *exact*
    article text.)

Outputs new files; the originals are preserved:
  data/qa_pairs_knowledge.jsonl       -> data/qa_pairs_knowledge_words.jsonl
  data/qa_pairs_knowledge_val.jsonl   -> data/qa_pairs_knowledge_words_val.jsonl

Usage:
  python scripts/convert_article_numbers_to_words.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from num2words import num2words

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Natural-language "Article N" / "Articles N" — case-insensitive on the word.
_RE_EN = re.compile(r"\b(Articles?)\s+(\d{1,4})\b")
# Arabic "المادة N" with optional shadda on the daal.
_RE_AR = re.compile(r"(المادّ?ة)\s+(\d{1,4})")
# kn_card header followed by a bare number on the next line(s).
_RE_CARD_EN = re.compile(r"(###\s*Article\s*Number\s*\n+)(\d{1,4})", re.IGNORECASE)
_RE_CARD_AR = re.compile(r"(###\s*رقم\s*المادة\s*\n+)(\d{1,4})")


def _en(n: int) -> str:
    return num2words(n, lang="en").replace("-", " ")  # "twenty-three" -> "twenty three"


def _ar(n: int) -> str:
    return num2words(n, lang="ar")


def convert(text: str) -> str:
    text = _RE_EN.sub(lambda m: f"{m.group(1)} {_en(int(m.group(2)))}", text)
    text = _RE_AR.sub(lambda m: f"{m.group(1)} {_ar(int(m.group(2)))}", text)
    text = _RE_CARD_EN.sub(lambda m: f"{m.group(1)}{_en(int(m.group(2)))}", text)
    text = _RE_CARD_AR.sub(lambda m: f"{m.group(1)}{_ar(int(m.group(2)))}", text)
    return text


def convert_record(rec: dict) -> tuple[dict, bool]:
    """Return (new_record, changed?)."""
    new = dict(rec)
    new["messages"] = []
    changed = False
    for msg in rec["messages"]:
        new_content = convert(msg["content"])
        if new_content != msg["content"]:
            changed = True
        new["messages"].append({"role": msg["role"], "content": new_content})
    return new, changed


def process(src: Path, dst: Path) -> tuple[int, int]:
    n_total = n_changed = 0
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new_rec, changed = convert_record(rec)
            fo.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            n_total += 1
            n_changed += int(changed)
    return n_total, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs_knowledge.jsonl")
    ap.add_argument("--val", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs_knowledge_val.jsonl")
    ap.add_argument("--train-out", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs_knowledge_words.jsonl")
    ap.add_argument("--val-out", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs_knowledge_words_val.jsonl")
    args = ap.parse_args()

    for src, dst in [(args.train, args.train_out), (args.val, args.val_out)]:
        n, c = process(src, dst)
        print(f"{src.name:>38}  ->  {dst.name:<38}   "
              f"records: {n}, changed: {c} ({100*c/n:.1f}%)")


if __name__ == "__main__":
    main()
