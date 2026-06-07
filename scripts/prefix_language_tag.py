"""Prefix every prompt with an explicit language tag — [AR] / [EN] / [BI].

Cheapest no-data-change way to lock the output language and reduce the script
bleed seen on bilingual Qwen QLoRA runs (Chinese / French tokens leaking into
Arabic answers). Only the FIRST user message gets the tag — the assistant
target is untouched. Operates in place; idempotent (re-running is a no-op).

Mapping by the row's ``language`` field:
    "en" -> "[EN] "
    "ar" -> "[AR] "
    "bi" -> "[BI] "      (kn_bilingual rows that emit both AR and EN)
    other / missing -> the message is left alone

Usage:
    # default: tag all four knowledge datasets in place
    python scripts/prefix_language_tag.py

    # explicit set
    python scripts/prefix_language_tag.py --files data/qa_pairs_knowledge.jsonl ...

    # dry-run (count what would change, write nothing)
    python scripts/prefix_language_tag.py --check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FILES = [
    PROJECT_ROOT / "data" / "qa_pairs_knowledge.jsonl",
    PROJECT_ROOT / "data" / "qa_pairs_knowledge_val.jsonl",
    PROJECT_ROOT / "data" / "qa_pairs_knowledge_words.jsonl",
    PROJECT_ROOT / "data" / "qa_pairs_knowledge_words_val.jsonl",
]

_TAG = {"en": "[EN] ", "ar": "[AR] ", "bi": "[BI] "}
_ANY_TAG = ("[EN] ", "[AR] ", "[BI] ")


def _tag_for(row: dict) -> str | None:
    lang = (row.get("language") or "").lower()
    return _TAG.get(lang)


def process_one(path: Path, *, check_only: bool) -> tuple[int, int, int]:
    """Return (n_rows, n_already_tagged, n_changed)."""
    n_rows = n_skip = n_changed = 0
    new_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            n_rows += 1
            rec = json.loads(raw)
            msgs = rec.get("messages") or []
            tag = _tag_for(rec)
            if msgs and tag:
                first = msgs[0]
                content = first.get("content") or ""
                if content.startswith(_ANY_TAG):
                    n_skip += 1
                else:
                    first["content"] = tag + content
                    n_changed += 1
            new_lines.append(json.dumps(rec, ensure_ascii=False))
    if not check_only and n_changed:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return n_rows, n_skip, n_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*", type=Path, default=DEFAULT_FILES)
    ap.add_argument("--check", action="store_true",
                    help="Report what would change, write nothing.")
    args = ap.parse_args()

    print(f"{'file':>52s}  {'rows':>6s}  {'already':>7s}  {'changed':>7s}")
    total_rows = total_skip = total_changed = 0
    for p in args.files:
        if not Path(p).exists():
            print(f"{str(p)[-52:]:>52s}  NOT FOUND")
            continue
        n, s, c = process_one(Path(p), check_only=args.check)
        total_rows += n; total_skip += s; total_changed += c
        print(f"{str(p)[-52:]:>52s}  {n:>6d}  {s:>7d}  {c:>7d}")
    print(f"{'TOTAL':>52s}  {total_rows:>6d}  {total_skip:>7d}  {total_changed:>7d}")
    if args.check:
        print("(--check: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
