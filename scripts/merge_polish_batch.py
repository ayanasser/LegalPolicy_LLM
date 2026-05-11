"""Merge a manual polish batch into data/_polish_cache.json.

Reads:
  data/_polish_batch_in.json   — the pairs that needed polishing (list)
  data/_polish_batch_out.json  — manual polishes, dict keyed by approximate
                                 prefix (we re-key by article+lang+template)

Writes the merged entries to data/_polish_cache.json under the
``claude|<article>|<lang>|<instruction[:120]>`` namespace, preserving any
existing entries (Gemini-polished ones live under ``google|...``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_polish_cache.json"
BATCH_IN = ROOT / "data" / "_polish_batch_in.json"
BATCH_OUT = ROOT / "data" / "_polish_batch_out.json"


def template_key(instr: str) -> str:
    """Identify which of the 6 templates a full instruction came from.
    The first ~25 chars are unique per (template, language)."""
    return instr[:25]


def main() -> int:
    pairs = json.loads(BATCH_IN.read_text(encoding="utf-8"))
    polishes = json.loads(BATCH_OUT.read_text(encoding="utf-8"))

    by_lookup: dict[tuple[str, str, str], str] = {}
    for raw_key, response in polishes.items():
        provider, article, lang, instr_prefix = raw_key.split("|", 3)
        if provider != "claude":
            print(f"SKIP non-claude key: {raw_key[:80]}", file=sys.stderr)
            continue
        by_lookup[(article, lang, template_key(instr_prefix))] = response

    cache: dict[str, str] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    added = 0
    skipped = 0
    missing = []
    for p in pairs:
        lookup = (p["article_key"], p["language"], template_key(p["instruction"]))
        if lookup not in by_lookup:
            missing.append(p)
            continue
        key = f"claude|{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        if key in cache:
            skipped += 1
            continue
        cache[key] = by_lookup[lookup]
        added += 1

    if missing:
        print(f"WARNING: {len(missing)} pairs in batch_in had no polish in batch_out:")
        for p in missing[:5]:
            print(f"  {p['article_key']} {p['language']} {p['instruction'][:60]!r}")
        return 1

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added {added} new claude polishes; skipped {skipped} duplicates.")
    print(f"Cache total: {len(cache)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
