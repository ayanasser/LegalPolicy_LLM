"""Dump the next N missing (article, language) combos from orig_data.json
that don't yet have a polished response in data/_polish_cache.json.

Each missing combo gets ONE polish (a single instruction variant), since the
goal here is full corpus coverage rather than multiple template variants.

Usage:
    python scripts/dump_missing_articles.py [--size 40] [--lang en|ar|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_explainer.finetune.dataset_builder import (
    article_label,
    load_corpus,
    EN_TEMPLATES,
    AR_TEMPLATES,
)

CACHE_PATH = ROOT / "data" / "_polish_cache.json"
CORPUS_PATH = ROOT / "data" / "orig_data.json"
BATCH_IN = ROOT / "data" / "_polish_batch_in.json"

# Use the first template for each language as the canonical "single polish"
EN_TEMPLATE = EN_TEMPLATES[0]   # "Explain {art_label} ... in plain language."
AR_TEMPLATE = AR_TEMPLATES[0]   # "اشرح {art_label} ... بلغة بسيطة وواضحة."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--lang", choices=["en", "ar", "both"], default="both")
    ap.add_argument("--min-len", type=int, default=80,
                    help="Minimum article text length (skip ultra-short articles)")
    ap.add_argument("--max-len", type=int, default=2000)
    args = ap.parse_args()

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus(CORPUS_PATH)

    # Track which (article, lang) combos already have ANY cached polish
    covered = set()
    for key in cache:
        try:
            _provider, article_key, language, _instr = key.split("|", 3)
            covered.add((article_key, language))
        except ValueError:
            pass

    # Find missing combos
    missing = []
    for article_key, data in corpus.items():
        if not article_key.startswith("Article") or not isinstance(data, dict):
            continue
        for lang, field, template in (("en", "english", EN_TEMPLATE),
                                       ("ar", "arabic", AR_TEMPLATE)):
            if args.lang != "both" and args.lang != lang:
                continue
            if (article_key, lang) in covered:
                continue
            text = (data.get(field) or "").strip()
            if not text or not (args.min_len <= len(text) <= args.max_len):
                continue
            missing.append({
                "article_key": article_key,
                "language": lang,
                "instruction": template.format(art_label=article_label(article_key, lang)),
                "raw_article": text,
            })

    en_total = sum(1 for m in missing if m["language"] == "en")
    ar_total = sum(1 for m in missing if m["language"] == "ar")
    print(f"Missing total: {len(missing)} (EN={en_total}, AR={ar_total})")

    # Take a balanced batch
    batch = []
    if args.lang == "both":
        half = args.size // 2
        en_missing = [m for m in missing if m["language"] == "en"]
        ar_missing = [m for m in missing if m["language"] == "ar"]
        en_take = min(half, len(en_missing))
        ar_take = min(args.size - en_take, len(ar_missing))
        if en_take + ar_take < args.size:
            en_take = min(args.size - ar_take, len(en_missing))
        batch = en_missing[:en_take] + ar_missing[:ar_take]
    else:
        batch = missing[:args.size]

    BATCH_IN.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    en_in_batch = sum(1 for m in batch if m["language"] == "en")
    ar_in_batch = sum(1 for m in batch if m["language"] == "ar")
    print(f"Wrote {len(batch)} pairs to {BATCH_IN} ({en_in_batch} EN + {ar_in_batch} AR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
