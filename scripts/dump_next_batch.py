"""Pick the next N pairs from the deterministic plan that are NOT yet in the
polish cache (under either ``google|...`` or ``claude|...``), and write them
to data/_polish_batch_in.json so they can be polished manually.

Usage:
    python scripts/dump_next_batch.py [--size 20]

Defaults to 20 pairs balanced between EN and AR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_explainer.finetune.dataset_builder import (
    select_articles,
    build_template_pairs,
    load_corpus,
)

CACHE_PATH = ROOT / "data" / "_polish_cache.json"
CORPUS_PATH = ROOT / "data" / "orig_data.json"
BATCH_IN = ROOT / "data" / "_polish_batch_in.json"

ARTICLES_PER_LANG = 350
VARIANTS = 2
SEED = 13


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20)
    args = ap.parse_args()

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus(CORPUS_PATH)

    en_keys = select_articles(corpus, "en", n=ARTICLES_PER_LANG, seed=SEED)
    ar_keys = select_articles(corpus, "ar", n=ARTICLES_PER_LANG, seed=SEED + 1)
    pairs = (
        build_template_pairs(corpus, en_keys, "en", VARIANTS, SEED)
        + build_template_pairs(corpus, ar_keys, "ar", VARIANTS, SEED + 1)
    )

    def cached(p):
        suffix = f"{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        return f"google|{suffix}" in cache or f"claude|{suffix}" in cache

    missing = [p for p in pairs if not cached(p)]
    en = [p for p in missing if p["language"] == "en"]
    ar = [p for p in missing if p["language"] == "ar"]
    print(f"Missing: total={len(missing)} EN={len(en)} AR={len(ar)}")

    half = args.size // 2
    en_take = min(half, len(en))
    ar_take = min(args.size - en_take, len(ar))
    if en_take + ar_take < args.size:
        en_take = min(args.size - ar_take, len(en))
    batch = en[:en_take] + ar[:ar_take]

    out = [
        {
            "article_key": p["article_key"],
            "language": p["language"],
            "instruction": p["instruction"],
            "raw_article": p["raw_article"],
        }
        for p in batch
    ]
    BATCH_IN.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} pairs to {BATCH_IN} ({en_take} EN + {ar_take} AR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
