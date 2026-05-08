"""Rebuild data/qa_pairs.jsonl + qa_pairs_val.jsonl from the polish cache.

Used between manual polishing batches to flush every cached polish into the
training files. Looks up cache keys under both ``google|...`` and
``claude|...`` so Gemini- and Claude-polished entries coexist.

Identical selection logic and split ratio to dataset_builder.py defaults
(seed=13, 350 articles per language, 2 variants per article, 15% val).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_explainer.finetune.dataset_builder import (
    select_articles,
    build_template_pairs,
    load_corpus,
    load_refusal_seeds,
    to_chat_record,
    write_jsonl,
)

CACHE_PATH = ROOT / "data" / "_polish_cache.json"
CORPUS_PATH = ROOT / "data" / "orig_data.json"
REFUSAL_PATH = ROOT / "src" / "legal_explainer" / "finetune" / "configs" / "refusal_seeds.yaml"
TRAIN_PATH = ROOT / "data" / "qa_pairs.jsonl"
VAL_PATH = ROOT / "data" / "qa_pairs_val.jsonl"

ARTICLES_PER_LANG = 350
VARIANTS = 2
SEED = 13
VAL_FRACTION = 0.15


def main() -> int:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus(CORPUS_PATH)

    en_keys = select_articles(corpus, "en", n=ARTICLES_PER_LANG, seed=SEED)
    ar_keys = select_articles(corpus, "ar", n=ARTICLES_PER_LANG, seed=SEED + 1)
    pairs = (
        build_template_pairs(corpus, en_keys, "en", VARIANTS, SEED)
        + build_template_pairs(corpus, ar_keys, "ar", VARIANTS, SEED + 1)
    )

    populated = []
    missing = 0
    provenance = {"google": 0, "claude": 0}
    for p in pairs:
        suffix = f"{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        for provider in ("google", "claude"):
            key = f"{provider}|{suffix}"
            if key in cache:
                p["polished_response"] = cache[key]
                provenance[provider] += 1
                p["kind"] = "explanation"
                populated.append(p)
                break
        else:
            missing += 1

    refusals = load_refusal_seeds(REFUSAL_PATH)

    rng = random.Random(SEED)
    all_pairs = populated + refusals

    seen, unique = set(), []
    for p in all_pairs:
        ckey = (p["instruction"][:200], p["polished_response"][:200])
        if ckey in seen:
            continue
        seen.add(ckey)
        unique.append(p)
    rng.shuffle(unique)

    n_val = max(1, int(len(unique) * VAL_FRACTION))
    val, train = unique[:n_val], unique[n_val:]
    write_jsonl([to_chat_record(p) for p in train], TRAIN_PATH)
    write_jsonl([to_chat_record(p) for p in val], VAL_PATH)

    print(f"Cache covered: {provenance}; missing: {missing}/{len(pairs)}")
    print(f"Refusals: {len(refusals)}; unique total: {len(unique)}")
    print(f"Train: {len(train):>4} -> {TRAIN_PATH}")
    print(f"Val:   {len(val):>4} -> {VAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
