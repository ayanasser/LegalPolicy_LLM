"""Rebuild data/qa_pairs.jsonl + qa_pairs_val.jsonl from the polish cache.

Used between manual polishing batches to flush every cached polish into the
training files. Looks up cache keys under both ``google|...`` and
``claude|...`` so Gemini- and Claude-polished entries coexist.

Iterates over EVERY cache entry (rather than a deterministic seed-based
selection) so we capture all polishes regardless of which templates were
active when each one was made.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_explainer.finetune.dataset_builder import (
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

SEED = 13
VAL_FRACTION = 0.15


def main() -> int:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus(CORPUS_PATH)

    populated = []
    provenance = {"google": 0, "claude": 0}

    # Iterate over every cache entry (key format: provider|article|lang|instruction[:120])
    for key, response in cache.items():
        try:
            provider, article_key, language, instruction_prefix = key.split("|", 3)
        except ValueError:
            print(f"WARN: skipping malformed cache key: {key[:80]}", file=sys.stderr)
            continue

        if provider not in ("google", "claude"):
            continue

        # The cached instruction may be truncated to [:120]. We need to
        # reconstruct or accept the truncated version. Since the truncation
        # only matters for VERY long instructions, we use the prefix as-is.
        instruction = instruction_prefix

        # Look up the raw article text for context (also serves as sanity
        # check that the article exists in the corpus).
        if article_key not in corpus:
            print(f"WARN: cache entry references unknown article {article_key}", file=sys.stderr)
            continue

        provenance[provider] += 1
        populated.append({
            "article_key": article_key,
            "language": language,
            "instruction": instruction,
            "polished_response": response,
            "kind": "explanation",
        })

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

    print(f"Cache covered: {provenance}; total cache entries: {len(cache)}")
    print(f"Refusals: {len(refusals)}; unique total: {len(unique)}")
    print(f"Train: {len(train):>4} -> {TRAIN_PATH}")
    print(f"Val:   {len(val):>4} -> {VAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
