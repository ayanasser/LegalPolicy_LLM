"""Build instruction-response training pairs from data/orig_data.json.

Generates ~700-800 bilingual EN/AR pairs by:
  1. Sampling Egyptian Civil Code articles and applying deterministic instruction templates.
  2. (Optional) Polishing each (instruction, article) into the project's house
     style via the Anthropic Claude API. Polished responses are cached on disk
     so re-runs are cheap.
  3. Loading hand-curated refusal pairs from configs/refusal_seeds.yaml.
  4. Deduping, shuffling, and writing 85/15 train/val splits as ChatML JSONL.

Usage:
    python scripts/build_dataset.py --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml
    # add --no-polish to skip Claude (uses raw article text as response)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

EN_TEMPLATES = [
    "Explain {art_label} of the Egyptian Civil Code in plain language.",
    "What does {art_label} of the Egyptian Civil Code establish? Summarize it for a non-lawyer.",
    "A user asks: 'Walk me through {art_label} of the Egyptian Civil Code.' Provide a clear, structured explanation.",
]
AR_TEMPLATES = [
    "اشرح {art_label} من القانون المدني المصري بلغة بسيطة وواضحة.",
    "ماذا تنص {art_label} من القانون المدني المصري؟ لخصها لشخص غير متخصص.",
    "يسأل مستخدم: 'وضح لي {art_label} من القانون المدني المصري.' قدم شرحاً منظماً ومفهوماً.",
]


def article_label(key: str, lang: str) -> str:
    n = key.replace("Article", "").strip()
    return f"Article {n}" if lang == "en" else f"المادة {n}"


def load_corpus(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_articles(corpus, lang: str, *, n: int, min_len=80, max_len=2000, seed=13):
    field = "english" if lang == "en" else "arabic"
    keys = [
        k for k in corpus
        if k.startswith("Article")
        and isinstance(corpus[k], dict)
        and (corpus[k].get(field) or "").strip()
        and min_len <= len(corpus[k][field].strip()) <= max_len
    ]
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys[:n]


def build_template_pairs(corpus, keys, lang, variants_per_article=2, seed=13):
    rng = random.Random(seed)
    field = "english" if lang == "en" else "arabic"
    templates = EN_TEMPLATES if lang == "en" else AR_TEMPLATES
    out = []
    for k in keys:
        text = corpus[k][field].strip()
        for t in rng.sample(templates, k=min(variants_per_article, len(templates))):
            out.append({
                "article_key": k,
                "language": lang,
                "instruction": t.format(art_label=article_label(k, lang)),
                "raw_article": text,
                "metadata": corpus[k].get("metadata", []),
                "kind": "explanation",
            })
    return out


def load_refusal_seeds(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for lang in ("en", "ar"):
        for entry in data.get(lang, []):
            out.append({
                "article_key": None,
                "language": lang,
                "instruction": entry["instruction"],
                "raw_article": "",
                "metadata": [],
                "kind": "refusal",
                "polished_response": entry["response"].strip(),
            })
    return out


def load_synthesis_prompts(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def polish_with_claude(pairs, prompts, *, model: str, cache_path: Path | None = None):
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; export it or pass --no-polish.")
    client = Anthropic(api_key=api_key)

    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached polishes from {cache_path}")

    for i, p in enumerate(pairs):
        cache_key = f"{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        if cache_key in cache:
            p["polished_response"] = cache[cache_key]
            continue
        art_label_str = article_label(p["article_key"], p["language"])
        user_msg = prompts["user_template"].format(
            instruction=p["instruction"],
            art_label=art_label_str,
            article_text=p["raw_article"],
        )
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=900,
                    system=prompts["system"],
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = resp.content[0].text.strip()
                p["polished_response"] = text
                cache[cache_key] = text
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  failed pair {i}: {e}")
                    p["polished_response"] = p["raw_article"]
                else:
                    time.sleep(2 ** attempt)
        if (i + 1) % 25 == 0:
            print(f"  polished {i+1}/{len(pairs)}")
            if cache_path:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    if cache_path:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return pairs


def to_chat_record(p):
    return {
        "messages": [
            {"role": "user", "content": p["instruction"]},
            {"role": "assistant", "content": p["polished_response"]},
        ],
        "language": p["language"],
        "kind": p.get("kind", "explanation"),
        "article_key": p.get("article_key"),
    }


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--corpus", default=PROJECT_ROOT / "data" / "orig_data.json", type=Path)
    ap.add_argument("--articles-per-lang", type=int, default=350)
    ap.add_argument("--variants-per-article", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--no-polish", action="store_true",
                    help="Skip Claude polish; use raw article text as response.")
    ap.add_argument("--polish-model", default="claude-sonnet-4-6")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_jsonl = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_jsonl = PROJECT_ROOT / cfg["data"]["val_jsonl"]
    cache_path = PROJECT_ROOT / "data" / "_polish_cache.json"

    rng = random.Random(args.seed)

    corpus = load_corpus(args.corpus)
    print(f"Loaded corpus: {len(corpus)} keys")

    en_keys = select_articles(corpus, "en", n=args.articles_per_lang, seed=args.seed)
    ar_keys = select_articles(corpus, "ar", n=args.articles_per_lang, seed=args.seed + 1)
    print(f"Selected: EN={len(en_keys)}  AR={len(ar_keys)}")

    template_pairs = (
        build_template_pairs(corpus, en_keys, "en", args.variants_per_article, args.seed)
        + build_template_pairs(corpus, ar_keys, "ar", args.variants_per_article, args.seed + 1)
    )
    rng.shuffle(template_pairs)
    print(f"Template pairs: {len(template_pairs)}")

    if not args.no_polish:
        prompts = load_synthesis_prompts(CONFIGS_DIR / "synthesis_prompts.yaml")
        template_pairs = polish_with_claude(
            template_pairs, prompts, model=args.polish_model, cache_path=cache_path,
        )
    else:
        for p in template_pairs:
            p["polished_response"] = p["raw_article"]

    refusal_pairs = load_refusal_seeds(CONFIGS_DIR / "refusal_seeds.yaml")
    print(f"Refusal pairs: {len(refusal_pairs)}")

    all_pairs = template_pairs + refusal_pairs
    seen, unique = set(), []
    for p in all_pairs:
        key = (p["instruction"][:200], p["polished_response"][:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    rng.shuffle(unique)
    print(f"Unique pairs: {len(unique)}")

    n_val = max(1, int(len(unique) * args.val_fraction))
    val, train = unique[:n_val], unique[n_val:]
    write_jsonl([to_chat_record(p) for p in train], train_jsonl)
    write_jsonl([to_chat_record(p) for p in val], val_jsonl)
    print(f"Train: {len(train):>4} -> {train_jsonl}")
    print(f"Val:   {len(val):>4} -> {val_jsonl}")


if __name__ == "__main__":
    main()
