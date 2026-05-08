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
import re
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

load_dotenv(PROJECT_ROOT / ".env")

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


def _claude_call(client, system: str, user: str, model: str) -> str:
    resp = client.messages.create(
        model=model, max_tokens=900, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _gemini_call(client, system: str, user: str, model: str) -> str:
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=900,
            temperature=0.3,
        ),
    )
    return (resp.text or "").strip()


def _make_client(provider: str):
    if provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env or pass --no-polish.")
        return Anthropic(api_key=api_key), _claude_call
    if provider == "google":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; add it to .env or pass --no-polish.")
        return genai.Client(api_key=api_key), _gemini_call
    raise ValueError(f"Unknown polish provider: {provider}")


def _retry_delay_from_exception(e: Exception) -> float | None:
    """Extract the server-suggested retry delay from a 429 error, in seconds."""
    msg = str(e)
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]", msg)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
    if m:
        return float(m.group(1))
    return None


def _is_daily_quota_error(e: Exception) -> bool:
    """Detect Gemini / Vertex 'requests per day' quota exhaustion (vs. per-minute)."""
    msg = str(e)
    return ("PerDayPerProjectPerModel" in msg
            or "RequestsPerDay" in msg
            or "perDay" in msg)


class DailyQuotaExhausted(RuntimeError):
    """Raised when the polish provider's per-day quota is exhausted.

    Retrying same-day will not help. The polish loop bails so we don't waste
    minutes retrying or, worse, fall back to raw statute text for hundreds of
    pairs and silently poison the cache."""


def polish_pairs(pairs, prompts, *, provider: str, model: str,
                 cache_path: Path | None = None,
                 throttle_seconds: float = 0.0,
                 max_attempts: int = 5):
    client, call = _make_client(provider)

    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached polishes from {cache_path}")

    last_call_at: float | None = None

    for i, p in enumerate(pairs):
        cache_key = f"{provider}|{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        if cache_key in cache:
            p["polished_response"] = cache[cache_key]
            continue
        art_label_str = article_label(p["article_key"], p["language"])
        user_msg = prompts["user_template"].format(
            instruction=p["instruction"],
            art_label=art_label_str,
            article_text=p["raw_article"],
        )
        for attempt in range(max_attempts):
            if throttle_seconds > 0 and last_call_at is not None:
                wait = throttle_seconds - (time.monotonic() - last_call_at)
                if wait > 0:
                    time.sleep(wait)
            try:
                text = call(client, prompts["system"], user_msg, model)
                last_call_at = time.monotonic()
                p["polished_response"] = text
                cache[cache_key] = text
                break
            except Exception as e:
                if _is_daily_quota_error(e):
                    if cache_path:
                        cache_path.write_text(
                            json.dumps(cache, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    raise DailyQuotaExhausted(
                        f"Daily quota exhausted on model={model!r}. "
                        f"Cached {len(cache)} polishes so far. "
                        f"Resume tomorrow (cache will skip done pairs) or pass "
                        f"--polish-model with a different model. Original error: {e}"
                    ) from e
                if attempt == max_attempts - 1:
                    print(f"  failed pair {i} after {max_attempts} attempts: {e}")
                    p["polished_response"] = p["raw_article"]
                    break
                wait = _retry_delay_from_exception(e)
                if wait is None:
                    wait = min(60.0, (2 ** attempt) * 2.0)
                else:
                    wait += 1.5
                print(f"  pair {i} rate-limited; sleeping {wait:.1f}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                last_call_at = time.monotonic()
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
                    help="Skip LLM polish; use raw article text as response.")
    ap.add_argument("--polish-provider", choices=["google", "anthropic"], default="google",
                    help="Which LLM polishes the responses. Default: google (Gemini).")
    ap.add_argument("--polish-model", default=None,
                    help="Defaults to gemini-2.5-flash-lite for google, claude-sonnet-4-6 for anthropic.")
    ap.add_argument("--throttle-seconds", type=float, default=None,
                    help="Minimum seconds between API calls. "
                         "Default: 4.5 for google (free-tier safe for flash-lite), 0 for anthropic.")
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
        model = args.polish_model or (
            "gemini-2.5-flash-lite" if args.polish_provider == "google" else "claude-sonnet-4-6"
        )
        throttle = args.throttle_seconds
        if throttle is None:
            throttle = 4.5 if args.polish_provider == "google" else 0.0
        print(f"Polishing via {args.polish_provider} ({model}); throttle={throttle:.1f}s/call ...")
        try:
            template_pairs = polish_pairs(
                template_pairs, prompts,
                provider=args.polish_provider, model=model,
                cache_path=cache_path, throttle_seconds=throttle,
            )
        except DailyQuotaExhausted as e:
            print(f"\nABORT: {e}")
            print("qa_pairs.jsonl was NOT written (dataset is incomplete).")
            print("Re-run the same command tomorrow; cached polishes will be skipped.")
            sys.exit(1)
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
