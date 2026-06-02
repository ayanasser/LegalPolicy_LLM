"""Generate natural-language scenario records (kn_user_scenario, kn_lawyer_scenario)
for every article in data/orig_data.json, using **Claude** via the Anthropic API.

For each article, the generator asks Claude for 4 (question, answer) pairs:
  - English layperson question + grounded answer that cites "Article N"
  - Arabic (Egyptian colloquial) layperson question + answer that cites "المادة N"
  - English lawyer question + answer
  - Arabic lawyer question + answer

Constraints baked into the prompt:
  - The QUESTION never mentions the article number (forces topic->article binding)
  - The ANSWER must cite the article explicitly so the model also learns the binding
  - Answer is grounded strictly in THIS article's rule (not a different one)

Output:
  - JSONL records appended to data/scenarios_full.jsonl (one record per generated
    voice; so 4 records per article)
  - Caches the raw JSON response per article in data/_scenario_cache.json so
    re-runs are free / resumable. Same idempotency pattern as _polish_cache.json.

Usage:
  python scripts/build_scenarios.py                              # all 1,093 articles
  python scripts/build_scenarios.py --limit 6                    # 6-article smoke
  python scripts/build_scenarios.py --start-at 500 --limit 100   # resumable batches
  python scripts/build_scenarios.py --model claude-haiku-4-5     # cheaper, default
  python scripts/build_scenarios.py --model claude-sonnet-4-6    # higher quality
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import anthropic  # noqa: E402

CORPUS_PATH    = PROJECT_ROOT / "data" / "orig_data.json"
CACHE_PATH     = PROJECT_ROOT / "data" / "_scenario_cache.json"
OUT_PATH       = PROJECT_ROOT / "data" / "scenarios_full.jsonl"
EXEMPLARS_PATH = PROJECT_ROOT / "data" / "scenarios_sample.jsonl"

SYSTEM_PROMPT = """You generate natural-language scenarios about the Egyptian Civil Code for fine-tuning a legal-domain language model.

For the given article you will produce exactly 4 (question, answer) pairs:

1. **en_user** — An English LAYPERSON question describing a real-life situation that this article's rule resolves. First-person, plain language, no legal jargon. The question MUST NOT contain any article number or the phrase "Article N".

2. **ar_user** — An EGYPTIAN COLLOQUIAL ARABIC layperson question. Informal dialect (آه/مش/إزاي/أقدر/عايز…), first-person, the kind a real Egyptian would ask. MUST NOT contain any article number or "المادة N".

3. **en_lawyer** — A professional English question framed by a LAWYER advising a client ("I'm advising a client who…", "I'm representing X who…"). Technical but readable. MUST NOT contain any article number.

4. **ar_lawyer** — A professional MODERN STANDARD ARABIC question framed by a lawyer ("أنا محامي…", "موكلي…"). MUST NOT contain "المادة N".

For each ANSWER:
  - Cite the article explicitly: "Article {N} of the Egyptian Civil Code" / "المادة {N} من القانون المدني المصري" — at least once, preferably embedded near the substantive rule
  - Paraphrase THIS article's actual rule — do not invent rules from other articles
  - Match the formality of the corresponding question (lay answers should be lay; lawyer answers should be technical)
  - Length: lay answers ~2-4 sentences; lawyer answers ~3-6 sentences with structured reasoning

Output STRICT JSON only, no preamble, no markdown:
{
  "en_user":   {"q": "...", "a": "..."},
  "ar_user":   {"q": "...", "a": "..."},
  "en_lawyer": {"q": "...", "a": "..."},
  "ar_lawyer": {"q": "...", "a": "..."}
}
"""

USER_TEMPLATE = """Article {n} of the Egyptian Civil Code

English text:
{en_text}

Arabic text:
{ar_text}

---
Reference exemplars (from OTHER articles — match this tone and structure, do not copy content):

{exemplars}

---
Now produce the 4 scenarios for **Article {n}**. Output only the JSON object."""


def load_exemplars() -> str:
    """Load a compact rendering of the 16 hand-crafted exemplar records as few-shot."""
    if not EXEMPLARS_PATH.exists():
        return ""
    rows = [json.loads(l) for l in EXEMPLARS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Group by article
    by_article: dict[str, dict[str, dict]] = {}
    for r in rows:
        art = r["article_key"]
        kind_lang = f"{r['kind'].replace('kn_', '')}_{r['language']}"
        by_article.setdefault(art, {})[kind_lang] = {
            "q": r["messages"][0]["content"],
            "a": r["messages"][1]["content"],
        }
    # Render up to 2 full article exemplars (= 8 records) — enough for tone without flooding
    out = []
    for art, pairs in list(by_article.items())[:2]:
        out.append(f"=== {art} ===")
        for lang_voice in ["user_scenario_en", "user_scenario_ar", "lawyer_scenario_en", "lawyer_scenario_ar"]:
            p = pairs.get(lang_voice)
            if p:
                voice_label = lang_voice.replace("_scenario", "").replace("_en", " (EN)").replace("_ar", " (AR)")
                out.append(f"[{voice_label}]")
                out.append(f"Q: {p['q']}")
                out.append(f"A: {p['a']}")
                out.append("")
    return "\n".join(out)


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_response(text: str) -> dict:
    """Extract the JSON object from Claude's response (strip any stray prose/fences)."""
    text = text.strip()
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Find first { ... last }
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(text[s : e + 1])


def to_records(article_key: str, payload: dict) -> list[dict]:
    """Convert {en_user, ar_user, en_lawyer, ar_lawyer} into 4 ChatML records."""
    n = article_key.replace("Article", "").strip()
    out = []
    for voice_key, kind, lang in [
        ("en_user",   "kn_user_scenario",   "en"),
        ("ar_user",   "kn_user_scenario",   "ar"),
        ("en_lawyer", "kn_lawyer_scenario", "en"),
        ("ar_lawyer", "kn_lawyer_scenario", "ar"),
    ]:
        pair = payload.get(voice_key)
        if not pair or "q" not in pair or "a" not in pair:
            continue
        out.append({
            "messages": [
                {"role": "user",      "content": pair["q"].strip()},
                {"role": "assistant", "content": pair["a"].strip()},
            ],
            "language":    lang,
            "kind":        kind,
            "article_key": article_key,
        })
    return out


def generate_one(client, model: str, article_key: str, en: str, ar: str,
                 exemplars: str, max_attempts: int = 4) -> dict:
    n = article_key.replace("Article", "").strip()
    user_msg = USER_TEMPLATE.format(n=n, en_text=en.strip(), ar_text=ar.strip(),
                                    exemplars=exemplars)
    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
            payload = parse_response(text)
            # Sanity: questions shouldn't say "Article N"
            for k in ("en_user", "ar_user", "en_lawyer", "ar_lawyer"):
                q = payload.get(k, {}).get("q", "")
                if re.search(r"\bArticle\s+\d", q, re.IGNORECASE) or re.search(r"المادّ?ة\s+\d", q):
                    raise ValueError(f"{k}'s question still mentions an article number: {q[:120]}")
            return payload
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(1.5)
        except anthropic.RateLimitError:
            time.sleep(min(60, 2 ** attempt * 5))
        except anthropic.APIError as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt * 2)
            else:
                raise
    raise RuntimeError(f"Failed after {max_attempts} attempts: {last_err}")


def article_num(key: str) -> int:
    s = key.replace("Article", "").strip()
    return int(s) if s.isdigit() else 10**9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Claude model. Default 'claude-haiku-4-5' for cost; "
                         "use 'claude-sonnet-4-6' for higher quality.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap the number of articles processed this run (0=all).")
    ap.add_argument("--start-at", type=int, default=1,
                    help="First article number to process (for resumable runs).")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="Seconds between API calls (set if rate-limited).")
    ap.add_argument("--append", action="store_true",
                    help="Append to data/scenarios_full.jsonl instead of overwriting.")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    cache = load_cache()
    exemplars = load_exemplars()

    keys = sorted(
        (k for k in corpus if k.startswith("Article") and isinstance(corpus[k], dict)),
        key=article_num,
    )
    keys = [k for k in keys if article_num(k) >= args.start_at]
    if args.limit:
        keys = keys[: args.limit]

    print(f"Model      : {args.model}")
    print(f"Articles   : {len(keys)} (start={args.start_at}, limit={args.limit or '∞'})")
    print(f"Cache hits : {sum(1 for k in keys if k in cache)} / {len(keys)}")
    print(f"Output     : {OUT_PATH} ({'append' if args.append else 'overwrite'})")
    print()

    mode = "a" if args.append else "w"
    written = 0
    api_calls = 0
    with OUT_PATH.open(mode, encoding="utf-8") as fo:
        for i, key in enumerate(keys, 1):
            entry = corpus[key]
            en = (entry.get("english") or "").strip()
            ar = (entry.get("arabic") or "").strip()
            if not en or not ar:
                print(f"[{i}/{len(keys)}] {key}: missing text, skipped")
                continue
            if key in cache:
                payload = cache[key]
            else:
                try:
                    payload = generate_one(client, args.model, key, en, ar, exemplars)
                except Exception as e:
                    print(f"[{i}/{len(keys)}] {key}: FAILED — {e}")
                    continue
                cache[key] = payload
                api_calls += 1
                if api_calls % 10 == 0:
                    save_cache(cache)
                if args.throttle > 0:
                    time.sleep(args.throttle)
            for rec in to_records(key, payload):
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            if i % 25 == 0 or i == len(keys):
                print(f"[{i}/{len(keys)}] {key}: written so far={written}, api_calls={api_calls}")

    save_cache(cache)
    print(f"\nDone. Records written: {written} -> {OUT_PATH}")
    print(f"Total API calls this run: {api_calls} (rest from cache).")


if __name__ == "__main__":
    main()
