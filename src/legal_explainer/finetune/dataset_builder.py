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
    "I'm not a lawyer. Could you tell me what {art_label} of the Egyptian Civil Code actually says, in everyday terms?",
    "Summarize the rule set out in {art_label} of the Egyptian Civil Code, with bullet points and a concrete example.",
    "What is the practical meaning of {art_label} of the Egyptian Civil Code for an ordinary citizen?",
    "Break down {art_label} of the Egyptian Civil Code: what it covers, when it applies, and what it requires.",
    "I came across {art_label} of the Egyptian Civil Code. Help me understand it without legal jargon.",
    "Give a structured, plain-language explanation of {art_label} of the Egyptian Civil Code, including a brief example.",
    "In simple Arabic-friendly English, can you explain {art_label} of the Egyptian Civil Code and what it means in real life?",
]
AR_TEMPLATES = [
    "اشرح {art_label} من القانون المدني المصري بلغة بسيطة وواضحة.",
    "ماذا تنص {art_label} من القانون المدني المصري؟ لخصها لشخص غير متخصص.",
    "يسأل مستخدم: 'وضح لي {art_label} من القانون المدني المصري.' قدم شرحاً منظماً ومفهوماً.",
    "لست محامياً. هل يمكنك أن تخبرني بما تقوله {art_label} من القانون المدني المصري بكلمات يومية؟",
    "لخّص القاعدة الواردة في {art_label} من القانون المدني المصري، مع نقاط ومثال عملي.",
    "ما المعنى العملي لـ {art_label} من القانون المدني المصري بالنسبة للمواطن العادي؟",
    "فصّل {art_label} من القانون المدني المصري: ماذا تغطي، ومتى تنطبق، وماذا تتطلب؟",
    "صادفت {art_label} من القانون المدني المصري. ساعدني على فهمها دون مصطلحات قانونية.",
    "أعطني شرحاً منظماً مبسطاً لـ {art_label} من القانون المدني المصري، مع مثال موجز.",
    "بأسلوب بسيط، هل يمكنك توضيح {art_label} من القانون المدني المصري وما تعنيه على أرض الواقع؟",
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


def select_articles_bilingual(corpus, *, n: int, min_len=80, max_len=2000, seed=13):
    """Select n articles that have acceptable text in BOTH english and arabic.

    Used for cross-language parity: each chosen article will be asked about
    in EN and in AR, so the model sees identical content twice.
    """
    keys = [
        k for k in corpus
        if k.startswith("Article")
        and isinstance(corpus[k], dict)
        and (corpus[k].get("english") or "").strip()
        and (corpus[k].get("arabic") or "").strip()
        and min_len <= len((corpus[k]["english"] or "").strip()) <= max_len
        and min_len <= len((corpus[k]["arabic"] or "").strip()) <= max_len
    ]
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys[:n]


def build_contrastive_pairs(corpus, *, n: int, seed=13) -> list[dict]:
    """Generate pairs that teach the model to refuse claims it can't ground.

    Each pair asks about an article number that does NOT exist in the corpus
    (e.g. Article 9999) and the gold response is a polite refusal that says
    so explicitly. This directly attacks the hallucination failure mode
    surfaced by the LLM-as-judge eval (model fabricates content for every
    article number).
    """
    rng = random.Random(seed)
    existing = {int(k.replace("Article", "").strip())
                for k in corpus if k.startswith("Article")
                and k.replace("Article", "").strip().isdigit()}
    max_real = max(existing) if existing else 1093
    candidates = []
    while len(candidates) < n * 2:
        candidate = rng.randint(max_real + 1, max_real + 5000)
        if candidate not in existing:
            candidates.append(candidate)

    out: list[dict] = []
    for i, num in enumerate(candidates[:n]):
        lang = "en" if i % 2 == 0 else "ar"
        if lang == "en":
            instruction = rng.choice([
                f"Explain Article {num} of the Egyptian Civil Code in plain language.",
                f"What does Article {num} of the Egyptian Civil Code establish?",
                f"Summarize Article {num} of the Egyptian Civil Code for a non-lawyer.",
            ])
            response = (
                f"I cannot find Article {num} in the Egyptian Civil Code that I am scoped to. "
                f"My corpus covers articles up to Article {max_real}, and Article {num} is not present. "
                f"I will not invent the content of an article that does not exist.\n\n"
                f"If you meant a different article number, please share it and I can help. "
                f"For questions about an article I cannot locate, please consult a qualified Egyptian attorney "
                f"or an authoritative published edition of the Egyptian Civil Code.\n\n"
                f"DISCLAIMER: This is general information only, not legal advice."
            )
        else:
            instruction = rng.choice([
                f"اشرح المادة {num} من القانون المدني المصري بلغة بسيطة.",
                f"ماذا تنص المادة {num} من القانون المدني المصري؟",
                f"لخّص المادة {num} من القانون المدني المصري لشخص غير متخصص.",
            ])
            response = (
                f"لا أجد المادة {num} في القانون المدني المصري ضمن النطاق المتاح لي. "
                f"يشمل المرجع لدي المواد حتى المادة {max_real}، ولا توجد المادة {num} ضمنه. "
                f"لن أختلق محتوى مادة غير موجودة.\n\n"
                f"إن كنت تقصد رقم مادة مختلفاً يُرجى تزويدي به. "
                f"وللأسئلة عن مادة لا أستطيع تحديدها يُرجى استشارة محامٍ مصري مؤهل "
                f"أو الرجوع إلى نسخة منشورة معتمدة من القانون المدني المصري.\n\n"
                f"تنبيه: هذه معلومات عامة وليست استشارة قانونية."
            )
        out.append({
            "article_key": None,
            "language": lang,
            "instruction": instruction,
            "raw_article": "",
            "metadata": [],
            "kind": "contrastive",
            "polished_response": response,
        })
    return out


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


# ---------------------------------------------------------------------------
# Stage 4 — PEFT-RAFT (Retrieval-Augmented Fine-Tuning)
#
# Each training example is rewritten so the question is preceded by a context
# block containing the article it is about (the "oracle") plus one or more
# distractor articles. The gold response is unchanged. This teaches the LoRA
# adapter to (a) locate the right article in the provided context and (b)
# rephrase it in house style — and, as a side effect, exposes the adapter to
# every article's text many times during training, which is the strongest
# content-learning signal available without enlarging the base model. Only the
# adapter weights update, so this remains strictly a PEFT method.
# ---------------------------------------------------------------------------

RAFT_HEADER_EN = (
    "Use the Egyptian Civil Code article(s) below to answer the question. "
    "Ground your explanation strictly in the article the question is about. "
    "If that article is not present below, say so plainly and do not invent its content."
)
RAFT_HEADER_AR = (
    "استعن بنصوص القانون المدني المصري الواردة أدناه للإجابة عن السؤال. "
    "اجعل شرحك مستنداً حصراً إلى المادة المطلوبة في السؤال. "
    "وإن لم تكن تلك المادة موجودة أدناه فقل ذلك صراحةً ولا تختلق محتواها."
)


def _clean_article_text(text: str, max_chars: int = 900) -> str:
    """Collapse whitespace and truncate article text for use in a context block."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " […]"
    return cleaned


def _format_context_block(article_entries: list[tuple[str, str]], lang: str) -> str:
    """article_entries: list of (article_label, article_text). Returns the prompt prefix."""
    header = RAFT_HEADER_EN if lang == "en" else RAFT_HEADER_AR
    lines = [header, ""]
    for label, text in article_entries:
        lines.append(f"[{label}]")
        lines.append(text)
        lines.append("")
    q_word = "Question" if lang == "en" else "السؤال"
    lines.append(f"{q_word}:")
    return "\n".join(lines) + " "


def _eligible_distractor_keys(corpus, *, min_len=80, max_len=2000) -> list[str]:
    return [
        k for k in corpus
        if k.startswith("Article") and isinstance(corpus[k], dict)
        and (corpus[k].get("english") or "").strip()
        and (corpus[k].get("arabic") or "").strip()
        and min_len <= len((corpus[k]["english"] or "").strip()) <= max_len
    ]


def apply_raft_context(pairs: list[dict], corpus, *, n_distractors: int = 1,
                       seed: int = 13, max_article_chars: int = 900) -> list[dict]:
    """Rewrite each pair's instruction to include a RAFT-style context block.

    - explanation pairs: context = [oracle article] + [n_distractors random others], shuffled.
    - contrastive pairs (asked article does not exist): context = [n_distractors+1 random
      real articles], none matching the asked number -> the gold "I can't find it" answer
      is the natural RAFT 'oracle-absent' case.
    - refusal pairs: context = "[No relevant Civil Code articles for this request.]" marker.
    """
    rng = random.Random(seed)
    distractor_pool = _eligible_distractor_keys(corpus)
    for p in pairs:
        lang = p["language"]
        field = "english" if lang == "en" else "arabic"
        if p.get("kind") == "explanation" and p.get("article_key"):
            oracle_key = p["article_key"]
            others = [k for k in distractor_pool if k != oracle_key]
            picks = rng.sample(others, k=min(n_distractors, len(others)))
            entries = [(article_label(oracle_key, lang),
                        _clean_article_text(corpus[oracle_key][field], max_article_chars))]
            for k in picks:
                entries.append((article_label(k, lang),
                                _clean_article_text(corpus[k][field], max_article_chars)))
            rng.shuffle(entries)
            p["instruction"] = _format_context_block(entries, lang) + p["instruction"]
        elif p.get("kind") == "contrastive":
            picks = rng.sample(distractor_pool, k=min(n_distractors + 1, len(distractor_pool)))
            entries = [(article_label(k, lang),
                        _clean_article_text(corpus[k][field], max_article_chars)) for k in picks]
            p["instruction"] = _format_context_block(entries, lang) + p["instruction"]
        else:  # refusal or anything else without an article
            marker = ("[No relevant Egyptian Civil Code articles for this request.]"
                      if lang == "en"
                      else "[لا توجد مواد ذات صلة من القانون المدني المصري لهذا الطلب.]")
            ctx_label = "Context" if lang == "en" else "السياق"
            p["instruction"] = _format_context_block([(ctx_label, marker)], lang) + p["instruction"]
    return pairs


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


# Surface-level wrappers for refusal-question paraphrasing. Same gist, different
# framing — sufficient to give the model multiple natural phrasings per seed
# without inventing new content. The gold response stays identical.
EN_REFUSAL_WRAPPERS = [
    "{q}",
    "Quick question — {q}",
    "Hi, {q}",
    "I have a question. {q}",
    "Could you help with this: {q}",
    "Hey assistant, {q}",
    "{q} Please advise.",
    "{q} Thanks.",
    "Need your input: {q}",
    "{q} Can you?",
]
AR_REFUSAL_WRAPPERS = [
    "{q}",
    "سؤال سريع — {q}",
    "مرحباً، {q}",
    "لدي سؤال. {q}",
    "هل يمكنك مساعدتي في هذا: {q}",
    "مساعد، {q}",
    "{q} من فضلك.",
    "{q} شكراً.",
    "أحتاج رأيك: {q}",
    "{q} هل يمكنك؟",
]


def expand_refusal_seeds(refusal_pairs: list[dict], k: int, seed: int = 13) -> list[dict]:
    """Augment each refusal seed with k paraphrased question framings.

    The gold response is preserved verbatim — only the surface form of the
    question changes. This raises the refusal share of the training mix
    without inventing new refusal categories or new responses, which would
    require additional human review or LLM polishing.
    """
    if k <= 1:
        return refusal_pairs
    rng = random.Random(seed)
    out: list[dict] = []
    for p in refusal_pairs:
        wrappers = EN_REFUSAL_WRAPPERS if p["language"] == "en" else AR_REFUSAL_WRAPPERS
        chosen = wrappers[:1] + rng.sample(wrappers[1:], k=min(k - 1, len(wrappers) - 1))
        for w in chosen:
            new_p = dict(p)
            new_p["instruction"] = w.format(q=p["instruction"])
            out.append(new_p)
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

    # Build a fast lookup of (provider, article_key, language) -> polish, drawn
    # from any cache entry that matches that triple regardless of the original
    # instruction. This lets a Stage-1 build with new question phrasings reuse
    # polishes produced by an earlier build with different phrasings, since the
    # answer (article explanation in house style) is identical across variants.
    polish_by_article: dict[tuple[str, str | None, str], str] = {}
    polish_by_article_any_provider: dict[tuple[str | None, str], str] = {}
    for k, v in cache.items():
        parts = k.split("|", 3)
        if len(parts) >= 3:
            polish_by_article.setdefault((parts[0], parts[1], parts[2]), v)
            polish_by_article_any_provider.setdefault((parts[1], parts[2]), v)

    last_call_at: float | None = None

    for i, p in enumerate(pairs):
        cache_key = f"{provider}|{p['article_key']}|{p['language']}|{p['instruction'][:120]}"
        if cache_key in cache:
            p["polished_response"] = cache[cache_key]
            continue
        article_polish_key = (provider, p["article_key"], p["language"])
        if article_polish_key in polish_by_article:
            polish = polish_by_article[article_polish_key]
            p["polished_response"] = polish
            cache[cache_key] = polish
            continue
        any_provider_key = (p["article_key"], p["language"])
        if any_provider_key in polish_by_article_any_provider:
            polish = polish_by_article_any_provider[any_provider_key]
            p["polished_response"] = polish
            cache[cache_key] = polish
            polish_by_article[article_polish_key] = polish
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
                polish_by_article[article_polish_key] = text
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
    ap.add_argument("--cross-lang-parity", action="store_true",
                    help="Use the SAME article set in both EN and AR so each article "
                         "is asked twice (once per language). Required for stage-1.")
    ap.add_argument("--n-contrastive", type=int, default=0,
                    help="Number of contrastive 'this article does not exist' pairs "
                         "to add. Trains the model to refuse claims it cannot ground.")
    ap.add_argument("--refusal-seeds", type=str, default="refusal_seeds.yaml",
                    help="Filename inside finetune/configs/ for refusal seeds. "
                         "Use refusal_seeds_v2.yaml for the stage-1 expanded set.")
    ap.add_argument("--refusal-variants", type=int, default=1,
                    help="Number of question paraphrases per refusal seed. "
                         "1 = use seeds verbatim. 8 = augment 8x to lift refusal share "
                         "of the training mix into the 5-10% band recommended for safety SFT.")
    ap.add_argument("--require-cached-polish", action="store_true",
                    help="Restrict article selection to articles that already have "
                         "BOTH EN and AR polishes in data/_polish_cache.json. Useful when "
                         "the polish API is rate-limited or unavailable, since no new API "
                         "calls will be needed.")
    ap.add_argument("--raft", action="store_true",
                    help="Stage-4 PEFT-RAFT mode: prepend a context block (oracle article + "
                         "distractors) to each question. Teaches the adapter to ground its "
                         "answer in the provided article text.")
    ap.add_argument("--raft-distractors", type=int, default=1,
                    help="Number of distractor articles to include alongside the oracle "
                         "article in each RAFT context block.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_jsonl = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_jsonl = PROJECT_ROOT / cfg["data"]["val_jsonl"]
    cache_path = PROJECT_ROOT / "data" / "_polish_cache.json"

    rng = random.Random(args.seed)

    corpus = load_corpus(args.corpus)
    print(f"Loaded corpus: {len(corpus)} keys")

    if args.cross_lang_parity:
        article_pool = corpus
        if args.require_cached_polish and cache_path.exists():
            existing_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_by_article: dict[str, set[str]] = {}
            for k in existing_cache:
                parts = k.split("|", 3)
                if len(parts) >= 3:
                    cached_by_article.setdefault(parts[1], set()).add(parts[2])
            keep = {a for a, langs in cached_by_article.items()
                    if "en" in langs and "ar" in langs and a in corpus}
            article_pool = {k: v for k, v in corpus.items()
                            if not k.startswith("Article") or k in keep}
            print(f"--require-cached-polish: restricting to {len(keep)} articles "
                  f"with both EN+AR already cached")
        shared_keys = select_articles_bilingual(
            article_pool, n=args.articles_per_lang, seed=args.seed,
        )
        en_keys = ar_keys = shared_keys
        print(f"Selected (bilingual parity): {len(shared_keys)} articles in both EN and AR")
    else:
        en_keys = select_articles(corpus, "en", n=args.articles_per_lang, seed=args.seed)
        ar_keys = select_articles(corpus, "ar", n=args.articles_per_lang, seed=args.seed + 1)
        print(f"Selected: EN={len(en_keys)}  AR={len(ar_keys)}")

    template_pairs = (
        build_template_pairs(corpus, en_keys, "en", args.variants_per_article, args.seed)
        + build_template_pairs(corpus, ar_keys, "ar", args.variants_per_article, args.seed + 1)
    )
    rng.shuffle(template_pairs)
    print(f"Template pairs: {len(template_pairs)} "
          f"(variants_per_article={args.variants_per_article})")

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

    refusal_path = CONFIGS_DIR / args.refusal_seeds
    if not refusal_path.exists():
        raise FileNotFoundError(
            f"Refusal seeds file not found: {refusal_path}. "
            f"Pass --refusal-seeds with a filename that exists in {CONFIGS_DIR}."
        )
    refusal_pairs = load_refusal_seeds(refusal_path)
    print(f"Refusal seeds: {len(refusal_pairs)} (from {refusal_path.name})")
    if args.refusal_variants > 1:
        refusal_pairs = expand_refusal_seeds(
            refusal_pairs, k=args.refusal_variants, seed=args.seed,
        )
        print(f"Refusal pairs after {args.refusal_variants}x augmentation: {len(refusal_pairs)}")

    contrastive_pairs: list[dict] = []
    if args.n_contrastive > 0:
        contrastive_pairs = build_contrastive_pairs(
            corpus, n=args.n_contrastive, seed=args.seed + 7,
        )
        print(f"Contrastive (article-doesn't-exist) pairs: {len(contrastive_pairs)}")

    all_pairs = template_pairs + refusal_pairs + contrastive_pairs

    # Dedup on the *original* question + response, before any RAFT context is
    # prepended (the RAFT header is identical across examples, so deduping after
    # would wrongly collapse distinct examples).
    seen, unique = set(), []
    for p in all_pairs:
        key = (p["instruction"][:200], p["polished_response"][:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    print(f"Unique pairs: {len(unique)}")

    if args.raft:
        unique = apply_raft_context(
            unique, corpus, n_distractors=args.raft_distractors, seed=args.seed,
        )
        print(f"RAFT mode: prepended context blocks (oracle + {args.raft_distractors} "
              f"distractor(s)) to {len(unique)} pairs")

    rng.shuffle(unique)

    n_val = max(1, int(len(unique) * args.val_fraction))
    val, train = unique[:n_val], unique[n_val:]
    write_jsonl([to_chat_record(p) for p in train], train_jsonl)
    write_jsonl([to_chat_record(p) for p in val], val_jsonl)
    print(f"Train: {len(train):>4} -> {train_jsonl}")
    print(f"Val:   {len(val):>4} -> {val_jsonl}")


if __name__ == "__main__":
    main()
