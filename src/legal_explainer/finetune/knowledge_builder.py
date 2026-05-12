"""Build a KNOWLEDGE-injection dataset from data/orig_data.json.

Unlike dataset_builder.py (house-style explanation SFT) and its RAFT mode, this
dataset is deliberately *plain and varied*. Its only job is to imprint the
verbatim content of the Egyptian Civil Code into the LoRA weights — i.e. it is
the "continued pre-training" / knowledge stage of a two-stage PEFT recipe:

    Stage A  (THIS dataset, qlora_qwen1_5b_knowledge.yaml)
             -> the LoRA adapter learns WHAT the Code says (1,093 articles).
    Stage B  (qa_pairs.jsonl, the existing Stage-1 house-style SFT)
             -> the same adapter, continued, learns HOW to explain it.

Why this exists
---------------
The LLM-as-judge evals (reports/eval/judge_report*.md) showed the house-style
SFT produced a "style adapter, not a knowledge adapter": legal accuracy stayed
at ~1/5 because each article was seen only ~2 times and only ever inside one
rigid template, so the model learned the template and treated the law as noise.
This builder attacks the two axes that caused that — *exposure* and *task
diversity*:

  * every article appears ~12-16 times,
  * across 5 task families that each force attention to the article TEXT rather
    than to a template slot:
        kn_verbatim    "Quote Article N exactly."            -> the text
        kn_complete    "Article N starts '<prefix> …'."      -> the full text
        kn_gap         "Fill the blank: '… _____ …'."        -> the missing span
        kn_reverse     "Which article says '<text>'?"        -> "Article N"
        kn_placement   "Where does Article N sit?"           -> book/chapter crumbs
        kn_translate   AR text <-> EN text                   -> the other language
        kn_bilingual   "Give Article N in AR and EN."        -> both texts
        kn_card        "Reference card for Article N."       -> number / topic / text
        kn_contrast    "Does Article N concern <topic>?"     -> yes/no + the real topic
        kn_roster      "Which articles deal with <topic>?"   -> the list of article Ns
  * in both EN and AR, with NO Claude/Gemini polish (free, fully deterministic),
  * with NO house-style template — that is taught in Stage B, on purpose.

Output is the same ChatML JSONL that train.py already consumes (`messages`,
`language`, `kind`, `article_key`), so no training-code changes are needed —
just point a config's `data.train_jsonl` / `data.val_jsonl` at the files this
writes.

Usage:
    python scripts/build_dataset_knowledge.py \
        --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_knowledge.yaml

    # smoke test on the first 25 articles:
    python scripts/build_dataset_knowledge.py --config <cfg> --max-articles 25
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

# Mangled paragraph markers from the PDF extraction: ") ١ (" / "( ١ (" / ") ١ )"
# etc. — a 1-2 digit number flanked by any mix of "(" / ")" — should all be "(N)".
_PARA_MARK = re.compile(r"[()]\s*([0-9٠-٩]{1,2})\s*[()]")

# --- breadcrumb / topic extraction helpers -------------------------------
_AR_CHARS = re.compile(r"[؀-ۿ]")
_LAT_CHARS = re.compile(r"[A-Za-z]")
# maximal runs of one script (letters/digits + inner spaces) inside a crumb
_AR_RUN = re.compile(r"[؀-ۿ][؀-ۿ\s]*")
_EN_RUN = re.compile(r"[A-Za-z][A-Za-z\s&/]*")
# leading/trailing numbering & punctuation noise on an extracted label
_EDGE_NOISE = re.compile(r"^[\s\d٠-٩.\-–،؛:()«»\"']+|[\s\d٠-٩.\-–،؛:()«»\"']+$")
# crumb labels too generic / structural to anchor a topic on
_GENERIC_CRUMB = {
    "نصوص القانون المدنى", "باب تمهيدي", "أحكام عامة", "أحكام تمهيدية",
    "القانون", "الحق", "أحكام", "نصوص",
    "general provisions", "preliminary provisions", "introductory provisions",
    "general rules", "preliminary title", "general dispositions",
    "law", "rights", "the law", "movables", "immovables",
}
_STRUCTURAL_RE = re.compile(
    r"^(book|section|chapter|part|title|الكتاب|الباب|الفصل|القسم|الفرع)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Light, safe normalisation of a raw article string from orig_data.json.

    - fix the OCR paragraph-marker artefacts  ') 1 (' / '( 1 ('  ->  '(1)'
    - collapse runs of whitespace (incl. the embedded newlines) into single spaces

    Deeper corpus cleaning (stray fragments, corrupted cross-references) is
    intentionally out of scope here — see the separate corpus-cleanup task.
    """
    t = (text or "").strip()
    t = _PARA_MARK.sub(r"(\1)", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " […]"


def article_number(key: str) -> str:
    return key.replace("Article", "").strip()


def en_label(n: str) -> str:
    return f"Article {n}"


def ar_label(n: str) -> str:
    return f"المادة {n}"


def _clean_label(s: str) -> str | None:
    s = _EDGE_NOISE.sub("", (s or "").strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _crumb_parts(crumb: str) -> tuple[str | None, str | None]:
    """Split one breadcrumb string into (english_part, arabic_part).

    Pulls out the maximal Arabic-script run(s) and Latin-script run(s) separately,
    so a combined crumb like "١ -القانون والحق 1. Laws and Rights" or
    "الاستيلاء على منقول The Appropriation of Movables" yields a clean pair.
    Leading/trailing numbering and punctuation noise is stripped from each side.
    """
    crumb = crumb or ""
    ar = _clean_label(" ".join(m.group(0) for m in _AR_RUN.finditer(crumb)))
    en = _clean_label(" ".join(m.group(0) for m in _EN_RUN.finditer(crumb)))
    return en, ar


def _is_anchorable(label: str | None) -> bool:
    """A topic label specific enough to use for reverse lookups / rosters."""
    if not label:
        return False
    s = label.strip()
    return len(s) >= 4 and s.lower() not in _GENERIC_CRUMB and not _STRUCTURAL_RE.match(s)


def topic_of(meta: list[str]) -> dict:
    """Best-effort topic for an article from its breadcrumb metadata.

    Returns a dict with:
      'en' / 'ar'  : the chosen topic label per language (may be None)
      'specific'   : True if at least one side is an anchorable label (good for
                     reverse lookups / rosters); if False the labels are still
                     usable on a reference card but not for those tasks
      'spec_en' / 'spec_ar' : the anchorable label per language, else None
      'section'    : a coarse book / part id (metadata[1]) used to keep
                     contrastive negatives clearly distant
    """
    crumbs = [c for c in (meta or []) if isinstance(c, str) and c.strip()]
    section = crumbs[1].strip() if len(crumbs) > 1 else (crumbs[0].strip() if crumbs else "?")
    best_en = best_ar = spec_en = spec_ar = None
    for c in crumbs:                       # broad -> narrow; later wins
        en, ar = _crumb_parts(c)
        if en:
            best_en = en
            if _is_anchorable(en):
                spec_en = en
        if ar:
            best_ar = ar
            if _is_anchorable(ar):
                spec_ar = ar
    return {
        "en": spec_en or best_en,
        "ar": spec_ar or best_ar,
        "spec_en": spec_en,
        "spec_ar": spec_ar,
        "specific": bool(spec_en or spec_ar),
        "section": section,
    }


def first_sentence(text: str, *, max_chars: int = 280) -> str:
    """First sentence-ish chunk of a cleaned article text, capped.

    Falls back to a hard truncation. Used as the 'key point' line on a card —
    it is the article's *own* opening words, not a paraphrase, so it cannot
    introduce content that is not in the source text.
    """
    if not text:
        return ""
    m = re.search(r"[.؟!]\s", text)
    head = text[: m.start() + 1].strip() if (m and m.start() >= 20) else text
    return truncate(head, max_chars)


# ---------------------------------------------------------------------------
# instruction / response templates  ({n} = article number, no house style)
# ---------------------------------------------------------------------------

VERBATIM_EN = [
    "Quote Article {n} of the Egyptian Civil Code exactly as it is written.",
    "What is the exact wording of Article {n} of the Egyptian Civil Code?",
    "Reproduce the full text of Article {n} of the Egyptian Civil Code.",
    "State Article {n} of the Egyptian Civil Code verbatim.",
    "Give me the English text of Article {n} of the Egyptian Civil Code.",
    "Write out Article {n} of the Egyptian Civil Code, word for word.",
]
VERBATIM_AR = [
    "اذكر نص المادة {n} من القانون المدني المصري حرفياً.",
    "ما هو النص الحرفي للمادة {n} من القانون المدني المصري؟",
    "أعد كتابة نص المادة {n} من القانون المدني المصري كاملاً.",
    "اكتب المادة {n} من القانون المدني المصري كما وردت بالضبط.",
    "أعطني النص العربي للمادة {n} من القانون المدني المصري.",
    "انقل نص المادة {n} من القانون المدني المصري كلمةً بكلمة.",
]

COMPLETE_EN = [
    "Here is the opening of Article {n} of the Egyptian Civil Code:\n\n\"{prefix} …\"\n\nWrite out the article in full.",
    "Article {n} of the Egyptian Civil Code begins:\n\n\"{prefix} …\"\n\nContinue it and give the complete text.",
    "Complete Article {n} of the Egyptian Civil Code. It starts: \"{prefix} …\"",
]
COMPLETE_AR = [
    "هذه بداية المادة {n} من القانون المدني المصري:\n\n\"{prefix} …\"\n\nاكتب نص المادة كاملاً.",
    "تبدأ المادة {n} من القانون المدني المصري بـ:\n\n\"{prefix} …\"\n\nأكمل النص بالكامل.",
    "أكمل المادة {n} من القانون المدني المصري. تبدأ بـ: \"{prefix} …\"",
]

GAP_EN = [
    "Fill in the blank in Article {n} of the Egyptian Civil Code (answer with the missing words only):\n\n\"{blanked}\"",
    "In Article {n} of the Egyptian Civil Code, what words belong in the blank?\n\n\"{blanked}\"",
]
GAP_AR = [
    "أكمل الفراغ في نص المادة {n} من القانون المدني المصري (اذكر الكلمات الناقصة فقط):\n\n\"{blanked}\"",
    "ما الكلمات التي تملأ الفراغ في المادة {n} من القانون المدني المصري؟\n\n\"{blanked}\"",
]

REVERSE_EN = [
    ("Which article of the Egyptian Civil Code contains the following provision? Reply with the article number only.\n\n\"{quote}\"",
     "{label} of the Egyptian Civil Code."),
    ("Identify the Egyptian Civil Code article that reads:\n\n\"{quote}\"",
     "That is {label} of the Egyptian Civil Code."),
    ("From which article of the Egyptian Civil Code is this taken?\n\n\"{quote}\"",
     "{label}."),
]
REVERSE_AR = [
    ("أي مادة من القانون المدني المصري تتضمن النص التالي؟ اذكر رقم المادة فقط.\n\n\"{quote}\"",
     "{label} من القانون المدني المصري."),
    ("حدد مادة القانون المدني المصري التي تنص على ما يلي:\n\n\"{quote}\"",
     "هي {label} من القانون المدني المصري."),
    ("من أي مادة في القانون المدني المصري أُخذ هذا النص؟\n\n\"{quote}\"",
     "{label}."),
]

PLACEMENT_EN = [
    ("Where does Article {n} sit within the Egyptian Civil Code? Give the book / chapter / section it belongs to.",
     "{label} of the Egyptian Civil Code appears under: {crumbs}."),
    ("What part of the Egyptian Civil Code is Article {n} in, and what subject does it deal with?",
     "{label} is located under: {crumbs}."),
]
PLACEMENT_AR = [
    ("في أي موضع من القانون المدني المصري تقع المادة {n}؟ اذكر الكتاب/الباب/الفصل الذي تتبعه.",
     "{label} من القانون المدني المصري تقع ضمن: {crumbs}."),
    ("إلى أي قسم من القانون المدني المصري تنتمي المادة {n}، وما الموضوع الذي تتناوله؟",
     "{label} مدرجة تحت: {crumbs}."),
]

XLAT_AR2EN = [
    "Here is Article {n} of the Egyptian Civil Code in Arabic:\n\n\"{ar}\"\n\nGive its English text.",
    "Translate Article {n} of the Egyptian Civil Code into English. The Arabic text is:\n\n\"{ar}\"",
    "This is the Arabic of Article {n} of the Egyptian Civil Code: \"{ar}\". What is the English version?",
]
XLAT_EN2AR = [
    "Here is Article {n} of the Egyptian Civil Code in English:\n\n\"{en}\"\n\nGive its Arabic text.",
    "ترجم المادة {n} من القانون المدني المصري إلى العربية. النص الإنجليزي هو:\n\n\"{en}\"",
    "هذا هو النص الإنجليزي للمادة {n} من القانون المدني المصري: \"{en}\". ما النص العربي؟",
]
BOTH_LANGS = [
    ("Provide the text of Article {n} of the Egyptian Civil Code in both Arabic and English.",
     "Arabic:\n{ar}\n\nEnglish:\n{en}"),
    ("Give me Article {n} of the Egyptian Civil Code — Arabic original and English translation.",
     "العربية:\n{ar}\n\nEnglish:\n{en}"),
    ("أعطني نص المادة {n} من القانون المدني المصري بالعربية والإنجليزية معاً.",
     "العربية:\n{ar}\n\nEnglish:\n{en}"),
]

# --- topic-aware tasks (need a breadcrumb-derived topic) ---------------------

CARD_EN = [
    "Give a reference card for Article {n} of the Egyptian Civil Code: the article number, the topic it falls under, and its text.",
    "Show Article {n} of the Egyptian Civil Code as a structured card (number, topic, text).",
    "Summarise where Article {n} of the Egyptian Civil Code belongs and what it says, as a labelled card.",
]
CARD_AR = [
    "أعطني بطاقة مرجعية للمادة {n} من القانون المدني المصري: رقم المادة، والموضوع الذي تندرج تحته، ونصها.",
    "اعرض المادة {n} من القانون المدني المصري في صورة بطاقة منظَّمة (الرقم، الموضوع، النص).",
    "لخّص في صورة بطاقة موضع المادة {n} من القانون المدني المصري ومضمونها.",
]

CONTRAST_POS_EN = [
    ("Does Article {n} of the Egyptian Civil Code deal with {topic}? Answer yes or no, then say what it covers.",
     "Yes. Article {n} of the Egyptian Civil Code falls under {topic}."),
    ("Is Article {n} of the Egyptian Civil Code about {topic}?",
     "Yes — Article {n} concerns {topic}."),
]
CONTRAST_NEG_EN = [
    ("Does Article {n} of the Egyptian Civil Code deal with {wrong}? Answer yes or no, then say what it actually covers.",
     "No. Article {n} of the Egyptian Civil Code does not deal with {wrong}; it falls under {topic}."),
    ("Is Article {n} of the Egyptian Civil Code about {wrong}?",
     "No — Article {n} is not about {wrong}; it concerns {topic}."),
]
CONTRAST_POS_AR = [
    ("هل تتناول المادة {n} من القانون المدني المصري موضوع {topic}؟ أجب بنعم أو لا ثم بيّن موضوعها.",
     "نعم. المادة {n} من القانون المدني المصري تندرج تحت {topic}."),
    ("هل المادة {n} من القانون المدني المصري تتعلق بـ{topic}؟",
     "نعم — المادة {n} تتعلق بـ{topic}."),
]
CONTRAST_NEG_AR = [
    ("هل تتناول المادة {n} من القانون المدني المصري موضوع {wrong}؟ أجب بنعم أو لا ثم بيّن موضوعها الحقيقي.",
     "لا. المادة {n} من القانون المدني المصري لا تتناول {wrong}؛ بل تندرج تحت {topic}."),
    ("هل المادة {n} من القانون المدني المصري تتعلق بـ{wrong}؟",
     "لا — المادة {n} ليست عن {wrong}؛ بل تتعلق بـ{topic}."),
]

ROSTER_EN = [
    "Which articles of the Egyptian Civil Code deal with {topic}? List the article numbers.",
    "List the Egyptian Civil Code articles that fall under {topic}.",
    "Give the article numbers of the Egyptian Civil Code provisions on {topic}.",
]
ROSTER_AR = [
    "ما المواد التي تتناول {topic} في القانون المدني المصري؟ اذكر أرقام المواد.",
    "اذكر مواد القانون المدني المصري التي تندرج تحت {topic}.",
    "أعطني أرقام مواد القانون المدني المصري المتعلقة بـ{topic}.",
]


def _card_en(n: str, topic: str, gist: str, text: str) -> str:
    lines = [f"### Article Number\n{n}", f"### Topic\n{topic}", f"### Key point\n{gist}"]
    if text.strip() and text.strip() != gist.strip():
        lines.append(f"### Text\n{text}")
    return "\n\n".join(lines)


def _card_ar(n: str, topic: str, gist: str, text: str) -> str:
    lines = [f"### رقم المادة\n{n}", f"### الموضوع\n{topic}", f"### النقطة الأساسية\n{gist}"]
    if text.strip() and text.strip() != gist.strip():
        lines.append(f"### النص\n{text}")
    return "\n\n".join(lines)


def _pick_wrong_topic(rng: random.Random, pool: list[str], by_topic: dict, key: str,
                      own_topic: str | None) -> str | None:
    """A topic from `pool` that does NOT contain this article (and isn't its own)."""
    if not pool:
        return None
    for _ in range(12):
        cand = rng.choice(pool)
        if cand == own_topic:
            continue
        if key in by_topic.get(cand, ()):  # truly unrelated only
            continue
        return cand
    return None


# ---------------------------------------------------------------------------
# record assembly
# ---------------------------------------------------------------------------

def to_record(instruction: str, response: str, *, language: str, kind: str,
              article_key: str | None) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "language": language,
        "kind": kind,
        "article_key": article_key,
    }


def build_for_article(key: str, *, en: str, ar: str, meta: list[str],
                      rng: random.Random, max_quote_chars: int,
                      min_words_cloze: int, topic: dict | None = None,
                      topic_ctx: dict | None = None) -> list[dict]:
    """All knowledge examples for one article. `en`/`ar` are already cleaned.

    `topic` is the dict returned by topic_of(meta); `topic_ctx` carries the
    corpus-wide pools used for contrastive negatives:
        {"all_en": [...], "all_ar": [...], "by_en": {topic: set(keys)}, "by_ar": {...}}
    Both may be None, in which case only the original 7 task families are built.
    """
    n = article_number(key)
    out: list[dict] = []
    topic = topic or {}
    topic_ctx = topic_ctx or {}

    # 1. verbatim recall — two phrasings per available language
    if en:
        for tmpl in rng.sample(VERBATIM_EN, k=min(2, len(VERBATIM_EN))):
            out.append(to_record(tmpl.format(n=n), en, language="en",
                                 kind="kn_verbatim", article_key=key))
    if ar:
        for tmpl in rng.sample(VERBATIM_AR, k=min(2, len(VERBATIM_AR))):
            out.append(to_record(tmpl.format(n=n), ar, language="ar",
                                 kind="kn_verbatim", article_key=key))

    # 2. tail completion — prefix the first ~45-65% of words, recall the whole text
    for text, lang, tmpls in ((en, "en", COMPLETE_EN), (ar, "ar", COMPLETE_AR)):
        if not text:
            continue
        words = text.split()
        if len(words) < min_words_cloze:
            continue
        cut = max(3, round(len(words) * rng.uniform(0.45, 0.65)))
        prefix = truncate(" ".join(words[:cut]), max_quote_chars)
        out.append(to_record(rng.choice(tmpls).format(n=n, prefix=prefix), text,
                             language=lang, kind="kn_complete", article_key=key))

    # 3. fill-the-gap — blank a 3-6 word span from the middle
    for text, lang, tmpls in ((en, "en", GAP_EN), (ar, "ar", GAP_AR)):
        if not text:
            continue
        words = text.split()
        if not (min_words_cloze + 6 <= len(words) <= 200):
            continue
        span_len = rng.randint(3, 6)
        hi = len(words) - span_len - 2
        if hi < 2:
            continue
        start = rng.randint(2, hi)
        missing = " ".join(words[start:start + span_len])
        blanked = " ".join(words[:start] + ["_____"] + words[start + span_len:])
        out.append(to_record(rng.choice(tmpls).format(n=n, blanked=blanked), missing,
                             language=lang, kind="kn_gap", article_key=key))

    # 4. reverse lookup — quote -> article number
    if en:
        q, a = rng.choice(REVERSE_EN)
        out.append(to_record(q.format(quote=truncate(en, max_quote_chars)),
                             a.format(label=en_label(n)),
                             language="en", kind="kn_reverse", article_key=key))
    if ar:
        q, a = rng.choice(REVERSE_AR)
        out.append(to_record(q.format(quote=truncate(ar, max_quote_chars)),
                             a.format(label=ar_label(n)),
                             language="ar", kind="kn_reverse", article_key=key))

    # 5. placement / topic — from the breadcrumb metadata
    crumb_list = [c for c in (m.strip().rstrip(" :.;،-") for m in (meta or []) if (m or "").strip()) if c]
    crumbs = " › ".join(crumb_list)
    if len(crumbs) >= 6:
        q, a = rng.choice(PLACEMENT_EN)
        out.append(to_record(q.format(n=n), a.format(label=en_label(n), crumbs=crumbs),
                             language="en", kind="kn_placement", article_key=key))
        q, a = rng.choice(PLACEMENT_AR)
        out.append(to_record(q.format(n=n), a.format(label=ar_label(n), crumbs=crumbs),
                             language="ar", kind="kn_placement", article_key=key))

    # 6. cross-language — both texts present and not too long to quote in a prompt
    if en and ar and len(en) <= max_quote_chars and len(ar) <= max_quote_chars:
        out.append(to_record(rng.choice(XLAT_AR2EN).format(n=n, ar=ar), en,
                             language="en", kind="kn_translate", article_key=key))
        out.append(to_record(rng.choice(XLAT_EN2AR).format(n=n, en=en), ar,
                             language="ar", kind="kn_translate", article_key=key))
        q, a = rng.choice(BOTH_LANGS)
        out.append(to_record(q.format(n=n), a.format(ar=ar, en=en),
                             language="bi", kind="kn_bilingual", article_key=key))

    # 7. structured reference card — number / topic / key point / text
    if topic.get("en") and en:
        gist = first_sentence(en)
        out.append(to_record(rng.choice(CARD_EN).format(n=n),
                             _card_en(n, topic["en"], gist, en),
                             language="en", kind="kn_card", article_key=key))
    if topic.get("ar") and ar:
        gist = first_sentence(ar)
        out.append(to_record(rng.choice(CARD_AR).format(n=n),
                             _card_ar(n, topic["ar"], gist, ar),
                             language="ar", kind="kn_card", article_key=key))

    # 8. topic <-> article contrastive — one positive + one negative per language,
    #    only when we have an anchorable (specific) topic to bind to.
    if topic.get("specific"):
        if topic.get("en") and topic_ctx.get("all_en"):
            q, a = rng.choice(CONTRAST_POS_EN)
            out.append(to_record(q.format(n=n, topic=topic["en"]),
                                 a.format(n=n, topic=topic["en"]),
                                 language="en", kind="kn_contrast", article_key=key))
            w = _pick_wrong_topic(rng, topic_ctx["all_en"], topic_ctx.get("by_en", {}),
                                  key, topic.get("spec_en") or topic.get("en"))
            if w:
                q, a = rng.choice(CONTRAST_NEG_EN)
                out.append(to_record(q.format(n=n, wrong=w),
                                     a.format(n=n, wrong=w, topic=topic["en"]),
                                     language="en", kind="kn_contrast", article_key=key))
        if topic.get("ar") and topic_ctx.get("all_ar"):
            q, a = rng.choice(CONTRAST_POS_AR)
            out.append(to_record(q.format(n=n, topic=topic["ar"]),
                                 a.format(n=n, topic=topic["ar"]),
                                 language="ar", kind="kn_contrast", article_key=key))
            w = _pick_wrong_topic(rng, topic_ctx["all_ar"], topic_ctx.get("by_ar", {}),
                                  key, topic.get("spec_ar") or topic.get("ar"))
            if w:
                q, a = rng.choice(CONTRAST_NEG_AR)
                out.append(to_record(q.format(n=n, wrong=w),
                                     a.format(n=n, wrong=w, topic=topic["ar"]),
                                     language="ar", kind="kn_contrast", article_key=key))

    return out


def build_topic_rosters(by_en: dict, by_ar: dict, rng: random.Random, *,
                        min_n: int = 2, max_n: int = 40) -> list[dict]:
    """Per-topic 'which articles deal with X?' -> the list of article numbers.

    Only emitted for topics with between `min_n` and `max_n` articles: below 2
    a roster is trivial, above ~40 the label is usually too coarse and the
    answer too long to imprint usefully (the per-article placement task already
    covers those).
    """
    out: list[dict] = []

    def _nums(keys) -> list[str]:
        s = {article_number(k) for k in keys}
        return sorted(s, key=lambda x: int(x) if x.isdigit() else 10 ** 9)

    for topic, keys in sorted(by_en.items()):
        nums = _nums(keys)
        if not (min_n <= len(nums) <= max_n):
            continue
        q = rng.choice(ROSTER_EN).format(topic=topic)
        a = f"The following Egyptian Civil Code articles fall under {topic}: Articles {', '.join(nums)}."
        out.append(to_record(q, a, language="en", kind="kn_roster", article_key=None))
    for topic, keys in sorted(by_ar.items()):
        nums = _nums(keys)
        if not (min_n <= len(nums) <= max_n):
            continue
        q = rng.choice(ROSTER_AR).format(topic=topic)
        a = f"المواد التالية من القانون المدني المصري تندرج تحت {topic}: المواد {'، '.join(nums)}."
        out.append(to_record(q, a, language="ar", kind="kn_roster", article_key=None))
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="YAML config; only its data.train_jsonl / data.val_jsonl are used here.")
    ap.add_argument("--corpus", default=PROJECT_ROOT / "data" / "orig_data.json", type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.04)
    ap.add_argument("--max-articles", type=int, default=0,
                    help="0 = all articles; otherwise process only the first N (smoke test).")
    ap.add_argument("--max-quote-chars", type=int, default=700,
                    help="Truncate article text quoted inside a prompt to this many chars.")
    ap.add_argument("--min-words-cloze", type=int, default=10,
                    help="Skip the completion/gap tasks for articles shorter than this (words).")
    ap.add_argument("--min-text-chars", type=int, default=15,
                    help="Skip an article entirely if it has no text >= this length in either language.")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_jsonl = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_jsonl = PROJECT_ROOT / cfg["data"]["val_jsonl"]

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))

    def _num(k: str) -> int:
        s = article_number(k)
        return int(s) if s.isdigit() else 10 ** 9

    keys = sorted(
        (k for k in corpus if k.startswith("Article") and isinstance(corpus[k], dict)),
        key=_num,
    )
    if args.max_articles:
        keys = keys[:args.max_articles]
    print(f"Corpus: {len(corpus)} keys; {len(keys)} article entries to process")

    # --- pass 1: keep usable articles, derive each one's topic, index topics ---
    arts: list[tuple[str, str, str, list, dict]] = []   # (key, en, ar, meta, topic)
    by_en: dict[str, set] = {}
    by_ar: dict[str, set] = {}
    for k in keys:
        entry = corpus[k]
        en = clean_text(entry.get("english", ""))
        ar = clean_text(entry.get("arabic", ""))
        if len(en) < args.min_text_chars and len(ar) < args.min_text_chars:
            continue
        meta = entry.get("metadata") or []
        topic = topic_of(meta)
        arts.append((k, en, ar, meta, topic))
        if topic.get("spec_en"):
            by_en.setdefault(topic["spec_en"], set()).add(k)
        if topic.get("spec_ar"):
            by_ar.setdefault(topic["spec_ar"], set()).add(k)
    used_articles = len(arts)
    topic_ctx = {
        "all_en": sorted(by_en), "all_ar": sorted(by_ar),
        "by_en": by_en, "by_ar": by_ar,
    }
    print(f"Distinct anchorable topics: EN={len(by_en)}  AR={len(by_ar)}")

    # --- pass 2: per-article example bundles ---
    records: list[dict] = []
    for k, en, ar, meta, topic in arts:
        # per-article RNG so phrasing choices vary across the corpus but stay reproducible
        rng = random.Random(args.seed * 7919 + _num(k))
        records.extend(build_for_article(
            k, en=en, ar=ar, meta=meta,
            rng=rng, max_quote_chars=args.max_quote_chars,
            min_words_cloze=args.min_words_cloze,
            topic=topic, topic_ctx=topic_ctx,
        ))

    # --- pass 3: corpus-level topic rosters (topic -> list of article numbers) ---
    records.extend(build_topic_rosters(by_en, by_ar, random.Random(args.seed * 3 + 1)))

    # dedup on (instruction prefix, response prefix)
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in records:
        sig = (r["messages"][0]["content"][:200], r["messages"][1]["content"][:200])
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)

    by_kind = Counter(r["kind"] for r in unique)
    by_lang = Counter(r["language"] for r in unique)
    print(f"Articles used: {used_articles}")
    print(f"Examples:      {len(unique)}  (≈{len(unique) / max(1, used_articles):.1f} per article)")
    print(f"  by kind: {dict(sorted(by_kind.items()))}")
    print(f"  by lang: {dict(sorted(by_lang.items()))}")

    rng = random.Random(args.seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * args.val_fraction))
    val, train = unique[:n_val], unique[n_val:]
    write_jsonl(train, train_jsonl)
    write_jsonl(val, val_jsonl)
    print(f"Train: {len(train):>6} -> {train_jsonl}")
    print(f"Val:   {len(val):>6} -> {val_jsonl}")


if __name__ == "__main__":
    main()
