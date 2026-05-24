"""Rule-based safety pre-check.

Cheap, deterministic, no LLM. Returns a `SafetyVerdict` the orchestrator can
either honor (block) or override (pass through). The goal is not to be
exhaustive — it's to catch the obvious off-policy queries before they reach
the model.

If you find yourself wanting to add lots of patterns here, that's a sign
this should be replaced with a model-based classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["allow", "refuse"]


@dataclass
class SafetyVerdict:
    verdict: Verdict
    reason: str | None
    suggested_response: str | None


# Patterns covering things this agent shouldn't engage with directly.
# Mixed AR + EN — both languages appear in real queries.
_REFUSE_PATTERNS = [
    # Specific-case advice
    (
        r"\b(?:my case|in my situation|i was (?:arrested|sued|fired)|"
        r"should i (?:sue|plead|sign))\b",
        "specific_case_advice",
    ),
    # Requests to draft binding documents
    (
        r"\bdraft (?:a|the|me) (?:contract|will|nda|agreement|complaint)\b",
        "binding_document_drafting",
    ),
    # Out-of-jurisdiction requests
    (
        r"\b(?:us(?:\s|c)?\s*title|gdpr article|california|new york|federal court)\b",
        "out_of_jurisdiction",
    ),
    # Personal-data / identity
    (
        r"\b(?:my (?:national id|social security|passport)|"
        r"رقم بطاقتي|بياناتي الشخصية)\b",
        "personal_data_request",
    ),
]

_REFUSAL_TEMPLATES = {
    "specific_case_advice": (
        "I can explain how the Egyptian Civil Code treats this kind of "
        "situation in general terms, but I can't advise on your specific "
        "case. For that, consult a qualified Egyptian attorney."
    ),
    "binding_document_drafting": (
        "I don't draft binding legal documents. I can explain the relevant "
        "provisions of the Egyptian Civil Code; consult a licensed attorney "
        "for the actual drafting."
    ),
    "out_of_jurisdiction": (
        "This system covers the Egyptian Civil Code only. For US, EU, or "
        "other foreign-law questions, consult sources or counsel specific "
        "to that jurisdiction."
    ),
    "personal_data_request": (
        "I don't handle personal identifiers. Please ask in general terms "
        "without sharing private data."
    ),
}


def check_safety(query_text: str) -> SafetyVerdict:
    q = query_text.lower()
    for pattern, tag in _REFUSE_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            return SafetyVerdict(
                verdict="refuse",
                reason=tag,
                suggested_response=_REFUSAL_TEMPLATES[tag],
            )
    return SafetyVerdict(verdict="allow", reason=None, suggested_response=None)