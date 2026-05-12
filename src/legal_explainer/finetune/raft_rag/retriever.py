"""Hand-rolled hybrid retriever (BM25 + dense, fused with Reciprocal-Rank Fusion).

Why hand-rolled and not LlamaIndex/Haystack: the corpus is ~1,100 tiny, fully
structured records, and the one thing that *must* be exactly right is that the
prompt fed to the RAFT adapter matches the training-time context-block format
(``dataset_builder._format_context_block``) — which is trivial to guarantee when
we own every step. A vector framework would be solving a problem we don't have.
The ``Retriever`` ABC keeps the door open to swap in another implementation
(e.g. the teammate's root-level RAG retriever) behind the same ``retrieve()``.
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .index import ArticleEntry, ArticleIndex, detect_lang, tokenize

# "Article 775", "article  775", "المادة ٧٧٥" — leading zeros and Arabic-Indic digits ok
_ARTICLE_NUM_RE = re.compile(r"(?:article|art\.?|المادة|مادة)\s*[#:]?\s*0*([0-9٠-٩]{1,4})", re.IGNORECASE)
_AR2EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass
class RetrievalResult:
    oracle: list[ArticleEntry]                       # article(s) to ground the answer on
    distractors: list[ArticleEntry] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    explicit_id: str | None = None                   # set if the query named an article number

    @property
    def context_entries(self) -> list[ArticleEntry]:
        return list(self.oracle) + list(self.distractors)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, lang: str | None = None, k: int = 1,
                 n_distractors: int = 1) -> RetrievalResult:
        ...


def _rrf(rankings: list[list[int]], rrf_k: int = 60) -> dict[int, float]:
    """Reciprocal-Rank Fusion of several ranked lists of doc indices (best first)."""
    out: dict[int, float] = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            out[idx] = out.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return out


def extract_article_id(query: str) -> str | None:
    m = _ARTICLE_NUM_RE.search(query or "")
    if not m:
        return None
    digits = m.group(1).translate(_AR2EN_DIGITS)
    return str(int(digits)) if digits.isdigit() else None


class HybridRetriever(Retriever):
    def __init__(self, index: ArticleIndex, *, rrf_k: int = 60, candidate_pool: int = 20,
                 distractor_seed: int = 13):
        self.index = index
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self._rng = random.Random(distractor_seed)

    @classmethod
    def load(cls, artifacts_dir, **kw) -> "HybridRetriever":
        return cls(ArticleIndex.load(artifacts_dir), **kw)

    # --- ranking primitives -----------------------------------------------
    def _dense_ranking(self, query: str) -> list[int]:
        if self.index.dense is None:
            return []
        q = self.index.encode_query(query)               # (D,), normalised
        sims = self.index.dense @ q                       # cosine (both normalised)
        return np.argsort(-sims)[: self.candidate_pool].tolist()

    def _sparse_ranking(self, query: str) -> list[int]:
        if self.index.bm25 is None:
            return []
        scores = np.asarray(self.index.bm25.get_scores(tokenize(query)), dtype=np.float32)
        return np.argsort(-scores)[: self.candidate_pool].tolist()

    def _explicit_idx(self, query: str) -> int | None:
        aid = extract_article_id(query)
        if aid is None:
            return None
        e = self.index.get(f"Article {aid}")
        return self.index.entries.index(e) if e is not None else None

    # --- API ---------------------------------------------------------------
    def retrieve(self, query: str, *, lang: str | None = None, k: int = 1,
                 n_distractors: int = 1) -> RetrievalResult:
        lang = lang or detect_lang(query)
        entries = self.index.entries
        n = len(entries)

        explicit = self._explicit_idx(query)
        dense_r = self._dense_ranking(query)
        sparse_r = self._sparse_ranking(query)
        fused = _rrf([r for r in (dense_r, sparse_r) if r], self.rrf_k)
        ranked = sorted(fused, key=lambda i: -fused[i])

        oracle_idx: list[int] = []
        if explicit is not None:
            oracle_idx.append(explicit)
        for i in ranked:
            if len(oracle_idx) >= k:
                break
            if i not in oracle_idx:
                oracle_idx.append(i)
        if not oracle_idx:                                # no signal at all -> first k
            oracle_idx = list(range(min(k, n)))

        distractor_idx: list[int] = []
        for i in ranked:                                  # next-best hits first
            if len(distractor_idx) >= n_distractors:
                break
            if i not in oracle_idx and i not in distractor_idx:
                distractor_idx.append(i)
        guard = 0
        while len(distractor_idx) < n_distractors and len(oracle_idx) + len(distractor_idx) < n and guard < 10 * n:
            guard += 1
            j = self._rng.randrange(n)
            if j not in oracle_idx and j not in distractor_idx:
                distractor_idx.append(j)

        return RetrievalResult(
            oracle=[entries[i] for i in oracle_idx],
            distractors=[entries[i] for i in distractor_idx],
            scores={
                "explicit": explicit is not None,
                "fused_top": {entries[i].key: round(fused.get(i, 0.0), 5) for i in ranked[:10]},
            },
            explicit_id=(entries[explicit].article_id if explicit is not None else None),
        )

    def force(self, key: str, *, n_distractors: int = 1) -> RetrievalResult:
        """Build a result with `key` as the oracle and random distractors — used by
        the 'oracle' eval mode to isolate adapter quality from retrieval quality."""
        e = self.index.get(key)
        pool = [x for x in self.index.entries if x.key != key]
        ds = self._rng.sample(pool, k=min(n_distractors, len(pool)))
        return RetrievalResult(oracle=([e] if e else []), distractors=ds,
                               explicit_id=(e.article_id if e else None))
