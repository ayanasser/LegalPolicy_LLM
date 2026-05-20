"""Build & persist the RAFT-RAG article index (hybrid: BM25 + optional dense).

The index is one record per Egyptian Civil Code article (EN + AR text, the
breadcrumb metadata, and the breadcrumb-derived topic), plus:
  * a BM25Okapi over a per-article "search blob" (text + metadata + "Article N"
    / "المادة N") — so exact terms like an article number, "suretyship",
    "الكفالة" are found reliably, which dense retrieval alone is weak at;
  * (optional) a dense passage-embedding matrix from a small multilingual
    sentence-transformer, L2-normalised so cosine == dot product.

Persisted under ``artifacts/raft_rag/``:
  records.jsonl   one ArticleEntry dict per line (article order)
  bm25.pkl        {"bm25": BM25Okapi, "tokenized": [[tok,...],...]}
  dense.npy       float32 (N, D), L2-normalised  (absent if built --no-dense)
  index_meta.json {embedder, query_prefix, passage_prefix, dim, n_articles, ...}

No GPU is needed; the embedder runs fine on CPU for ~1,100 short passages.
Requires ``rank-bm25`` (pure-python) and, for the dense part, ``sentence-transformers``.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# .../src/legal_explainer/finetune/raft_rag/index.py  ->  .../LegalPolicy_LLM
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "orig_data.json"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "raft_rag"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_AR_RE = re.compile(r"[؀-ۿ]")


def tokenize(text: str) -> list[str]:
    """Unicode-aware word tokeniser (works for Arabic and Latin) for BM25."""
    return _WORD_RE.findall((text or "").lower())


def detect_lang(text: str) -> str:
    """'ar' if the text is predominantly Arabic script, else 'en'."""
    if not text:
        return "en"
    non_space = len((text or "").replace(" ", "")) or 1
    return "ar" if len(_AR_RE.findall(text)) >= 0.20 * non_space else "en"


def _article_id(key: str) -> str:
    return key.replace("Article", "").strip()


@dataclass(frozen=True)
class ArticleEntry:
    key: str            # "Article 775"
    article_id: str     # "775"
    english: str
    arabic: str
    metadata: tuple     # breadcrumb crumbs from orig_data.json
    topic_en: str | None
    topic_ar: str | None

    def text(self, lang: str) -> str:
        primary = self.arabic if lang == "ar" else self.english
        return (primary or "").strip() or (self.english or self.arabic or "").strip()

    def topic(self, lang: str) -> str | None:
        return self.topic_ar if lang == "ar" else self.topic_en

    def label(self, lang: str) -> str:
        return f"المادة {self.article_id}" if lang == "ar" else f"Article {self.article_id}"

    def search_blob(self) -> str:
        return " ".join(p for p in (
            f"Article {self.article_id}", f"المادة {self.article_id}",
            self.english, self.arabic, " ".join(self.metadata or ()),
            self.topic_en or "", self.topic_ar or "",
        ) if p)


class ArticleIndex:
    """A loaded index: records + BM25 + (optional) dense matrix + embedder handle."""

    def __init__(self, entries: list[ArticleEntry], dense, bm25, tokenized_corpus, meta: dict):
        self.entries = entries
        self.by_key = {e.key: e for e in entries}
        self.dense = dense                       # np.ndarray (N, D) float32 normalised, or None
        self.bm25 = bm25                         # rank_bm25.BM25Okapi, or None
        self.tokenized_corpus = tokenized_corpus
        self.meta = dict(meta or {})
        self._embedder = None

    # --- persistence -------------------------------------------------------
    def save(self, out_dir) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
            for e in self.entries:
                d = asdict(e)
                d["metadata"] = list(e.metadata)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if self.dense is not None:
            np.save(out_dir / "dense.npy", np.asarray(self.dense, dtype=np.float32))
        if self.bm25 is not None:
            with (out_dir / "bm25.pkl").open("wb") as f:
                pickle.dump({"bm25": self.bm25, "tokenized": self.tokenized_corpus}, f)
        (out_dir / "index_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, in_dir) -> "ArticleIndex":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "index_meta.json").read_text(encoding="utf-8"))
        entries: list[ArticleEntry] = []
        for line in (in_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            entries.append(ArticleEntry(
                key=d["key"], article_id=d["article_id"],
                english=d.get("english", ""), arabic=d.get("arabic", ""),
                metadata=tuple(d.get("metadata") or ()),
                topic_en=d.get("topic_en"), topic_ar=d.get("topic_ar"),
            ))
        dense = None
        if (in_dir / "dense.npy").exists():
            dense = np.load(in_dir / "dense.npy")
        bm25 = tokenized = None
        if (in_dir / "bm25.pkl").exists():
            with (in_dir / "bm25.pkl").open("rb") as f:
                blob = pickle.load(f)
            bm25, tokenized = blob.get("bm25"), blob.get("tokenized")
        return cls(entries, dense, bm25, tokenized, meta)

    # --- lookups & query encoding -----------------------------------------
    def get(self, key: str) -> ArticleEntry | None:
        return self.by_key.get(key)

    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.meta["embedder"],
                                                 device=self.meta.get("device", "cpu"))
        return self._embedder

    def encode_query(self, query: str) -> np.ndarray:
        prefix = self.meta.get("query_prefix", "") or ""
        v = self.embedder().encode([prefix + query], normalize_embeddings=True,
                                   show_progress_bar=False)
        return np.asarray(v[0], dtype=np.float32)


def build_index(corpus_path=DEFAULT_CORPUS, *, embedder: str = "intfloat/multilingual-e5-small",
                query_prefix: str = "query: ", passage_prefix: str = "passage: ",
                device: str = "cpu", batch_size: int = 32, with_dense: bool = True) -> ArticleIndex:
    """Build (in memory) the article index from orig_data.json. ``save(...)`` to persist."""
    from rank_bm25 import BM25Okapi  # pure-python, tiny
    from legal_explainer.finetune.knowledge_builder import clean_text, topic_of

    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    entries: list[ArticleEntry] = []
    for key, v in corpus.items():
        if not (key.startswith("Article") and isinstance(v, dict)):
            continue
        en = clean_text(v.get("english", ""))
        ar = clean_text(v.get("arabic", ""))
        if not en and not ar:
            continue
        t = topic_of(v.get("metadata") or [])
        entries.append(ArticleEntry(
            key=key, article_id=_article_id(key), english=en, arabic=ar,
            metadata=tuple(v.get("metadata") or ()), topic_en=t.get("en"), topic_ar=t.get("ar"),
        ))
    entries.sort(key=lambda e: int(e.article_id) if e.article_id.isdigit() else 10 ** 9)

    tokenized = [tokenize(e.search_blob()) for e in entries]
    bm25 = BM25Okapi(tokenized)

    dense, dim = None, None
    if with_dense:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(embedder, device=device)
        passages = [(passage_prefix or "") + (e.text("en") + " " + e.text("ar")).strip() for e in entries]
        dense = np.asarray(st.encode(passages, normalize_embeddings=True, batch_size=batch_size,
                                     show_progress_bar=True), dtype=np.float32)
        dim = int(dense.shape[1])

    meta = {
        "embedder": embedder, "query_prefix": query_prefix, "passage_prefix": passage_prefix,
        "device": device, "dim": dim, "n_articles": len(entries), "with_dense": bool(with_dense),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return ArticleIndex(entries, dense, bm25, tokenized, meta)
