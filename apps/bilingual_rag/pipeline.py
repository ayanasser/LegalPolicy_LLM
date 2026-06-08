"""
Bilingual RAG pipeline over the Egyptian Civil Code (Arabic + English).

A faithful, importable port of the Kaggle notebook
`notebooks/bilingual-rag-system-over-the-egyptian-civil-code_fixed_trial.ipynb`.
The notebook indexed into Chroma + Qdrant + (optional) Elasticsearch; this module
keeps the default path — **Chroma** (persistent) — and drops the other two
backends to keep the standalone service simple and reproducible.

Flow (identical to the notebook's `rag_pipeline`):
    question
      → exact article-number lookup if the question names one (e.g. "المادة 446")
      → else: keyword extraction (Qwen via Ollama)
              → BGE-M3 vector search over Chroma
              → (optional) multilingual cross-encoder rerank against the question
      → grounded bilingual prompt
      → Qwen answer (Ollama)

Heavy objects (embedder, Chroma collection, reranker, Ollama client) are lazy
singletons so importing this module is cheap.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import BRagSettings, get_settings

# ── Text cleaning / chunking (verbatim from the notebook) ─────────────────────

_KEEP_RE = re.compile(r"[^؀-ۿA-Za-z0-9\s\.,;:?!\-\(\)\"'،؛؟]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)
    text = _KEEP_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentence-aware sliding-window chunker (AR + EN sentence delimiters)."""
    if len(text) <= chunk_size:
        return [text]
    sentences = re.split(r"(?<=[\.!\?؟؛])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size - overlap):
                    chunks.append(sent[i:i + chunk_size])
                current = ""
            else:
                tail = current[-overlap:] if current and overlap > 0 else ""
                current = (tail + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks


def load_records(json_path: str) -> list[dict[str, Any]]:
    """Read the `"Article N" -> {arabic, english, metadata}` corpus into flat,
    per-language records (skips __metadata__ / __unmatched_text__)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict[str, Any]] = []
    for key, value in data.items():
        if not key.startswith("Article") or not isinstance(value, dict):
            continue
        m = re.search(r"(\d+)", key)
        article_num = int(m.group(1)) if m else -1
        meta_str = " | ".join(value.get("metadata", []) or [])
        for lang_key, lang_code in [("arabic", "ar"), ("english", "en")]:
            text = (value.get(lang_key) or "").strip()
            if not text:
                continue
            records.append({
                "id": f"art{article_num}_{lang_code}",
                "article_number": article_num,
                "language": lang_code,
                "section_path": meta_str,
                "text": text,
            })
    return records


def detect_language(text: str) -> str:
    """Any Arabic letter → 'ar', else 'en'."""
    for ch in text:
        if "؀" <= ch <= "ۿ":
            return "ar"
    return "en"


# ── Exact article-number lookup (verbatim from the *_fixed_trial notebook) ─────
# Dense/semantic search is great for *meaning* but weak at matching an exact
# article number (e.g. "نص المادة 446") — it returns semantically-near articles
# instead of the exact one. When the question names a specific article number we
# fall back to a direct metadata lookup on `article_number` in the vector DB.

# Map Arabic-Indic digits (٠-٩) → ASCII so "٤٤٦" == "446".
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def extract_article_numbers(question: str) -> list[int]:
    """Detect explicit article numbers in a question (Arabic OR English, any digit script)."""
    q = question.translate(_AR_DIGITS)
    nums: list[int] = []
    has_article_word = bool(re.search(r"المادة|مادة|article|art", q, re.IGNORECASE))
    if has_article_word:
        # An 'article' word is present → treat every standalone number as an article
        # number. Handles "المادة 446", "articles 3 and 7", "المادة 3 و 7", etc.
        nums = [int(x) for x in re.findall(r"\d{1,4}", q)]
    else:
        # No 'article' word: only trust a number glued to an 'article'-like token.
        for m in re.finditer(r"(?:المادة|مادة|articles?|art\.?)\s*[#:\-]?\s*(\d{1,4})",
                             q, re.IGNORECASE):
            nums.append(int(m.group(1)))
    # de-duplicate, preserve order
    seen, out = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ── Pipeline ──────────────────────────────────────────────────────────────────

class BilingualRAGPipeline:
    def __init__(self, settings: BRagSettings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._embedder = None
        self._chroma_collection = None
        self._reranker = None
        self._ollama = None

    # ── Lazy heavy objects ────────────────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[brag] loading embedder {self.cfg.embed_model_name} on {self.cfg.embed_device} …")
            # BAAI/bge-m3 ships only pytorch_model.bin (no safetensors on the hub).
            # transformers 5.x refuses torch.load on torch < 2.6 (CVE-2025-32434),
            # so we load from the local snapshot (where a converted model.safetensors
            # lives — see scripts/ensure_bge_m3_safetensors.py) and force safetensors.
            model_ref = self.cfg.embed_model_name
            try:
                from huggingface_hub import snapshot_download
                model_ref = snapshot_download(model_ref, local_files_only=True)
            except Exception:
                pass  # not cached locally — fall back to the hub id
            self._embedder = SentenceTransformer(
                model_ref, device=self.cfg.embed_device,
                model_kwargs={"use_safetensors": True},
            )
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"[brag] loading reranker {self.cfg.reranker_name} on cpu …")
            self._reranker = CrossEncoder(self.cfg.reranker_name, device="cpu")
        return self._reranker

    @property
    def ollama(self):
        if self._ollama is None:
            import ollama as ollama_lib
            self._ollama = ollama_lib.Client(host=self.cfg.ollama_host)
        return self._ollama

    def _open_collection(self, create: bool = False):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        Path(self.cfg.chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=self.cfg.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        names = [c.name for c in client.list_collections()]
        if create:
            if self.cfg.collection_name in names:
                client.delete_collection(self.cfg.collection_name)
            return client.create_collection(
                name=self.cfg.collection_name, metadata={"hnsw:space": "cosine"}
            )
        if self.cfg.collection_name not in names:
            raise RuntimeError(
                f"Chroma collection '{self.cfg.collection_name}' not found in "
                f"{self.cfg.chroma_dir}. Build it first:  "
                f"python -m apps.bilingual_rag.build_index"
            )
        return client.get_collection(self.cfg.collection_name)

    @property
    def collection(self):
        if self._chroma_collection is None:
            self._chroma_collection = self._open_collection(create=False)
        return self._chroma_collection

    # ── Index builder ─────────────────────────────────────────────────────────

    def build_index(self) -> int:
        """Embed every article chunk and (re)create the persistent Chroma store.
        Returns the number of chunks indexed."""
        records = load_records(self.cfg.corpus_path)
        chunked: list[dict[str, Any]] = []
        for rec in records:
            cleaned = clean_text(rec["text"])
            if not cleaned:
                continue
            pieces = chunk_text(cleaned, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for idx, piece in enumerate(pieces):
                chunked.append({
                    "id": f"{rec['id']}_c{idx}",
                    "text": piece,
                    "article_number": rec["article_number"],
                    "language": rec["language"],
                    "section_path": rec["section_path"],
                    "chunk_index": idx,
                    "chunk_total": len(pieces),
                })
        print(f"[brag] {len(records)} records → {len(chunked)} chunks; embedding …")

        texts = [c["text"] for c in chunked]
        embeddings = self.embedder.encode(
            texts, batch_size=16, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )

        collection = self._open_collection(create=True)
        ids = [c["id"] for c in chunked]
        metadatas = [{
            "chunk_id": c["id"], "text": c["text"],
            "article_number": c["article_number"], "language": c["language"],
            "section_path": c["section_path"],
            "chunk_index": c["chunk_index"], "chunk_total": c["chunk_total"],
        } for c in chunked]
        bs = self.cfg.insert_batch_size
        for start in range(0, len(ids), bs):
            end = start + bs
            collection.add(
                ids=ids[start:end], documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end].tolist(),
            )
        self._chroma_collection = collection
        print(f"[brag] indexed {collection.count()} chunks into '{self.cfg.collection_name}'.")
        return collection.count()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def vector_search(self, query: str, k: int = 5, language: str | None = None) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        where = {"language": language} if language else None
        res = self.collection.query(query_embeddings=q_emb, n_results=k, where=where)
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({
                "text": doc,
                "article_number": meta["article_number"],
                "language": meta["language"],
                "section_path": meta.get("section_path", ""),
                "score": 1 - dist,  # cosine distance → similarity
            })
        return hits

    def fetch_articles_by_number(self, numbers: list[int], language: str | None = None,
                                 max_chunks: int = 12) -> list[dict]:
        """Direct exact lookup of article(s) by `article_number` from Chroma metadata.
        Returns hits in the same shape as vector_search (score = 1.0, exact match)."""
        if not numbers:
            return []
        where: dict[str, Any] = {"article_number": {"$in": numbers}}
        if language:
            where = {"$and": [{"article_number": {"$in": numbers}}, {"language": language}]}
        res = self.collection.get(where=where, include=["documents", "metadatas"])
        hits: list[dict] = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            hits.append({
                "text": doc,
                "article_number": meta["article_number"],
                "language": meta["language"],
                "section_path": meta.get("section_path", ""),
                "score": 1.0,
                "_ci": meta.get("chunk_index", 0),
            })
        # Order by the requested article order, then chunk index (multi-chunk reads in order).
        order = {n: i for i, n in enumerate(numbers)}
        hits.sort(key=lambda h: (order.get(h["article_number"], 999), h["_ci"]))
        for h in hits:
            h.pop("_ci", None)
        return hits[:max_chunks]

    def extract_keywords(self, question: str, max_keywords: int = 8) -> list[str]:
        kw_system = (
            "You are a keyword extraction engine for a legal search system. "
            "Extract the most important search keywords/phrases from the user's question. "
            "Keep legal terms, entities and topic words; drop stopwords and filler. "
            "Keep each keyword in the SAME language as the question. "
            f"Return ONLY a JSON array of at most {max_keywords} short strings, nothing else.\n\n"
            "أنت محرّك لاستخراج الكلمات المفتاحية لنظام بحث قانوني. "
            "استخرج أهم الكلمات أو العبارات المفتاحية من سؤال المستخدم، "
            "واحتفظ بالمصطلحات القانونية والكيانات والكلمات الموضوعية واحذف كلمات الوقف والحشو. "
            "اجعل كل كلمة بنفس لغة السؤال. "
            f"أعد فقط مصفوفة JSON تحتوي على {max_keywords} عناصر كحدّ أقصى، دون أي نص آخر."
        )
        raw = self._chat(kw_system, f"Question / السؤال: {question}\n\nKeywords JSON:")
        keywords: list[str] = []
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                keywords = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            except Exception:
                keywords = []
        if not keywords:
            keywords = [w for w in re.split(r"[\s،,]+", question) if len(w) > 1]
        seen, out = set(), []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                out.append(kw)
        return out[:max_keywords]

    # ── LLM ───────────────────────────────────────────────────────────────────

    def _chat(self, system: str, user: str) -> str:
        resp = self.ollama.chat(
            model=self.cfg.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": self.cfg.llm_temperature,
                "num_predict": self.cfg.llm_num_predict,
            },
        )
        return resp["message"]["content"].strip()

    def _chat_stream(self, system: str, user: str):
        """Yield incremental answer text from the Ollama LLM (token streaming)."""
        for part in self.ollama.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": self.cfg.llm_temperature,
                     "num_predict": self.cfg.llm_num_predict},
            stream=True,
        ):
            chunk = part["message"]["content"]
            if chunk:
                yield chunk

    @staticmethod
    def build_rag_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
        system_prompt = (
            "You are a precise, trustworthy legal assistant specialised in the Egyptian Civil Code "
            "(القانون المدني المصري). Follow these rules strictly:\n"
            "1. Answer using the provided context excerpts as your source of law. Do NOT add facts "
            "from outside knowledge, but DO reason over the excerpts to apply them to the question.\n"
            "2. The user may ask in colloquial/Egyptian dialect (عامية) while the articles are in "
            "formal Arabic. Map the everyday wording to the legal concept (e.g. 'استعملت حقي وضرّيت "
            "حد' → التعسف في استعمال الحق) and answer from the matching article(s).\n"
            "3. Always cite the relevant article numbers inline, e.g. (Article 12) / (المادة 12).\n"
            "4. Only if NONE of the excerpts relate to the question at all, say exactly:\n"
            "   - EN: 'The provided articles do not contain enough information to answer this question.'\n"
            "   - AR: 'لا تحتوي المواد المقدمة على معلومات كافية للإجابة على هذا السؤال.'\n"
            "   Do NOT use this refusal when a relevant article is present — give the answer instead.\n"
            "5. Never invent articles, numbers, or facts.\n"
            "6. Reply in the SAME language as the user's question.\n"
            "7. Be clear, concise, and accurate.\n\n"
            "أنت مساعد قانوني دقيق وموثوق متخصص في القانون المدني المصري. التزم بالقواعد التالية بدقة:\n"
            "1. أجب اعتمادًا على المقتطفات المرفقة بوصفها مصدر القانون، ولا تُضِف وقائع من معرفة خارجية، "
            "لكن استنتج من المقتطفات وطبّقها على السؤال.\n"
            "2. قد يسأل المستخدم بالعامية المصرية بينما المواد مكتوبة بالعربية الفصحى. ترجم الصياغة "
            "الدارجة إلى المفهوم القانوني (مثال: 'استعملت حقي وضرّيت حد' ← التعسف في استعمال الحق) "
            "ثم أجب من المادة المناسبة.\n"
            "3. اذكر دائمًا أرقام المواد ذات الصلة داخل الإجابة، مثل (المادة 12).\n"
            "4. فقط إذا لم تكن أي من المقتطفات ذات صلة بالسؤال إطلاقًا، فاكتب بالضبط: "
            "'لا تحتوي المواد المقدمة على معلومات كافية للإجابة على هذا السؤال.' "
            "ولا تستخدم هذه الجملة إذا كانت هناك مادة ذات صلة، بل قدّم الإجابة.\n"
            "5. لا تختلق أي مواد أو أرقام أو معلومات.\n"
            "6. أجب بنفس لغة سؤال المستخدم.\n"
            "7. كن واضحًا ومختصرًا ودقيقًا."
        )
        context_blocks = [
            f"[{i}] (Article {h['article_number']} | {h['language']})\n{h['text']}"
            for i, h in enumerate(hits, 1)
        ]
        context_str = "\n\n".join(context_blocks)
        user_prompt = (
            "Context excerpts from the Egyptian Civil Code / مقتطفات من القانون المدني المصري:\n"
            f"---\n{context_str}\n---\n\n"
            f"Question / السؤال: {question}\n\n"
            "Answer (with article citations) / الإجابة (مع ذكر أرقام المواد):"
        )
        return system_prompt, user_prompt

    # ── End-to-end ────────────────────────────────────────────────────────────

    def retrieve(self, question: str, k: int | None = None,
                 candidate_k: int | None = None, use_rerank: bool | None = None,
                 restrict_language: bool | None = None,
                 use_keywords: bool | None = None) -> dict:
        k = k or self.cfg.top_k
        candidate_k = candidate_k or self.cfg.candidate_k
        use_rerank = self.cfg.use_reranker if use_rerank is None else use_rerank
        restrict_language = self.cfg.restrict_language if restrict_language is None else restrict_language
        use_keywords = self.cfg.use_keywords if use_keywords is None else use_keywords

        lang = detect_language(question) if restrict_language else None

        # Step 0: if the question names a specific article number, fetch it DIRECTLY
        # by metadata (exact match). Semantic search can't reliably match a number
        # like 446, so this guarantees the exact article is returned.
        article_numbers = extract_article_numbers(question)
        if article_numbers:
            hits = self.fetch_articles_by_number(article_numbers, language=lang)
            if hits:
                return {
                    "question": question,
                    "keywords": [f"المادة {n}" for n in article_numbers],
                    "article_numbers": article_numbers,
                    "search_query": f"article={article_numbers}",
                    "detected_language": lang, "hits": hits,
                }

        keywords = None
        search_query = question
        if use_keywords:
            keywords = self.extract_keywords(question)
            if keywords:
                search_query = " ".join(keywords)

        if use_rerank:
            candidates = self.vector_search(search_query, k=candidate_k, language=lang)
            if candidates:
                pairs = [(question, c["text"]) for c in candidates]
                for c, s in zip(candidates, self.reranker.predict(pairs)):
                    c["rerank_score"] = float(s)
                candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
            hits = candidates[:k]
        else:
            hits = self.vector_search(search_query, k=k, language=lang)

        return {
            "question": question, "keywords": keywords, "article_numbers": [],
            "search_query": search_query, "detected_language": lang, "hits": hits,
        }

    def answer(self, question: str, **kw) -> dict:
        t0 = time.perf_counter()
        r = self.retrieve(question, **kw)
        system_prompt, user_prompt = self.build_rag_prompt(question, r["hits"])
        r["answer"] = self._chat(system_prompt, user_prompt)
        r["processing_time_ms"] = round((time.perf_counter() - t0) * 1000)
        return r

    def answer_stream(self, question: str, **kw):
        """Token-streaming generator. Yields event dicts: a `meta` line (hits +
        keywords + detected language), then `delta` lines, then a `done` line."""
        t0 = time.perf_counter()
        r = self.retrieve(question, **kw)
        yield {"type": "meta", "hits": r["hits"], "keywords": r.get("keywords"),
               "article_numbers": r.get("article_numbers") or [],
               "search_query": r.get("search_query", ""),
               "detected_language": r.get("detected_language")}
        system_prompt, user_prompt = self.build_rag_prompt(question, r["hits"])
        for delta in self._chat_stream(system_prompt, user_prompt):
            yield {"type": "delta", "text": delta}
        yield {"type": "done", "processing_time_ms": round((time.perf_counter() - t0) * 1000)}
