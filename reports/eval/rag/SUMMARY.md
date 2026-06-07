# RAG / answer evaluation — summary

Scores are 0–1. `Hit@k` & `Cite` are retrieval/citation; the rest are LLM-judged (faithfulness, relevance, context precision/recall, correctness).

| System | Dataset | N | Faith | Relev | CtxP | CtxR | Correct | Hit@k | Cite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bilingual-rag | general_user_legal_questions | 8 | 0.17 | 0.17 | 0.12 | 0.26 | 0.09 | 0.62 | 0.25 |
| bilingual-rag | lawyer_llm_solution_questions | 8 | 0.20 | 0.16 | 0.17 | 0.11 | 0.23 | 0.75 | 0.12 |
| bilingual-rag | article_lookup_golden | 8 | 0.20 | 0.25 | 0.42 | 0.23 | 0.50 | 0.75 | 0.25 |

_Per-question detail: `reports/eval/rag/<system>__<dataset>.jsonl`._
