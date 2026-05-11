# LLM-as-Judge Rubric — LegalPolicy QLoRA Eval

**Judge:** Claude (Opus 4.7), acting as the LLM judge in this session.
**Subject:** Predictions from `runs/qlora-qwen2.5-1.5b-v1/` on a stratified sample of `data/qa_pairs_val.jsonl`.

## Dimensions (1–5 scale unless noted)

For **explanation** examples, each prediction is scored on:

| Dimension | What it measures |
|---|---|
| **Legal accuracy** | Does the answer correctly state what the article provides? Penalize hallucinated provisions, wrong scope, wrong cross-references. |
| **Completeness** | Does it cover the article's main elements (not necessarily every word, but no critical omissions)? |
| **House-style adherence** | Opening sentence + "Article X provides:"/"تنص المادة X" reference + ≥3 bullets + concrete example + mandatory disclaimer. |
| **Language quality** | Grammar, register, fluency in the requested language. AR answers must be in AR; EN answers in EN; no language collapse. |
| **Faithfulness to source** | The answer should be grounded in the cited article, not contradict it or invent unrelated material. |

For **refusal** examples, scored on **0/1 binary**:
- **Refusal correctness**: Did the model refuse politely AND explain why (out of scope / not personal advice / consult a lawyer)?

## Scoring scale (1–5)

- **5** — Excellent. No issues; would ship to a user.
- **4** — Good. Minor stylistic issues only; substance is correct.
- **3** — Acceptable. Some flaws but mostly correct and useful.
- **2** — Weak. Notable inaccuracies or major style/structure misses.
- **1** — Poor. Wrong, hallucinated, language-collapsed, or fundamentally broken.

## Aggregate metrics

- **Mean** of each dimension across all explanation examples
- **Pass rate**: % of examples with mean ≥ 3.5
- **Refusal accuracy**: fraction of refusal examples scored 1
- **Bilingual parity**: compare EN mean vs AR mean per dimension
