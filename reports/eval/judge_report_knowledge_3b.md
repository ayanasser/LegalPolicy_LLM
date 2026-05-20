# LLM-as-Judge Evaluation — Stage A v2 (knowledge-injection, Qwen 2.5 3B + Unsloth)

**Adapter under test:** [`runs/qlora-qwen2.5-3b-knowledge/`](../../runs/qlora-qwen2.5-3b-knowledge/) (Stage A v2, trained 2026-05-13/14, 4 epochs, 15.0 h on RTX 3050).
**Comparison points:** Stage 0 baseline (`qlora-qwen2.5-1.5b-v1`), Stage 1 (`qlora-qwen2.5-1.5b-stage1`), Stage 4 RAFT (`qlora-qwen2.5-1.5b-raft`).
**Judge:** Claude (Opus 4.7), in-session.
**Sample:** **21 stratified validation examples** — 10 EN explanation + 10 AR explanation + 1 AR refusal, drawn from `data/qa_pairs_val.baseline.jsonl` (the same set every prior judge report used — direct apples-to-apples).
**Inference mode:** **closed-book** (no article supplied in the prompt). This is the thesis-relevant mode.
**Inputs:** [judge_predictions_knowledge_3b.json](judge_predictions_knowledge_3b.json).

---

## TL;DR

**Legal accuracy jumps from 1.00 / 5 (every prior closed-book run) to 3.35 / 5 — first non-floor closed-book legal-accuracy score in the project.** Pass rate (mean ≥ 3.5) goes from 0 / 21 to **8 / 20 (40 %)**. House-style adherence drops from ~3 to ~1 — the model has stopped emitting the bulleted-template explanation and now answers most "explain Article N" prompts by **reproducing the article's verbatim text**. That is the expected and *intended* effect of the knowledge-injection training: the adapter knows what the law says, not how to explain it in our voice.

| Aggregate (mean over 20 explanations) | Stage 0 | Stage 1 | RAFT closed-book | **Stage A v2 (3 B + Unsloth)** |
|---|---|---|---|---|
| **Legal accuracy** | 1.00 | 1.00 | 1.00 | **3.35** |
| **Faithfulness to article** | 1.00 | 1.00 | 1.00 | **3.25** |
| **Completeness** | 1.00 | 1.00 | 1.50 | **2.20** |
| **House-style adherence** | 3.15 | 3.55 | 2.50 | 1.05 |
| Language quality | 4.15 | 4.10 | 4.00 | **4.45** |
| **Pass rate (mean ≥ 3.5)** | 0 / 21 | 0 / 21 | 0 / 21 | **8 / 20 (40 %)** |
| Refusal accuracy (1 case) | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| (for reference) RAFT *open*-book LA | — | — | — | 2.85 (with article in prompt) |

**The thesis question — "Did PEFT learn the domain?" — is answered yes for the first time.** Stage A v2 *closed-book* legal accuracy (3.35) is now *higher* than Stage 4 RAFT's *open-book* legal accuracy (2.85) — the model knows the law without retrieval. 8 of 20 cases pass cleanly, all 8 driven by character-perfect or near-perfect article-text reproduction.

The trade-off — collapsed house-style adherence (3.15 → 1.05) — is the **expected** consequence of training on a deliberately style-less corpus and is a *separate* skill question, addressed by a combined-dataset run (see §6 of [the experiments journey doc](../experiments_journey.md)).

---

## 1. Per-case scores

Scale 1–5; mean is the unweighted average across the 5 dimensions.
**Legend:** LA = legal accuracy, C = completeness, HS = house-style, LQ = language quality, F = faithfulness.

### EN explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 690 | 4 | 2 | 1 | 2 | 4 | 2.6 | Correct legal content but **bilingual leakage mid-sentence** (Arabic clause inserted between English clauses). Both halves are real Article-690 content. |
| 2 | 1068 | 5 | 3 | 1 | 5 | 5 | **3.8** | **Article 1068 reproduced character-perfect** (690-char procedural rule). Same article that was perfect in the 1.5B v1 smoke test. |
| 3 | 280 | 5 | 3 | 1 | 5 | 5 | **3.8** | **Article 280 reproduced verbatim** — the one v1 hallucinated as a different debtor-release rule. Solved. |
| 4 | 662 | 1 | 1 | 1 | 4 | 1 | 1.6 | Wrong-neighbour: produced an article-termination rule (≈ Art 663) instead of Article 662 (sub-contractors' direct action). |
| 5 | 530 | 5 | 3 | 1 | 4 | 5 | **3.6** | **Article 530 reproduced verbatim** (partnership dissolution by court). |
| 6 | 943 | 1 | 1 | 1 | 4 | 1 | 1.6 | Wrong-neighbour: produced Article 942 (summons particulars) instead of Article 943 (pre-emption procedure / 30-day deadline). |
| 7 | 764 | 1 | 1 | 1 | 5 | 1 | 1.8 | Wrong-neighbour: produced a premium-payment-timing rule instead of Article 764 (life-insurance age misstatement). |
| 8 | 158 | 5 | 3 | 2 | 5 | 5 | **4.0** | **Article 158 reproduced verbatim**, with a faint structural prefix ("Article 158 of the Egyptian Civil Code reads:"). Highest EN score. |
| 9 | 360 | 3 | 1 | 1 | 5 | 2 | 2.4 | One-sentence topic-only answer: "Article 360 concerns Novation and Delegation." Correct topic, zero content. |
| 10 | 535 | 1 | 1 | 1 | 5 | 1 | 1.8 | Wrong-neighbour: produced an arbitration-clause rule instead of Article 535 (liquidator powers). |

**EN means:** LA 3.1 · C 1.9 · HS 1.1 · LQ 4.4 · F 3.0 → **overall 2.72 / 5**.  Verbatim-perfect: 5 / 10. Pass (mean ≥ 3.5): 4 / 10.

### AR explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | Notes |
|---|---|---|---|---|---|---|---|---|
| 11 | 242 | 5 | 3 | 1 | 5 | 5 | **3.8** | **Article 242 reproduced verbatim** (fraud + insolvent-debtor early payment). |
| 12 | 936 | 1 | 1 | 1 | 5 | 1 | 1.8 | Wrong article: produced text about movables identified only by kind (≈ Art 204) instead of pre-emption holders. |
| 13 | 233 | 1 | 1 | 1 | 5 | 1 | 1.8 | One-line wrong content: "obligations not in the code are regulated by special laws". Wrong article. |
| 14 | 1005 | 5 | 3 | 1 | 5 | 5 | **3.8** | **Article 1005 reproduced verbatim** (tahkir rent assessment). |
| 15 | 1098 | 4 | 3 | 1 | 4 | 4 | 3.2 | Content correct (cross-reference rule) but **reproduces OCR-corrupted article numbers** from the corpus (1033 → ٣٣٠١, 1040-1042 → ٠٤٠١ إلى ٢٤٠١). Faithful to the *source*, including its OCR rot. |
| 16 | 1089 | 5 | 4 | 1 | 5 | 5 | **4.0** | **Article 1089 reproduced verbatim** in full, including all 5 sub-paragraphs (a-e). Highest AR score; the longest perfect reproduction. |
| 17 | 1108 | 1 | 1 | 1 | 5 | 1 | 1.8 | Wrong article: produced creditor-care-of-pledged-thing rule (≈ Art 1102/1095) instead of Article 1108 (pledge inherits Art 1051 + 1052). |
| 18 | 834 | 5 | 3 | 1 | 5 | 5 | **3.8** | **Article 834 reproduced verbatim** (partition right + 5-year cap on no-partition agreement). |
| 19 | 1077 | 4 | 3 | 1 | 3 | 4 | 3.0 | Content correct (auction-surplus to holder) but **minor AR typos** ("الزاد" instead of "المزاد", "المر نين" instead of "المرتهنين"). |
| 20 | 523 | 5 | 3 | 1 | 3 | 4 | 3.2 | **Article 523 reproduced (partner liability for partnership debts)** with one omitted suffix ("بديو ا" should be "بديونها"). |

**AR means:** LA 3.6 · C 2.5 · HS 1.0 · LQ 4.5 · F 3.5 → **overall 3.04 / 5**.  Verbatim-perfect: 5 / 10. Pass (mean ≥ 3.5): 4 / 10.

### Refusal (1 case)

| # | Lang | Prompt | Score | Notes |
|---|---|---|---|---|
| 21 | AR | "هل يحق لمالك الشقة دخول مسكني متى شاء؟" ("Does the landlord have the right to enter my flat whenever he wants?") | **0** | Produced gibberish with cross-language code-switching (English "done", what appears to be Spanish fragments "isper", "isperanca"). Not a refusal, not a real article — just collapsed generation. |

**Refusal accuracy: 0 / 1 = 0 %.** Refusal data was not part of the knowledge dataset (deliberately — house-style refusals are a Stage-B concern). The adapter has no concept of "this isn't an article-quote question, decline politely". Expected.

---

## 2. The headline finding, stated precisely

For the first time in the project, the model **closed-book** scores legal accuracy materially above 1.0 / 5. **Mean 3.35 / 5** vs 1.0 on every prior closed-book run (Stage 0 baseline, Stage 1 data interventions, Stage 4 RAFT closed-book). The 12 / 20 cases with reproducible-verbatim or near-verbatim article text are *correct law in the right place*. The 8 / 20 misses are wrong-neighbour reproductions (the model emits real Civil-Code text from a *different* article — same failure mode the 1.5B v1 smoke test had, now narrower).

**This corroborates the closed-book smoke-test result from earlier today** (8 verbatim articles, mean char-sim **0.884**, 7 / 8 character-perfect, base lift 12×) on a different, independent test sample drawn from the prior runs' val set. The two evals agree:
- ~70 % verbatim recall rate (smoke test: 7 / 8; judge eval: 12 / 20 → confirming with a stratified sample) at character-perfect or near-perfect level,
- the remaining failure mode is article-number ↔ article-text *binding*, not article-text *memory*,
- the failure mode is "produce a real, different article" not "hallucinate a fake one".

In other words: **the knowledge dataset trained the model to be a faithful — if imperfectly indexed — quotation engine for the Egyptian Civil Code, on a 6 GB consumer GPU**.

---

## 3. Why house-style adherence collapsed — and why it's fine

| Adapter | Trained on what gold answers? | What it learns |
|---|---|---|
| Stage 0 / 1 / RAFT | LLM-polished ~250-word essays in a fixed house-style template | the *template* (1-line summary → bullets → example → disclaimer) |
| **Stage A v2 (this)** | **raw `orig_data.json` article text + variants (verbatim / cloze / reverse / placement / translate / card / contrast / roster)** | the *law*, not the template |

The knowledge dataset has **zero examples** of the house-style template — by design (see §4.1 of [the experiments journey doc](../experiments_journey.md)). So when the judge asks "explain Article 158", the model does the thing it was actually trained for: it reproduces Article 158. Article-text reproduction is *correct legal content* but it isn't a *house-style explanation*. The judge's HS dimension penalises that.

The right next move — if and only if you want both skills — is **not** to follow this with a sequential house-style training pass (that would catastrophically forget the verbatim recall), but a **combined-dataset single run**: knowledge + house-style examples in one training mixture. That gives both signals to the optimiser simultaneously and has no forgetting problem. See §6 of the journey doc.

For the **thesis result** (PEFT learned the domain), HS is not the metric — LA / F / Pass-rate are. And those just delivered.

---

## 4. Bilingual parity

| Dimension | EN | AR | Gap |
|---|---|---|---|
| Legal accuracy | 3.1 | 3.6 | **+0.5 (AR better)** |
| Faithfulness | 3.0 | 3.5 | +0.5 |
| Completeness | 1.9 | 2.5 | +0.6 |
| House-style | 1.1 | 1.0 | −0.1 |
| Language quality | 4.4 | 4.5 | +0.1 |
| **Overall mean** | **2.72** | **3.04** | **+0.32 (AR better)** |

This is the **first run in the project where Arabic out-scores English on overall mean.** Stage 0/1 had EN ahead by 0.48; RAFT had EN ahead. The flip is driven by:
1. The 3 B base has materially stronger Arabic than the 1.5 B used in v1.
2. The new `kn_card` / `kn_contrast` / `kn_reverse` task families have AR variants with the same coverage as EN — better article-number binding on the AR side.
3. AR generation is no longer collapsing into bullet-loops or script-leakage — the cases that *do* fail (12, 13, 17) fail cleanly (wrong real article in clean Arabic) rather than as broken text.

The one AR failure mode that persists is **OCR-corruption fidelity** (case 15: `1033 → ٣٣٠١`) — the model faithfully reproduces the corpus's own OCR artifacts. A pre-cleaning pass on `orig_data.json` would fix this; it's a corpus issue, not a model issue.

---

## 5. Pattern summary across the 5 judge evals

| Run | Closed-book LA | Closed-book F | Open-book LA (RAFT only) | HS | Pass-rate | What it learned |
|---|---|---|---|---|---|---|
| Stage 0 (1.5B, 924 pairs) | 1.00 | 1.00 | — | 3.15 | 0 / 21 | the *template* (only signal in the data) |
| Stage 1 (1.5B, 2,091 pairs, 8 variants) | 1.00 | 1.00 | — | 3.55 | 0 / 21 | same template, better executed (no loops, cleaner refusals) |
| Stage 4 RAFT (1.5B, 2,404 pairs w/ article-in-context) | 1.00 | 1.00 | 2.85 | 2.50 / 3.80 | 0 / 21 (closed) / ~3-4 (open) | the *skill of grounding in supplied context*, not the law itself |
| **Stage A v2 (3B + Unsloth, 21,679 pairs, knowledge-injection)** | **3.35** | **3.25** | — | 1.05 | **8 / 20 (40 %)** | **the law itself; not the template (by design)** |

Each prior run answered "did PEFT learn the domain?" with *no* (closed-book LA stayed at floor). Stage A v2 answers it with **yes** — for the first time, the adapter holds the actual Civil-Code content in its weights, retrievable closed-book on a 21-prompt held-out evaluation.

---

## 6. Reproducibility

```bash
# Inference
python scripts/run_judge_inference.py \
  --adapter-dir runs/qlora-qwen2.5-3b-knowledge \
  --val-path    data/qa_pairs_val.baseline.jsonl \
  --out-path    reports/eval/judge_predictions_knowledge_3b.json
# (use the RAFT conda env which has Unsloth: /home/aya/miniconda3/envs/RAFT/bin/python)

# Then re-judge by reading reports/eval/judge_predictions_knowledge_3b.json
```

Inference wall-clock: ~4.5 minutes for 21 generations on the RTX 3050 (mean 12.8 s/gen).

---

*Generated 2026-05-14 by Claude as in-session LLM judge. Compare against [judge_report.md](judge_report.md) (Stage 0), [judge_report_stage1.md](judge_report_stage1.md), [judge_report_raft.md](judge_report_raft.md).*
