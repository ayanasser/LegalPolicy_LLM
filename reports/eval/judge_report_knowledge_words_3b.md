# LLM-as-Judge Evaluation — Stage A v3 (knowledge-injection, **words-form** numbers + language-tagged, Qwen 2.5 3B + Unsloth, r=32 / α=64)

**Adapter under test:** [`runs/qlora-qwen2.5-3b-knowledge-words/`](../../runs/qlora-qwen2.5-3b-knowledge-words/) — trained 2026-05-31 → 2026-06-01, 4 epochs, ~17.1 h on RTX 3050, **r=32 / α=64 + MLP modules**, on `data/qa_pairs_knowledge_words.jsonl` (the 22 294-row words-form variant of Stage A v2's dataset, with `[AR]/[EN]/[BI]` prompt tags applied via `scripts/prefix_language_tag.py`).
**Comparison points (apples-to-apples):** Stage 0 baseline (`qlora-qwen2.5-1.5b-v1`), Stage 1 (`qlora-qwen2.5-1.5b-stage1`), Stage 4 RAFT (`qlora-qwen2.5-1.5b-raft`), Stage A v2 (`qlora-qwen2.5-3b-knowledge`).
**Judge:** Claude (Opus 4.7), in-session.
**Sample:** **21 stratified validation examples** — 10 EN explanation + 10 AR explanation + 1 AR refusal, drawn from `data/qa_pairs_val.baseline.jsonl` (the same fixed set every prior judge report used).
**Inference mode:** **closed-book** (no article supplied in the prompt). `--language-tag auto` was set in `scripts/run_judge_inference.py`, so each prompt was prefixed with `[AR]` / `[EN]` to match the adapter's training distribution.
**Inputs:** [`judge_predictions_knowledge_words_3b.json`](judge_predictions_knowledge_words_3b.json).

---

## TL;DR

**This iteration is a regression vs Stage A v2 on the closed-book judge: legal accuracy collapses from 3.35 / 5 back to ~1.0, and pass rate from 8 / 20 (40 %) to 0 / 20.** The two positives are real but narrow: the refusal case is answered perfectly (verbatim match with the gold disclaimer — refusal training is the one thing the bigger adapter + tags fully internalised), and language quality nudges up (4.45 → 4.65) — the `[AR]/[EN]` prompt tags do reduce the kind of bilingual leakage Stage A v2 sometimes showed (Stage A v2 had a mid-EN-answer Arabic clause in case 1; v3 has fewer such artefacts but it still surfaces, e.g. case 3).

The dominant failure mode under v3 is **"wrong article, confidently"**: the model overwhelmingly produces a *real, fluent* paragraph of Egyptian Civil Code text — but bound to a different article from the one asked. The clearest tell is case 9 (asked for Article 360) returning the verbatim text of Article 280, while case 3 (asked for Article 280) returned text from a different article entirely — the *texts* are still memorised, but the *number ↔ text binding* is no longer reliable. In Stage A v2, character-perfect reproductions drove the 3.35 LA score; here those reproductions are misrouted.

| Aggregate (mean over 20 explanations) | Stage 0 | Stage 1 | RAFT closed-book | **Stage A v2 (3B knowledge)** | **Stage A v3 (this run: knowledge_words, r=32, [AR]/[EN])** |
|---|---|---|---|---|---|
| **Legal accuracy** | 1.00 | 1.00 | 1.00 | **3.35** | **1.05** ⬇ |
| **Faithfulness to article** | 1.00 | 1.00 | 1.00 | **3.25** | **1.00** ⬇ |
| **Completeness** | 1.00 | 1.00 | 1.50 | **2.20** | **1.10** ⬇ |
| House-style adherence | 3.15 | 3.55 | 2.50 | 1.05 | **1.05** = |
| **Language quality** | 4.15 | 4.10 | 4.00 | 4.45 | **4.65** ⬆ |
| **Pass rate (mean ≥ 3.5)** | 0 / 21 | 0 / 21 | 0 / 21 | **8 / 20 (40 %)** | **0 / 20 (0 %)** ⬇ |
| **Refusal accuracy** | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 | **1 / 1** ⬆ |

**Why** — the two changes intended to *help* number-binding actually broke it. Spelling article numbers as words (`"Article seven hundred and seventy five"`) makes the number a multi-token phrase that the BPE tokeniser shares with thousands of other contexts — so the *number string* and the *article text* are no longer rare-token-anchored to each other. Going r=16 → r=32 + 4 epochs of these blurry IDs gives the model more capacity to memorise *texts* (you can see how *real* the wrong outputs are) but encourages it to use that capacity for content rather than number-to-content addressing. The article number stops being a sharp index and starts behaving like a hint.

This does **not** mean the work was wasted — the run produced clean refusals and cleaner language than v2 — but the next iteration should revert the words-form transformation and rely on the `[AR]/[EN]` tagging alone (plus possibly a higher per-article exposure on the digit-form numbers).

---

## 1. Per-case scores

Scale 1–5; mean is the unweighted average across the 5 dimensions.
**Legend:** LA = legal accuracy · C = completeness · HS = house-style · LQ = language quality · F = faithfulness to source.

### EN explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 690 | 1 | 1 | 1 | 4 | 1 | **1.6** | Returned a *contract-for-work dissolution by death* rule (Art. 666 territory) instead of Art. 690 (master's duty to pay salary). Includes a number bleed: `"Article six98"` — the words-form transformation showing through. |
| 2 | 1068 | 1 | 1 | 1 | 4 | 1 | **1.6** | Returned a **`### Subject / ### Key point / ### Text` card** describing **The Right of Hekr** (Art. 1005 territory) instead of mortgage-purge procedure. Card format itself is real (a training-time `kn_card` row), but bound to the wrong article. |
| 3 | 280 | 1 | 1 | 1 | 2 | 1 | **1.2** | Returned a *payment-imputation* rule (expenses → interest → principal, ≈ Art. 343). Worse: a mid-sentence Arabic clause is spliced in — `"...to pay expenses من الفوائد ثم من أصل الدين ، كل هذا ما لم يتفق على غيره. and interest in addition..."` — exactly the bilingual-leakage failure mode v2 also showed. |
| 4 | 662 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned an article on works destroyed before delivery (Art. 666 territory) instead of sub-contractors' direct action against the master. Coherent English, fluent — but wrong article. |
| 5 | 530 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned the exchange-as-sale rule (Art. 482). Wrong article; coherent text. |
| 6 | 943 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a 15-year acquisitive-prescription rule (Art. 968 territory) instead of pre-emption procedure. |
| 7 | 764 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a suretyship-discharge rule (Art. 783 territory) instead of life-insurance age misstatement. |
| 8 | 158 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a fraud-in-contract rule (Art. 125 territory) instead of automatic-rescission clauses. |
| 9 | 360 | 1 | 1 | 1 | 5 | 1 | **1.8** | **Returned the verbatim text of Article 280** ("solidarity between creditors, debtor may pay any one...") — which is itself the gold for case 3. Texts memorised; the number→text map is *swapped* with case 3. |
| 10 | 535 | 1 | 2 | 1 | 5 | 1 | **2.0** | Returned a partner-exclusion rule (Art. 528 territory) but at least named the article in words-form and tagged its topic ("falls under Ways in which a partnership comes to an end") — slight scaffold credit. Still the wrong article. |
| **EN means** |   | **1.00** | **1.10** | **1.00** | **4.50** | **1.00** | **1.72** |   |

### AR explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | Notes |
|---|---|---|---|---|---|---|---|---|
| 11 | 242 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned the *astreinte / penalty for refusal to perform a personal act* rule (Art. 213) instead of the Paulian-action / preferential-payment rule. Clean Arabic, no bleed. |
| 12 | 936 | 2 | 2 | 1 | 5 | 1 | **2.2** | **Topically adjacent miss** — returned text on competing pre-emptors (Art. 937 territory) instead of *who* has the right of pre-emption (Art. 936). Best AR case: same chapter, wrong article. |
| 13 | 233 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned Paulian-action territory (Art. 238 ff.) instead of commercial-interest rules on current accounts. |
| 14 | 1005 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a mortgage-purge rule about transfer of title to the holder if no sale is requested in time (Art. 1067 territory) instead of hekr rent-revaluation criteria. |
| 15 | 1098 | 1 | 1 | 1 | 5 | 1 | **1.8** | **Same prediction as case 14** — the model repeats the Art. 1067-style text for two different Art. 10xx prompts. Strong evidence that within a chapter the number↔text addressing collapses. |
| 16 | 1089 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned the third-party-holder defences rule (Art. 1095 territory) instead of the specialty-mortgage filing procedure. |
| 17 | 1108 | 1 | 1 | 2 | 3 | 1 | **1.6** | A `### Subject / ### النص` **card** but with internal inconsistency: subject reads `"تنازع القوانين من حيث الزمان"` (conflict-of-laws-in-time, Art. 6–8) while the body is hekr rent text (Art. 1005). The card scaffold itself earns a point. |
| 18 | 834 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a *condominium common-areas* rule (Art. 856 territory) instead of the right to demand partition. |
| 19 | 1077 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a *servitudes* rule about not impairing the use (Art. 1019 territory) instead of mortgage-auction surplus. |
| 20 | 523 | 1 | 1 | 1 | 5 | 1 | **1.8** | Returned a *landlord's repair obligations / tenant self-help* rule (Art. 575 territory) instead of partner liability for partnership debts. |
| **AR means** |   | **1.10** | **1.10** | **1.10** | **4.80** | **1.00** | **1.82** |   |

### Refusal (1 case)

| # | Prompt | Refusal correctness | Notes |
|---|---|:---:|---|
| 21 | `هل يحق لمالك الشقة دخول مسكني متى شاء؟` | **1 / 1** | **Verbatim match with the gold disclaimer** (down to the `تنبيه:` line). First adapter in the project to score 1 / 1 on the refusal probe. |

---

## 2. Aggregates

| Dimension | EN (10) | AR (10) | Combined (20) |
|---|---|---|---|
| Legal accuracy | 1.00 | 1.10 | **1.05** |
| Completeness | 1.10 | 1.10 | **1.10** |
| House-style adherence | 1.00 | 1.10 | **1.05** |
| Language quality | 4.50 | 4.80 | **4.65** |
| Faithfulness to source | 1.00 | 1.00 | **1.00** |
| **Mean (5 dims)** | 1.72 | 1.82 | **1.77** |
| **Pass rate (mean ≥ 3.5)** | 0 / 10 | 0 / 10 | **0 / 20 (0 %)** |
| **Refusal accuracy** | — | — | **1 / 1 (perfect)** |

**Bilingual parity** is genuine: EN and AR are within 0.10 of each other on every dimension. The model is equally wrong in both languages — that is not nothing for a 3B model on Egyptian-Civil-Code Arabic, but it doesn't help LA.

---

## 3. Failure-mode breakdown (20 explanations)

| Failure mode | Count | Example case |
|---|---|---|
| Wrong article — text is real but from a different article | 18 / 20 | every case except the two with slight scaffold |
| Same wrong prediction repeated for two different prompts | 1 pair | cases 14 & 15 (both Art. 10xx) |
| Number-text *swap* with another case in the same sample | 1 pair | case 9 returns case 3's gold text |
| Bilingual / mid-sentence script bleed | 1 / 20 | case 3 (Arabic clause in an English answer) |
| Number bleed (`"six98"` from words-form) | 1 / 20 | case 1 |
| `### Subject / ### Text` **kn_card** scaffold leaking into a plain "explain" prompt | 2 / 20 | cases 2 (EN), 17 (AR) |
| Refused unnecessarily | 0 / 20 | — |
| Empty / degenerate output | 0 / 20 | — |

The complete absence of "refused unnecessarily" is worth flagging: the earlier `[AR]`-tagged CSV eval saw ~87 % unnecessary refusals on **colloquial conversational** prompts (`"أنا استعملت حقي بشكل عادي ..."`), but on the **canonical val baseline** prompts (`"اشرح المادة 233 ..."`) the model never refuses. So the over-strong refusal attractor is OOD-triggered, not in-distribution.

---

## 4. Honest interpretation

This iteration's design was: (i) tag prompts with `[AR]/[EN]/[BI]` to lock the output language, (ii) spell article numbers as words to give the BPE tokeniser something semantic to anchor on, (iii) bump LoRA capacity to r=32 / α=64. The hypothesis was that (i)+(ii) would close the bilingual-leakage and article-number-binding gaps Stage A v2 showed, and (iii) would give room for the corpus.

What actually happened:
- **(i) `[AR]/[EN]` tags worked** — language quality is the highest in the project (4.65) and the egregious bilingual-leakage of Stage A v2 is mostly gone (one case left, vs several in v2). This is a real win and worth keeping.
- **(ii) words-form numbers backfired** — by exploding `"Article 775"` (a single rare token) into `"Article seven hundred and seventy five"` (a long phrase whose pieces appear in countless other contexts), we diluted the number's role as a sharp index. The model still memorised *texts* (you can read them in cases 1–20: real Civil-Code paragraphs throughout) but lost the discriminative number → text addressing that drove Stage A v2's 3.35 LA score. The smoking gun is cases 3 and 9 swapping each other's gold texts.
- **(iii) r=32 + 4 epochs** of (ii) reinforced this: more capacity went into encoding more passages, not into sharpening the (now-blurry) number index.
- **Refusal training overshot in scope** (the CSV result) but is **perfectly calibrated in distribution** (this report's refusal case is verbatim correct).

**Recommendation for the next training run**:

1. **Keep `[AR]/[EN]/[BI]` tags.** They paid off and didn't cost anything.
2. **Drop the words-form numbers — revert to digit-form `qa_pairs_knowledge.jsonl`.** This is the change to undo; the number → text binding needs digits as a sharp anchor.
3. **Keep r=32 / α=64 + MLP modules.** Capacity itself isn't the problem here.
4. **Up-weight `kn_reverse` and `kn_card` rows** (number → text and number ↔ topic), and/or down-weight `kn_verbatim` (text-only memorisation). The current ratio over-rewards "produce real Code text" relative to "produce the *right* Code text for *this* number."
5. **Cap epochs at 3** unless eval-loss is still falling — Stage A v2 (4 epochs) and v3 (4 epochs) both show the eval curve flat after epoch 3, and the extra epoch may be where the number index degraded under v3's blurry-ID setup.

Final note: Stage A v2's `qlora-qwen2.5-3b-knowledge` adapter remains the project's strongest closed-book artefact. This v3 run is an honest negative result — useful for ruling out the words-form hypothesis — and a confirmation that the `[AR]/[EN]` tagging and refusal-handling work as intended.

---

## 5. Run metadata

- Wall time (training): **~17 h 4 min**, 5 576 optimiser steps (4 epochs · 22 294 rows / grad-accum 16) on RTX 3050.
- Trainable params: 59 867 136 / 3 145 805 824 (1.90 %).
- Final eval-loss trajectory: epoch 1 0.2815 → epoch 2 0.1709 → epoch 3 0.1543 → **epoch 4 0.1530** (best checkpoint loaded).
- Tracking: **W&B** `wandb.ai/aya-nasser-mohammed-valeo/legalpolicy-knowledge/runs/0vger7zt` + **TensorBoard** under `runs/qlora-qwen2.5-3b-knowledge-words/runs/`.
- Predictions (this report): [`judge_predictions_knowledge_words_3b.json`](judge_predictions_knowledge_words_3b.json), 21 cases, avg gen time 19.4 s / case.
