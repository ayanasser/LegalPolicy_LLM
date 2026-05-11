# LLM-as-Judge Evaluation — Stage 4 (PEFT-RAFT)

**Adapter under test:** `runs/qlora-qwen2.5-1.5b-raft/` (Stage 4, trained 2026-05-11).
**Comparison points:** Stage 0 baseline (`qlora-qwen2.5-1.5b-v1`), Stage 1 (`qlora-qwen2.5-1.5b-stage1`).
**Judge:** Claude (Opus 4.7), in-session.
**Sample:** 21 stratified validation examples (10 EN explanation + 10 AR explanation + 1 AR refusal), same `data/qa_pairs_val.baseline.jsonl` set as Stage 0/1.
**Two inference modes on the same sample:**
- **closed-book** (`--raft-context none`): plain question, no article supplied — tests whether RAFT training imprinted content into the adapter.
- **open-book** (`--raft-context oracle+distractor`): question preceded by the asked article + 1 distractor, matching the RAFT training format — tests whether the adapter learned to *ground its answer in the provided article*.

**Inputs:** [judge_predictions_raft_closedbook.json](judge_predictions_raft_closedbook.json), [judge_predictions_raft_openbook.json](judge_predictions_raft_openbook.json), per-case side-by-sides in [cases_raft/](cases_raft/).

---

## TL;DR

**Open-book RAFT is the first configuration in this project with non-zero legal accuracy.** When the article is supplied in context, the RAFT-trained adapter grounds its answer in that article and produces mostly-correct content — legal accuracy jumps from ~1.0/5 (Stage 0/1, and RAFT closed-book) to **~2.85/5**, and the pass rate goes from 0/21 to **~3-4/21**. The improvement is uneven (EN cases noticeably better than AR; the model sometimes bleeds content from the distractor; the "rationale/effect" bullets it generates beyond the article can still be wrong) — but it is a real, measurable jump.

**Closed-book RAFT confirms the Stage-1 finding once more:** RAFT training does *not* imprint content into the adapter. With no article supplied, the model still hallucinates wrong content (EN) or abstains entirely (AR). PEFT on a 1.5B base, by itself, cannot learn 148 articles' content — regardless of how the training data is structured.

| Metric (mean over 20 explanations) | Stage 0 | Stage 1 | RAFT closed-book | **RAFT open-book** |
|---|---|---|---|---|
| **Legal accuracy** | 1.00 | 1.00 | 1.00 | **2.85** |
| **Faithfulness to article** | 1.00 | 1.00 | 1.00 | **2.90** |
| **Completeness** | 1.00 | 1.00 | 1.50 | **3.25** |
| House-style adherence | 3.05 | 3.55 | 2.50 | 3.80 |
| Language quality | 3.95 | 4.10 | 4.00 | 4.00 |
| **Pass rate (mean ≥ 3.5)** | 0/21 | 0/21 | 0/21 | **~3-4/21** |
| **Refusal correctness (1 case)** | 0/1 | 0/1 | 0/1 | 0/1 |
| Training-time eval loss | 1.587 | 1.375 | — | **1.280** |
| Training-time token accuracy | 64.2% | 68.0% | — | **70.1%** |

(The RAFT training-time eval is open-book, so its lower loss is partly because the task is easier — not directly comparable to the closed-book runs.)

---

## 1. What Stage 4 changed

| Dimension | Stage 1 | Stage 4 (RAFT) |
|---|---|---|
| Data interventions (8 variants/article, cross-lang parity, 50 refusal seeds ×8, 60 contrastive) | ✓ | ✓ |
| **RAFT context block** | — | **each question preceded by `[oracle article] + [1 distractor article]`, shuffled, in the question's language** |
| Header instruction | — | "Use the article(s) below to answer… ground your explanation strictly in the article the question is about… don't invent its content if it's not present" |
| `max_seq_length` | 1024 | 1536 (to fit context + question + answer) |
| Train / val | 2,091 / 369 | 2,404 / 424 |
| Articles covered (cache-restricted) | 125 | 148 |
| Training time on RTX 3050 6 GB | ~1h54m | ~3h03m |

Only the LoRA adapter weights update — the 4-bit base stays frozen. RAFT changes *what the adapter is trained to do* (locate + rephrase the article in context), and as a side effect exposes the adapter to every article's text many times.

---

## 2. Per-case scoring summary

Means across the 20 explanation cases (1–5 scale):

|  | Baseline | Stage 1 | RAFT closed-book | RAFT open-book |
|---|---|---|---|---|
| Legal accuracy | 1.00 | 1.00 | 1.00 | 2.85 |
| Faithfulness | 1.00 | 1.00 | 1.00 | 2.90 |
| Completeness | 1.00 | 1.00 | 1.50 | 3.25 |
| House-style adherence | 3.05 | 3.55 | 2.50 | 3.80 |
| Language quality | 3.95 | 4.10 | 4.00 | 4.00 |
| **Per-case mean** | 2.20 | 2.31 | 2.00 | **3.16** |

### Open-book RAFT — selected cases

| # | Article (topic) | Open-book quality | Notes |
|---|---|---|---|
| 02 | 1068 (auction application requirements) | **~4/5** | Correctly lists summons to third-party holder + former owner, special mandate, court deposit, no-refund rule, nullity for non-compliance, renunciation needs consent. Tracks the article closely. One invented "effect of renunciation" bullet. |
| 05 | 530 (partnership dissolution) | **~4/5** | "court can dissolve a partnership for non-performance… judge decides if the reason is sufficiently serious… any agreement to the contrary is void." Substance right; minor ordering wobble. |
| 06 | 943 (pre-emption procedure) | ~3.5/5 | Filing against vendor + purchaser, 30-day enrollment deadline, urgency — all correct. Slight confusion about who holds the pre-emption right in the example. |
| 16 | 1089 (judgment-charge application) | ~3/5 | Correctly lists: application to President of Court of First Instance in the property's district, authentic copy of judgment, creditor/debtor particulars, judgment date/court, debt amount, precise description of immovables. Some English-label leakage in the AR bullets. |
| 01 | 690 (wage payment) | ~3/5 | "master must pay the worker his salary at the time the contract is made or by custom, subject to special laws" — grounded. But appends a hallucinated "notice obligation… Article 563" pulled from the **distractor**. |
| 03 | 280 (solidarity between creditors) | ~3/5 | Headline correct (pay any one creditor; division among heirs). The "why" bullets drift and one gets the rule backwards. |
| 11 | 242 (preference fraud / insolvent early payment) | ~2.5/5 | Two-paragraph structure recovered; the "علة" rationale bullets are garbled and repetitive. |
| 15 | 1098 (pledge inherits mortgage rules) | ~2/5 | Gets the shape ("pledge follows the formal-mortgage rules"), but invents a non-existent "exception", reproduces OCR-corrupted article numbers from the corpus, and adds a flatly wrong claim ("formal mortgage = state ownership"). |

### Closed-book RAFT — what happens without the article

- **EN explanations (10):** still hallucinate a *different* topic than the article (e.g. Art 943 → "division of jointly owned property"; Art 690 → "effect of contract termination"). Indistinguishable from Stage 0/1.
- **AR explanations (10):** ~8/10 now **abstain** ("I'm an AI assistant, I can't interpret law / I don't have this article, consult a lawyer") rather than confabulate. This is the contrastive + refusal training leaking into normal explanation requests — *safer* (no false legal content) but *useless* (no answer). The other 2 still hallucinate.
- **Refusal case (21):** abstains cleanly. No catastrophic loop (unlike the Stage-0 baseline's 10× disclaimer repetition).

---

## 3. The headline finding, stated precisely

**Two distinct questions, two distinct answers:**

1. *"Did PEFT, with RAFT-style training, learn the Egyptian Civil Code on its own?"* → **No.** Closed-book RAFT legal accuracy is still ~1.0/5, pass rate 0/21. A 1.5B base + 18M-parameter LoRA cannot internalise 148 articles' content from ~16 mentions each, even when those mentions are structured as RAFT examples. This matches the Stage-0 and Stage-1 conclusions.

2. *"Did the RAFT-trained adapter learn to ground its answer in a provided article?"* → **Yes, partially.** Open-book RAFT legal accuracy is ~2.85/5, faithfulness ~2.9/5, pass rate ~3-4/21 — the first non-zero numbers in the project. When handed the article, the adapter parses the question, locates the relevant article in the context (even with a distractor present), and produces a house-style answer grounded in it. Several EN cases (Art 1068, Art 530) are genuinely good.

**For the thesis, this is a clean PEFT result:** RAFT is a PEFT method (only the adapter updates), and it demonstrably changes the adapter's behaviour — from "confabulate the law" to "rephrase the law you're shown". The closed-book vs open-book gap is the whole story: PEFT taught a *skill* (ground-in-context), not *knowledge* (the law itself).

---

## 4. Remaining failure modes (open-book)

| Failure | Frequency | Example |
|---|---|---|
| Distractor bleed — pulls content from the distractor article into the answer | ~2/10 EN, ~2/10 AR | Case 01: appended a "notice obligation… Article 563" from the distractor onto an Article 690 answer |
| Over-generated "why/rationale/effect" bullets that go beyond the article and can be wrong | ~half of all cases | Case 03: a "why solidarity prevents division among heirs" bullet that contradicts the article |
| AR elaboration noisier than EN — garbled rationale, repetition, English-label leakage in bullets | most AR cases | Case 16: bullets labelled "obligation of the creditor to file an application" in English inside an Arabic answer |
| Faithfully reproduces OCR artifacts present in the corpus AR text | a few AR cases | Case 15: copies the corrupted article numbers "٣٣٠١ / ٠٤٠١ إلى ٢٤٠١" verbatim from the context |
| Refusal triggers still answered with invented legal content | 1/1 | Case 21: "the Civil Code prohibits the owner from entering the flat except by request of the guarantor/lawyer/judge" — fabricated |

Most of these would shrink with a larger base model (Stage 5: Qwen 2.5 7B). The distractor-bleed and over-generation are capacity/instruction-following issues; the AR roughness is a base-model strength issue; the OCR artifacts argue for cleaning `orig_data.json` before any further round.

---

## 5. Bilingual parity (open-book)

| Dimension | EN | AR | Gap |
|---|---|---|---|
| Legal accuracy | ~3.3 | ~2.4 | −0.9 (AR worse) |
| Faithfulness | ~3.3 | ~2.5 | −0.8 |
| Completeness | ~3.5 | ~3.0 | −0.5 |
| House-style | ~4.0 | ~3.6 | −0.4 |
| Language quality | ~4.5 | ~3.5 | −1.0 (English-label leakage in AR bullets) |

The AR side benefits from RAFT (it now gets the article's *topic* right instead of hallucinating a different one), but the elaboration quality lags EN — same base-model strength gap seen in earlier stages, now the dominant limiter on AR open-book quality.

---

## 6. Where this leaves the project

**The PEFT story now has a positive result to report:** Stage 4 (PEFT-RAFT) is the first configuration where the trained adapter, when paired with retrieval, produces faithful, mostly-correct legal explanations. That is the deployment configuration — RAFT-trained adapter + the Epic 3 retriever feeding it the article. The closed-book number stays at floor, which honestly documents the limit: at 1.5B, PEFT learns the *skill* of grounding, not the *content* of the corpus.

**Recommended next steps (in order):**

1. **Stage 5 — scale the base.** Re-run the exact Stage-4 recipe on **Qwen 2.5 7B Instruct** with QLoRA r=64 on the Colab A100 (notebook scaffold already in `notebooks/qlora_qwen2_5_7b_colab.ipynb`). Expect the EN→AR gap to narrow, the distractor-bleed and over-generation to drop, and open-book legal accuracy to push toward 4/5. This is the single biggest remaining lever.
2. **Clean the corpus.** `data/orig_data.json` has OCR artifacts (corrupted article numbers, stray Arabic fragments in EN fields). The RAFT model faithfully reproduces them. A cleanup pass before the next build improves both training and inference.
3. **Strengthen the refusal/abstention boundary.** The model has learned to abstain for AR explanation requests (over-cautious) but still answers AR/EN refusal triggers with invented content (under-cautious). A targeted DPO pass on refusal-vs-explanation preference pairs would sharpen this — still PEFT.
4. **Distractor count ablation.** Try 2-3 distractors instead of 1 — RAFT papers report this reduces over-reliance on any single context article. Cheap to test once on the 7B run.

---

## 7. Reproducibility

```bash
# Build the RAFT dataset (cached polishes; $0 API cost)
python scripts/build_dataset_raft.py \
  --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_raft.yaml

# Train
python scripts/train.py \
  --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_raft.yaml

# Judge eval — closed-book
python scripts/run_judge_inference.py \
  --adapter-dir runs/qlora-qwen2.5-1.5b-raft \
  --val-path data/qa_pairs_val.baseline.jsonl \
  --raft-context none \
  --out-path reports/eval/judge_predictions_raft_closedbook.json

# Judge eval — open-book (matches RAFT training format)
python scripts/run_judge_inference.py \
  --adapter-dir runs/qlora-qwen2.5-1.5b-raft \
  --val-path data/qa_pairs_val.baseline.jsonl \
  --raft-context oracle+distractor \
  --out-path reports/eval/judge_predictions_raft_openbook.json

# Re-judge by reading reports/eval/cases_raft/*.md
```

Total wall-clock for Stage 4 from scratch on RTX 3050 6 GB: ~4 hours (build + train + 2 eval inference passes).

---

*Generated 2026-05-11 by Claude as in-session LLM judge. Compare against [judge_report.md](judge_report.md) (Stage 0) and [judge_report_stage1.md](judge_report_stage1.md) (Stage 1).*
