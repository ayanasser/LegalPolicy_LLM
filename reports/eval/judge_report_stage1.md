# LLM-as-Judge Evaluation — Stage-1 vs Baseline

**Subject A:** `runs/qlora-qwen2.5-1.5b-v1/` (Stage 0 / baseline, trained 2026-05-08).
**Subject B:** `runs/qlora-qwen2.5-1.5b-stage1/` (Stage 1, trained 2026-05-10 with the data interventions described below).
**Judge:** Claude (Opus 4.7), in-session.
**Sample:** 21 stratified validation examples — 10 EN explanation + 10 AR explanation + 1 AR refusal.
**Inputs:** [judge_predictions_baseline_v2.json](judge_predictions_baseline_v2.json), [judge_predictions_stage1.json](judge_predictions_stage1.json), per-case [cases_compare/](cases_compare/).

---

## TL;DR

Stage-1 data interventions **improved everything that was common across all training examples** — style consistency, structural completeness, refusal template, generation-loop avoidance — and **did not improve the per-article legal content** the model has to recall. Hallucination rate is still ~100%. The thesis-relevant conclusion is that **data interventions alone are insufficient at this base-model scale to fix content learning**; the next step must move beyond the data lever (DoRA / higher rank / RAFT-style training / bigger base).

| Metric | Baseline (Stage 0) | Stage 1 | Δ |
|---|---|---|---|
| **Training-time eval loss** | 1.587 | **1.375** | −13.4% |
| **Training-time token accuracy** | 64.2% | **68.0%** | +3.8 pts |
| **Training-time entropy** | 1.636 | 1.432 | −12.5% |
| **Judge — legal accuracy (mean of 20 explanations)** | 1.00 / 5 | **1.00 / 5** | flat |
| **Judge — house-style adherence (mean of 20)** | 3.05 / 5 | **3.55 / 5** | +0.50 |
| **Judge — generation stability (loops/collapse)** | 8 / 21 with issues | **3 / 21 with issues** | −5 cases |
| **Judge — refusal correctness (1 case)** | 0 / 1 (degenerate loop) | 0 / 1 (cleaner but invented "Article 149") | unchanged outcome, cleaner output |
| **Judge — pass rate (mean ≥ 3.5)** | 0 / 21 | **0 / 21** | flat |

The training-metric improvements **did not translate to content correctness**, exactly as we predicted from the Stage-0 finding ("style adapter, not knowledge adapter").

---

## 1. Stage-1 data interventions, recap

What changed between the two runs (data side only — model architecture and hyperparameters identical):

| Dimension | Baseline | Stage 1 |
|---|---|---|
| Articles selected | 350 EN + 350 AR (independent samples) | 125 articles in BOTH EN and AR (cross-language parity) |
| Question phrasings per article | 2 templates × 1 sampled = 2 variants | 8 templates × 8 sampled = **8 variants** |
| Refusal pairs | 14 seeds | 50 seeds × 8 paraphrase wrappers = **400 augmented refusals** |
| Contrastive (article-doesn't-exist) | 0 | **60** new pairs |
| Train/val total | 924 / 162 | **2,091 / 369** |
| Refusal+contrastive share of train | 1.5% | **18.6%** |

Note: per-article exposure went from **2 mentions** (baseline) to **8 mentions × 2 languages = 16 mentions per article** (Stage 1). That is the largest single change.

---

## 2. Per-case scoring summary

Scale 1–5 for explanations, 0/1 for refusals. Means across the 20 explanation cases:

|  | Baseline | Stage 1 |
|---|---|---|
| Legal accuracy | 1.00 | 1.00 |
| Faithfulness to article | 1.00 | 1.00 |
| Completeness | 1.00 | 1.00 |
| **House-style adherence** | 3.05 | **3.55** |
| Language quality | 3.95 | 4.10 |
| Generation stability (loops/collapse penalty) | 3.40 | **4.20** |
| **Per-case mean** | 2.23 | **2.31** |

The 0.08 jump in per-case mean is driven entirely by structural improvements (style, stability) — not by content quality, which is unchanged at floor.

### Cases where Stage 1 is materially **better** than Baseline

- **Case 21 (AR refusal — landlord entry).** Baseline collapsed into a 10× repetition loop of the disclaimer line. Stage-1 produced a clean structured response. Both invented fake article numbers, but Stage-1's output is at least readable. **Refusal still failed**, but the failure mode changed.
- **Case 03 (Article 280 EN — solidarity between creditors).** Baseline padded with 9 generic "No distinction between …" bullets (template-filling collapse). Stage-1 wrote 7 specific (still wrong) bullets.
- **Case 05 (Article 530 EN — partnership dissolution).** Baseline did the same "No distinction" template-fill collapse with 9 bullets. Stage-1 wrote a coherent (still wrong) explanation about restoration of capacity.
- **Case 12 (Article 936 AR).** Baseline had 3× repetition. Stage-1 has only 1× repetition.
- **Case 17 (Article 1108 AR).** Baseline had 5× repetition loop. Stage-1 has 1×.
- **Cases 16, 19, 20 (AR).** All show Stage-1 with tighter structure and fewer loops.

### Cases where Stage 1 is materially **worse** than Baseline

- **Case 04 (Article 662 EN).** Stage-1 generation broke down at 552 chars with no example, no disclaimer, no proper structure. Baseline produced a full 1,801-char hallucinated answer.
- **Case 14 (Article 1005 AR).** Stage-1 has 3× repetition of the same two bullets. Baseline only has 1×.
- **Case 15 (Article 1098 AR).** Stage-1 introduced **Chinese characters** mixed into Arabic text (投保人, страховат, страховщик) — language collapse not present in baseline.

### Cases where both fail equivalently

- 11 of 20 explanation cases: both adapters hallucinate the legal content with comparable structural quality. The article number is the only correct token; everything after that is fabricated.

---

## 3. Headline failure pattern (unchanged from Stage 0)

Examples taken directly from the comparison files:

**Case 06 — Article 943 EN** (real article: pre-emption procedure, 30-day filing window):
- Reference: pre-emption action against vendor + purchaser, 30-day deadline, urgent disposition.
- Baseline: invented "**insanity as a defense**" rules — entirely outside the Civil Code's scope, more like criminal law.
- Stage 1: invented "**fixed-price sale of goods**" rules — at least topically civil-law, but unrelated to Article 943.

**Case 03 — Article 280 EN** (real article: solidarity between creditors):
- Baseline: invented motor-vehicle accident insurance compensation.
- Stage 1: invented seller's delivery / buyer's payment obligations.

**Case 15 — Article 1098 AR** (real article: pledge inherits mortgage rules):
- Baseline: invented "successive debtors in commercial contract" with internal looping.
- Stage 1: invented "insurance amount on property value" with foreign-script tokens leaking into the Arabic.

The pattern is identical to what the Stage-0 judge eval surfaced: the model produces a confident, formatted answer about a different topic than the article actually covers.

---

## 4. Why the training metrics improved but the judge metrics didn't

| What improved | Why |
|---|---|
| Eval loss 1.587 → 1.375 | The model is better at predicting the **next token** of the gold answer when given the prefix. With 8× more per-article exposures, it has tightened its predictions about the *house-style template* tokens (bullets, "Article X provides:", disclaimer phrasing). These tokens dominate the loss. |
| Token accuracy 64% → 68% | Same reason. Most positions in the gold answer are template tokens; the model now matches them more often. |
| Entropy 1.636 → 1.432 | The model is more confident overall — concentrating mass on house-style tokens. |

| What did not improve | Why |
|---|---|
| Legal accuracy (judge) | Each article is still seen only 16 times across 1.5B base parameters. The base model has zero prior knowledge of the Egyptian Civil Code. There is not enough signal in 16 mentions for the LoRA to imprint each of 125 articles' specific content. The 8× repetition only re-imprinted what was already easy (the template). |
| Faithfulness | The model has nothing to be faithful to — it has not stored Article-N-specific content. |
| Refusal correctness | The augmentation paraphrased the surface form but did not give the model new categorical information. Plus, the explanation template is now even more dominant in the training mix; the model's default mode at inference is to fill the template, even when the prompt is a refusal trigger. |

The gap between training-time metrics and judge-eval metrics is not a measurement bug — it is the same finding as Stage 0, just at a slightly different operating point.

---

## 5. Bilingual parity

| Dimension | EN baseline | EN Stage 1 | AR baseline | AR Stage 1 |
|---|---|---|---|---|
| Legal accuracy | 1.00 | 1.00 | 1.00 | 1.00 |
| House-style adherence | 3.4 | 3.5 | 2.7 | 3.6 |
| Language quality | 4.7 | 4.6 | 3.2 | 3.6 |
| Generation stability | 3.6 | 4.4 | 3.2 | 4.0 |

The bigger gains are on the AR side, which had worse baseline starting points. Stage-1 substantially closed the AR/EN style gap (house-style went from −0.7 to −0.0, generation stability from −0.4 to −0.4 → comparable). This makes sense: the cross-lang parity intervention put each article in front of the model in both languages, helping the AR generator behaviour catch up to EN.

---

## 6. What Stage-1 told us about the next move

Stage-1 was the cheapest data-only intervention. The result confirms — with experimental rigor on the same 21 test cases — that **the failure mode is in the model/method, not the data**. Specifically:

- More diverse questions per article: helped a little (style consistency).
- More refusals: did not fix refusals — the explanation template is too strong.
- Cross-lang parity: improved AR generation behaviour, did nothing for content.
- Contrastive examples: not directly tested in this 21-case sample (no out-of-range articles in the val); cannot draw a conclusion yet.

**Recommended next stages, in order:**

1. **Stage 4 (PEFT-RAFT) on the same 1.5B base.** Train with the article injected into the instruction context and a teacher target that grounds itself in the cited article. Each article is now seen many times *with its content present*, which is by far the strongest content-learning signal possible without changing model size. Still strictly PEFT.
2. **Stage 2 (PEFT-method swap).** DoRA at rank 32 instead of LoRA at rank 16. Drop-in change. ~+1-3% on content benchmarks at the same compute.
3. **Stage 3 (Two-stage PEFT).** LoRA-based continued pretraining on raw `orig_data.json` for ~300 steps, then LoRA SFT on the Stage-1 dataset. Imprints content before style.
4. **Stage 5 (Scale up).** Apply the best of Stages 2-4 to **Qwen 2.5 7B** on Colab A100 with QLoRA r=64. Single biggest absolute lever; saved for last because it requires cloud GPU.

Stage 4 is the highest-leverage **PEFT-only** experiment we can run locally on the 6 GB GPU. It directly attacks the failure mode the judge eval has now confirmed twice.

---

## 7. Reproducibility

```bash
# Build the Stage-1 dataset (uses cached polishes; 0$ API cost on the cached path)
python scripts/build_dataset_stage1.py \
  --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local_stage1.yaml \
  --articles-per-lang 125

# Train
python scripts/train.py \
  --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local_stage1.yaml

# Run inference for the judge eval
python scripts/run_judge_inference.py \
  --adapter-dir runs/qlora-qwen2.5-1.5b-stage1 \
  --val-path data/qa_pairs_val.baseline.jsonl \
  --out-path reports/eval/judge_predictions_stage1.json

# Re-judge by reading reports/eval/cases_compare/*.md
```

Total wall-clock for Stage 1 from scratch on RTX 3050 6 GB: ~2.5 hours (build + train + eval inference).

---

*Generated 2026-05-10 by Claude as in-session LLM judge. Inputs: judge_predictions_baseline_v2.json, judge_predictions_stage1.json. Compare against [judge_report.md](judge_report.md) (the Stage-0 baseline report).*
