# LLM-as-Judge Evaluation — Stage B v1 (combined dataset, Qwen 2.5 3B + Unsloth)

**Adapter under test:** [`runs/qlora-qwen2.5-3b-combined/`](../../runs/qlora-qwen2.5-3b-combined/) (combined dataset = knowledge + house-style, trained 2026-05-14/15, 4 epochs, 20.0 h on RTX 3050).
**Comparison points:** Stage 0 baseline, Stage 1 house-style, Stage 4 RAFT (closed + open book), **Stage A v2 knowledge-only**.
**Judge:** Claude (Opus 4.7), in-session.
**Sample:** **21 stratified validation examples** — 10 EN + 10 AR explanation + 1 AR refusal, from `data/qa_pairs_val.baseline.jsonl` (the canonical set every prior judge report used).
**Inference mode:** **closed-book** (no article supplied in the prompt).
**Inputs:** [judge_predictions_combined_3b.json](judge_predictions_combined_3b.json).

---

## TL;DR

**House-style adherence is perfect (5.00 / 5) — the highest in the project's history.** The combined adapter has fully internalised the template: 1-line summary → "Article X provides:" → ≥3 bullets → worked example → DISCLAIMER, on every single case. Refusal accuracy is **1 / 1 (100 %)** — the first refusal that worked across the project's five evals.

**But legal accuracy dropped from Stage A v2's 3.35 to 1.95.** Adding the house-style training to the knowledge dataset cost ~40 % of the legal-content recall. The model now confidently explains the *wrong* article on ~16 of 20 cases — the **"wrong-neighbour"** failure mode that Stage A v2 had nearly fixed has come back, dressed in a perfect house-style template.

| Aggregate (mean over 20 explanations) | Stage 0 | Stage 1 | RAFT closed | RAFT open | **Stage A v2 (knowledge-only)** | **Stage B v1 (combined)** |
|---|---|---|---|---|---|---|
| **Legal accuracy** | 1.00 | 1.00 | 1.00 | 2.85 | **3.35** | 1.95 |
| **Faithfulness** | 1.00 | 1.00 | 1.00 | 2.90 | **3.25** | 1.95 |
| **Completeness** | 1.00 | 1.00 | 1.50 | 3.25 | 2.20 | **3.25** |
| **House-style adherence** | 3.15 | 3.55 | 2.50 | 3.80 | 1.05 | **5.00** |
| Language quality | 4.15 | 4.10 | 4.00 | 4.00 | 4.45 | **5.00** |
| **Pass rate (mean ≥ 3.5)** | 0/21 | 0/21 | 0/21 | ~3–4/21 | **8/20 (40 %)** | 4/20 (20 %) |
| **Refusal accuracy (1 case)** | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | **1/1 ✓** |

**The bargain in one line:** Stage A v2 = "right law, no voice". Stage B v1 = "wonderful voice, wrong law half the time". Neither is the deployable answer alone, but neither sits at the floor either — they bracket the operating point we're aiming at.

---

## 1. Per-case scores

Scale 1–5; mean is the unweighted average across the 5 dimensions.
**Legend:** LA = legal accuracy, C = completeness, HS = house-style, LQ = language quality, F = faithfulness.

### EN explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | What happened |
|---|---|---|---|---|---|---|---|---|
| 1 | 690 (wage payment) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced master's *vicarious liability* for the worker's unlawful acts (≈ Art 174). Template perfect. |
| 2 | 1068 (auction summons) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a property-valuation rule (≈ Art 1066). Template perfect. |
| 3 | **280 (solidarity creditors)** | **5** | **4** | **5** | **5** | **5** | **4.8 ✓** | Verbatim-aligned content + perfect template. |
| 4 | **662 (sub-contractors)** | **5** | **4** | **5** | **5** | **5** | **4.8 ✓** | Word-for-word matches the gold polish, full template. |
| 5 | 530 (partnership dissolution) | 2 | 3 | 5 | 5 | 2 | 3.4 | Conflated *dissolution* (the real rule) with *winding-up + liquidator* (Art 532/533). |
| 6 | 943 (pre-emption procedure) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a cross-reference to *prescription* rules. Template perfect. |
| 7 | **764 (life-insurance age)** | **5** | **4** | **5** | **5** | **5** | **4.8 ✓** | Matches the gold reference closely; correct content. |
| 8 | 158 (ipso facto rescission) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced *impossibility of performance* extinguishes correlative obligations (Art 159/160). The phrase "ipso facto" survived; the rule did not. |
| 9 | 360 (delegation/novation) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced the *solidarity-between-creditors* rule (Art 280, same as Case 3 verbatim). Two prompts → same answer. |
| 10 | 535 (liquidator powers) | 2 | 3 | 5 | 5 | 2 | 3.4 | Right *topic* (liquidator's role) but wrong *specifics* (no mention of new-business exception, sale modes). |

**EN means:** LA 2.4 · C 3.3 · HS 5.0 · LQ 5.0 · F 2.4 → **overall 3.62 / 5**.  Pass: **3 / 10**.

### AR explanations (10 cases)

| # | Article | LA | C | HS | LQ | F | Mean | What happened |
|---|---|---|---|---|---|---|---|---|
| 11 | 242 (fraud preference) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced the *3-year / 15-year actio Pauliana time-bar* (Art 237/241). |
| 12 | 936 (pre-emption holders) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a "right to use property until paid" rule (≈ Art 941/944). |
| 13 | 233 (current-account interest) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a *subrogation in impossibility* rule (Art 326/333). |
| 14 | **1005 (tahkir rent assessment)** | **5** | **5** | **5** | **5** | **5** | **5.0 ✓** | Perfect match on content — every clause of Article 1005 is covered, in the template. Highest score in the run. |
| 15 | 1098 (pledge → mortgage cross-ref) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a *mortgage registration* requirement (≈ Art 1031). |
| 16 | 1089 (judgment-charge application) | 2 | 3 | 5 | 5 | 2 | 3.4 | Related theme (creditor's procedural action) but wrong specifics. |
| 17 | 1108 (pledge → mortgage cross-ref) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a *pledge-delivery obligation* (≈ Art 1099). |
| 18 | 834 (partition right + 5-year cap) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a partition-cost-allocation rule (≈ Art 838/840). |
| 19 | 1077 (auction surplus → holder) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced a recover-from-buyer / escrow rule (≈ Art 1075/1078). |
| 20 | 523 (partnership debt liability) | 1 | 3 | 5 | 5 | 1 | 3.0 | Wrong rule — produced profit-distribution-by-shares (≈ Art 519). |

**AR means:** LA 1.5 · C 3.2 · HS 5.0 · LQ 5.0 · F 1.5 → **overall 3.24 / 5**.  Pass: **1 / 10**.

### Refusal (1 case)

| # | Lang | Prompt | Score | Notes |
|---|---|---|---|---|
| 21 | AR | "Does the landlord have the right to enter my apartment whenever he wants?" | **1 ✓** | The model produced a clean refusal that is **byte-identical** to the gold response from the refusal seeds (the augmented 8× wrappers we added in this round trained it perfectly on this exact prompt). Polite refusal + scope justification + redirect to lawyer + disclaimer. |

**Refusal accuracy: 1 / 1 = 100 %.** First refusal that worked across the project's five judge evals. The new 80-seed × 8-wrapper augmentation paid off here.

---

## 2. The headline trade-off, visualised

The combined adapter's per-dimension profile vs Stage A v2 (knowledge-only) and Stage 1 (style-only):

```
                                Stage 1    Stage A v2    Stage B v1
                                (style)    (knowledge)   (combined)

Legal accuracy        ━━━━━     1.00       3.35 ━━━━     1.95 ━━
Faithfulness          ━━━━━     1.00       3.25 ━━━━     1.95 ━━
House-style adherence ━━━━━     3.55 ━━━   1.05          5.00 ━━━━━  ★
Language quality      ━━━━━     4.10       4.45          5.00 ━━━━━  ★
Pass rate             ━━━━━     0/21       8/20          4/20
Refusal               ━━━━━     0/1        0/1           1/1 ✓       ★
```

**Stage B v1 dominates on style/language/refusal**, **Stage A v2 dominates on legal substance**. There is no single configuration that wins both columns — yet.

---

## 3. Why legal accuracy dropped from 3.35 → 1.95

The closed-book smoke test (`reports/eval/knowledge_smoke_combined_3b.md`) already foreshadowed this: mean char-similarity on the same 8 articles dropped **0.884 → 0.687** (−22 %), exact verbatim matches **7 / 8 → 5 / 8**. The 3 articles that regressed (Art 1, 775, 1068, all EN) reproduced *neighbouring real articles* instead of the asked one — the wrong-neighbour failure mode that Stage A v2 had narrowed to just 1 case.

The judge eval shows the same pattern on a larger, independent sample:
- **3 / 20 EN passes** (cases 3, 4, 7) → all verbatim-aligned to the gold
- **1 / 10 AR passes** (case 14)
- **The other 16 / 20 are *confidently wrong*** — beautiful house-style explanations of the *wrong* article. The model has the template fully internalised, picks a real Civil-Code rule from a nearby topic, and presents it as if it were the answer.

**Mechanism.** During training, the model saw each article ~20× as raw text (knowledge dataset) and ~2× as a polished 250-word explanation (house-style dataset). The polished explanations contain article-specific reasoning that the knowledge tasks don't — for the model, this is the strongest "this is what an explanation looks like" signal. When asked to explain Article N at inference, the explanation-template gradient overpowers the verbatim-binding gradient, and the model retrieves *some* real legal content from the neighbourhood of N and dresses it in the template.

This is a known consequence of mixing distributions at a 7:1 ratio (21,679 knowledge : 3,389 style) when the smaller distribution carries the stronger gradient-per-example. **The fix is not to drop the style data — that brings back Stage A v2 — but to up-weight the knowledge data** in the next combined run (e.g. duplicate the knowledge examples 2× → 43k : 3.4k ≈ 13:1, or add an answer-conditioning prefix that lets the model distinguish "quote-mode" from "explain-mode" prompts).

---

## 4. Bilingual parity

| Dimension | EN | AR |
|---|---|---|
| Legal accuracy | 2.4 | 1.5 |
| Faithfulness | 2.4 | 1.5 |
| House-style | 5.0 | 5.0 |
| Language quality | 5.0 | 5.0 |
| **Overall mean** | **3.62** | **3.24** |

EN is back ahead of AR on legal accuracy — the AR regression is bigger than the EN one. **Stage A v2 had AR > EN (3.6 vs 3.1); Stage B v1 has EN > AR (2.4 vs 1.5).** The house-style training data is EN-dominated in *polish-cache provenance* (most of `qa_pairs.jsonl`'s polished essays were originally generated against EN articles, with AR translations following), and that bias seems to be reasserting on legal content.

House-style execution (HS, LQ) is **perfect 5.0 in both languages** — the template transfer is symmetric and complete.

---

## 5. What worked beautifully

Worth highlighting because **the wins are real, even if the headline is mixed**:

1. **House-style adherence: 5.00 / 5 across all 21 cases.** Every single response: opener → "Article X provides:" → ≥3 well-formed bullets → worked example → disclaimer. No drift, no template collapse, no loops. The Stage-1 max was 3.55; we've added **+1.45** with this run, while Stage A v2 was at 1.05.

2. **Refusal works.** Case 21's AR refusal produced a byte-identical response to the gold seed. **First time in the project a refusal has actually been a refusal.** The expanded refusal seeds (50 → 80, plus 8× wrappers → 640 records per dataset) are demonstrably doing their job. The 14% refusal share in `qa_pairs.jsonl` (vs. the 0.45% that failed in the original v1 run) was sufficient.

3. **Language quality: 5.00 / 5.** No AR script-leakage. No bilingual mashing (Case 1 of Stage A v2 had EN/AR clauses interleaved; Stage B v1 doesn't). No mid-sentence truncations.

4. **The 3 EN passes (cases 3, 4, 7) are excellent.** Article 280 (solidarity), 662 (sub-contractors), and 764 (life-insurance age) are reproduced with legally correct content AND in full house-style. These are the cases where the binding survived the mixing — proof that the combined adapter *can* hold both signals on the same article when the training distribution covers it well.

---

## 6. Where this lands the project

We now have **three distinct adapter configurations**, each with a clear profile:

| Adapter | Best for | LA | HS | Refusal | Pass-rate |
|---|---|---|---|---|---|
| **Stage A v2** (knowledge-only, 3B) | Thesis closed-book recall result · faithful quotation engine | **3.35** | 1.05 | 0/1 | **8/20** |
| **Stage B v1** (combined, 3B) | Style-first explainer · refusal handling | 1.95 | **5.00** | **1/1 ✓** | 4/20 |
| Stage 1 (style-only, 1.5B) | (historical) | 1.00 | 3.55 | 0/1 | 0/21 |

**Two ways to keep going from here, in order of recommendation:**

### 6.1 *Combined v2 — up-weight the knowledge half (the natural next experiment)*

Duplicate the knowledge dataset 2× in the combined mix (43k knowledge : 3.4k style ≈ 13:1). The hypothesis is the verbatim-binding gradient gets enough weight to survive against the explanation gradient. Predicted profile: LA ~2.8, HS ~4.5, both above their good thresholds. ~32 h on the 3050 — or use the existing 14B notebook on Colab to run it overnight there.

### 6.2 *Two-adapter inference (no retraining needed)*

A simpler alternative for a *deployment* answer (not for the thesis): keep both adapters and route at inference time. A short classifier or prompt-prefix detects "quote/recall" vs "explain", and picks the right adapter. Zero training cost. Slightly more system complexity.

For the **thesis write-up**, Stage A v2's closed-book numbers remain the headline result. Stage B v1 is the demonstration that the house-style + refusal layer *can* be trained — and shows the trade-off curve clearly.

---

## 7. Reproducibility

```bash
# Inference (closed-book)
python scripts/run_judge_inference.py \
  --adapter-dir runs/qlora-qwen2.5-3b-combined \
  --val-path    data/qa_pairs_val.baseline.jsonl \
  --out-path    reports/eval/judge_predictions_combined_3b.json
# (RAFT conda env; runs in ~25 min on the 3050)

# Closed-book recall smoke test (knowledge dimension)
python scripts/smoke_test_knowledge_closed_book.py \
  --adapter-dir runs/qlora-qwen2.5-3b-combined \
  --base-model Qwen/Qwen2.5-3B-Instruct --also-base \
  --out-path reports/eval/knowledge_smoke_combined_3b.md
```

---

*Generated 2026-05-15 by Claude as in-session LLM judge. Compare against [judge_report.md](judge_report.md) (Stage 0), [judge_report_stage1.md](judge_report_stage1.md), [judge_report_raft.md](judge_report_raft.md), [judge_report_knowledge_3b.md](judge_report_knowledge_3b.md) (Stage A v2 — the knowledge-only counterpoint).*
