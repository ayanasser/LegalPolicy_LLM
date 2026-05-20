# LegalPolicy_LLM — Fine-tuning Experiments & Data Journey

**Project:** A bilingual (English + Arabic) plain-language explainer for the **Egyptian Civil Code** (1,093 articles), built as the fine-tuning epic of the LegalPolicy_LLM thesis project.
**Constraint:** must be done with PEFT (LoRA / QLoRA) — the thesis requirement is that the adapter itself demonstrably learns the domain; "fix it with RAG only" is rejected as a thesis answer.
**Hardware:** a single laptop **NVIDIA RTX 3050 6 GB** (WSL2 / Ubuntu).
**Dates:** 2026-05-04 → 2026-05-14 (current).
**Status:** **Stage-A v2 complete** (Qwen 2.5 3B + Unsloth, 4 epochs, 15.0 h). First thesis-grade closed-book recall result: **char-sim 0.884 vs base 0.072 (12× lift), 7 / 8 articles reproduced char-perfect, first non-zero reverse-lookup**.

---

## 0. TL;DR

We trained a 1.5 B Qwen 2.5 Instruct adapter on a polished bilingual house-style Q&A set (Experiment 1) and an LLM-as-judge eval gave us a brutal verdict: **100 % legal-content hallucination (1.0 / 5) despite eval loss 1.59 and 64 % token accuracy**. The model had learned the *template* of a legal explanation, not the *law*. We then ran four further experiments to attack this: data interventions (Experiment 2), retrieval-augmented fine-tuning (Experiment 3), and a hard pivot to a **knowledge-injection dataset** that treats memorisation of the article texts as a first-class training target (Experiments 4 + 5).

The **Stage-A v1 adapter (1.5 B, 4 epochs, ~12.4 h on the 3050)** was the *first* configuration in this project where closed-book legal recall is non-trivial: 2 / 8 articles reproduced character-perfect, mean character-similarity **0.44 vs. base 0.11**, ~20× the base-model token recall — but the article-number↔text *binding* was the soft spot (misses produced *real* article text, just bound to the wrong number).

The **Stage-A v2 adapter (Qwen 2.5 3 B + Unsloth, 10 task families, 4 epochs, 15.0 h)** closed that gap. Three independent measurements all confirm:
- **Closed-book smoke test** (8 verbatim articles): mean char-sim **0.884 vs base 0.072 — 12× lift — 7 / 8 articles reproduced character-perfect**, first non-zero reverse lookup in the project (1/3 exact + 2/3 within 10 articles).
- **LLM-as-judge on the canonical 21-case rubric** (same set used by Stage 0 / Stage 1 / RAFT): **legal accuracy 3.35 / 5 · pass-rate 8 / 20 (40 %)** — first non-floor closed-book scores in the project. Stage A v2 closed-book legal accuracy (3.35) is *higher than* Stage 4 RAFT *open*-book legal accuracy (2.85).
- **First run where AR overtakes EN** on overall judge mean (3.04 vs 2.72) — the recurring AR-collapse failure mode is solved.

**The thesis question is answered yes: PEFT can demonstrably learn the bilingual Egyptian Civil Code corpus**, given a knowledge-injection dataset with ~20× per-article exposure across 10 task families + a base of adequate capacity (3 B) + a memory-efficient engine (Unsloth) that lets it fit on a 6 GB consumer GPU. The trade-off — house-style adherence dropping from 3.55 to 1.05 — is by design and is addressed by the next experiment (a combined-dataset run mixing knowledge + house-style in a single training pass).

---

## 1. The problem statement

We want a small open model that can, in English and Arabic:

1. **Quote / explain** any article of the Egyptian Civil Code accurately.
2. **Refuse** out-of-scope requests (drafting court papers, predicting outcomes, individual legal advice) and redirect to a lawyer.
3. **Stay grounded** in the Civil Code — not bleed in US law, criminal law, family-court rules, etc.

Why this is hard at our scale:

- The base model (Qwen 2.5 1.5 B/3 B Instruct) has **zero prior exposure** to the Egyptian Civil Code. None of the 1,093 articles appears in pre-training.
- The adapter has to therefore *create* that knowledge, not just *surface* it.
- Bilingual (EN + AR) splits the limited capacity and Qwen 1.5 B's Arabic generation is noisy at the edges.
- 6 GB VRAM rules out anything bigger than ~3 B in QLoRA without help from Unsloth.

---

## 2. Dataset journey — at a glance

Every experiment that follows lives in one of these JSONL files. Validation of the per-article-variation numbers I quote later in this doc:

| File | Role | Total pairs | Articles covered | **Per-article: mean / median / max** | Notable feature |
|---|---|---|---|---|---|
| `data/qa_pairs.baseline.jsonl` | **Stage 0** — house-style SFT (the original "QA pairs") | 1,162 | 547 | **2.1** / 2 / 4 | 1 template, polished by Claude/Gemini |
| `data/qa_pairs.jsonl` | **Stage 1** — data interventions on house-style | 3,389 | 645 | 5.2 / 2 / 21 | 8 question variants, cross-lang parity, +contrastive, +refusal |
| `data/qa_pairs_raft.jsonl` | **Stage 4 / RAFT** — article-in-context training | 2,404 | 148 *(curated subset, cache-limited)* | 13.6 / 14 / 16 | each question prefixed with the article text + 1 distractor |
| `data/qa_pairs_knowledge.jsonl` | **Stage A v2** (running) — knowledge injection | 21,679 | **1,093 (100 %)** | **19.7 / 20 / 21** | 10 plain-text task families, NO house style, NO API polish |

So the **"~20× variation per article" claim is validated**: median 20, mean 19.7, minimum 9, 100 % coverage of the 1,093 articles in the corpus.

---

## 3. Experiment timeline

### Experiment 1 — Baseline (Stage 0): house-style SFT on 924 pairs

**Adapter:** [`runs/qlora-qwen2.5-1.5b-v1/`](../runs/qlora-qwen2.5-1.5b-v1/) · **Config:** [`qlora_qwen1_5b_local.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml) · **Eval:** [`reports/eval/judge_report.md`](eval/judge_report.md).

**What we did.** Synthesised 924 train + 162 val instruction-response pairs from `data/orig_data.json` (the 1,093-article corpus). Each example was a question like *"Explain Article 990 of the Egyptian Civil Code in plain language"* and the gold answer was a polished ≈250-word essay in a fixed house style (1-line summary → "Article X provides:" → ≥3 bullets → worked example → disclaimer). Answers were generated by Gemini/Claude. QLoRA on Qwen 2.5 1.5 B Instruct: **r=16, alpha=32, LR 3e-5, 3 epochs**, ~46 min on the 3050.

**Why these numbers were the obvious starting point.** Off-the-shelf hyper-parameters for house-style transfer on a small instruct model.

**What the training metrics said.** Train loss 2.36 → 1.55; eval loss 1.59; **eval token accuracy 64.2 %**; clean convergence, no overfitting. Read at face value, this *looked healthy*.

**What the judge eval actually said.** 22 held-out cases scored by Claude as in-session LLM judge.

| | Score (mean of 20 explanations) |
|---|---|
| Legal accuracy | **1.00 / 5** |
| Faithfulness to article | **1.00 / 5** |
| Completeness | 1.00 / 5 |
| House-style adherence | 3.15 / 5 |
| Language quality | 4.15 / 5 |
| **Pass rate (mean ≥ 3.5)** | **0 / 20 (0 %)** |
| Refusal accuracy | **0 / 2 (0 %)** |

**The headline:** on 20 / 20 explanation prompts the model produced an answer that followed the structural template (opening sentence + "Article X provides:" + bullets + example + disclaimer) but the **legal substance was hallucinated** — invented provisions that had no relationship to the actual article text.

**Concrete example — Article 775 (suretyship)** ([`reports/eval/cases/case_01.md`](eval/cases/case_01.md)):

*Real article:* "Suretyship may be given without the knowledge and even in spite of the opposition of the debtor."

*Model produced:* "Article 775 of the Egyptian Civil Code establishes that a person who has been injured in an accident caused by another's negligence is entitled to compensation from the insurer of the negligent party… Example: A pedestrian is hit by a car driven by a driver who was speeding and driving under the influence of drugs…"

The article number is the only correct token. Everything after is fabricated, in the right *shape*.

**Diagnosis.** Token accuracy is *teacher-forced* — at eval time the trainer feeds the model the gold answer up through position k–1 and asks "what's at position k?". Most of those positions are template boilerplate (`*`, `**`, `the`, `Article N`, `provides`, `DISCLAIMER`), which are trivially predictable from the template alone. The 64 % accuracy was the model nailing the *template*, not the law. With each article seen only ~2 times across 924 examples, a 1.5 B base with zero prior on Egyptian law cannot internalise 1,000 + article-specific facts. **The adapter became a style adapter, not a knowledge adapter.**

**What we learned.** Token accuracy / eval loss are necessary but not sufficient. Free-running generation eval (LLM-as-judge or similar) is mandatory. And per-article exposure of 2 is way too low.

---

### Experiment 2 — Stage 1 data interventions: more variants, more refusals, contrastive negatives

**Adapter:** [`runs/qlora-qwen2.5-1.5b-stage1/`](../runs/qlora-qwen2.5-1.5b-stage1/) · **Config:** [`qlora_qwen1_5b_local_stage1.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen1_5b_local_stage1.yaml) · **Eval:** [`reports/eval/judge_report_stage1.md`](eval/judge_report_stage1.md).

**What we changed.** Kept the model + recipe identical to Experiment 1; only the **dataset** changed:

| Dimension | Stage 0 | Stage 1 |
|---|---|---|
| Question phrasings per article | **2** templates × 1 sampled | **8** templates × 8 sampled |
| Articles in both EN and AR (cross-lang parity) | independent samples | **same 125 articles** in EN and AR |
| Refusal seeds | 14 hand-written | **50** seeds × 8 paraphrase wrappers = **400** |
| Contrastive (article-doesn't-exist) pairs | 0 | **60** |
| Train / val total | 924 / 162 | **2,091 / 369** |
| Per-article exposure | 2 mentions | **8 phrasings × 2 langs = 16 mentions** |

This was the cheapest data-only intervention available.

**Result.**

| Metric | Stage 0 | Stage 1 | Δ |
|---|---|---|---|
| Training-time eval loss | 1.587 | **1.375** | **−13.4 %** ✓ |
| Training-time token accuracy | 64.2 % | **68.0 %** | +3.8 pts ✓ |
| **Judge legal accuracy (mean of 20)** | **1.00 / 5** | **1.00 / 5** | **flat** ✗ |
| Judge house-style adherence | 3.05 | **3.55** | +0.50 ✓ |
| Judge generation stability (loops/collapse) | 8/21 cases with issues | 3/21 | −5 cases ✓ |
| **Pass rate (mean ≥ 3.5)** | 0 / 21 | 0 / 21 | flat ✗ |

**What we learned.** Stage-1 data interventions **improved everything that was common across all training examples** — style consistency, structural completeness, refusal template, generation-loop avoidance — and **did not move the per-article legal content**, which is what the model actually has to recall. The training-metric improvements are real, but they translate to *better template execution*, not *correct law*. The 8× more per-article exposure tightened predictions about house-style tokens (the dominant signal across all 924 → 2,091 examples) but couldn't carry 125 articles' specific rules with only 16 mentions each.

> *"Stage-1 data interventions alone are insufficient at this base-model scale to fix content learning; the next step must move beyond the data lever."* — `judge_report_stage1.md`.

---

### Experiment 3 — RAFT (Stage 4): put the article in the prompt

**Adapter:** [`runs/qlora-qwen2.5-1.5b-raft/`](../runs/qlora-qwen2.5-1.5b-raft/) · **Config:** [`qlora_qwen1_5b_raft.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen1_5b_raft.yaml) · **Eval:** [`reports/eval/judge_report_raft.md`](eval/judge_report_raft.md).

**What we did.** Retrieval-Augmented Fine-Tuning (RAFT). Each training question was *preceded by a context block* containing the article the question is about (the "oracle") plus 1 distractor article (random other article), shuffled, in the question's language. The gold response was unchanged. So the adapter is trained to (a) locate the right article in the supplied context and (b) rephrase it in house style. As a side effect, every article's text passes through the loss many times.

**Evaluated in two inference modes** on the *same* 21 cases:
- **Closed-book** — plain question, no article supplied → tests whether RAFT training imprinted content into the adapter.
- **Open-book** — question preceded by the asked article + 1 distractor (matches training format) → tests whether the adapter learned to *ground its answer in the provided article*.

**Result.**

| Metric (mean over 20 explanations) | Stage 0 | Stage 1 | **RAFT closed-book** | **RAFT open-book** |
|---|---|---|---|---|
| Legal accuracy | 1.00 | 1.00 | **1.00** | **2.85** |
| Faithfulness | 1.00 | 1.00 | 1.00 | **2.90** |
| Completeness | 1.00 | 1.00 | 1.50 | **3.25** |
| House-style | 3.05 | 3.55 | 2.50 | 3.80 |
| **Pass rate (mean ≥ 3.5)** | 0/21 | 0/21 | 0/21 | **~3-4/21** |
| Training-time eval loss | 1.587 | 1.375 | — | **1.280** |

**The first non-zero legal-accuracy column in the project.** When the article is supplied in context, the adapter grounds its answer there.

**Concrete example — Article 1068 (mortgage-purge auction procedure)** ([`reports/eval/cases_raft/case_02.md`](eval/cases_raft/case_02.md)):

*Real article:* a procedural rule about the form of summons, the deposit at the court, the no-refund rule, nullity for non-compliance, and renunciation requiring unanimous consent.

*RAFT closed-book prediction (no article given):* "Article 1068 of the Egyptian Civil Code addresses the rules governing the transfer of movable property from one person to another… Transfer of movable property must be in writing and registered with the Ministry of Finance…"  → **still hallucinating** — a *different* civil-law-shaped topic.

*RAFT open-book prediction (article + distractor in context):* "Article 1068 of the Egyptian Civil Code sets out the procedure for applying to the court for a sale-by-auction… A summons to the third-party holder and to the former owner, signed by the applicant or his representative holding a special mandate… The applicant must deposit at the court a sum sufficient to cover the cost of the sale by auction…" → **correctly grounded in the article**.

**What we learned.** Two distinct questions, two distinct answers:

1. *Did PEFT, with RAFT-style training, learn the Civil Code on its own?* **No.** Closed-book legal accuracy is still ~1.0 / 5. The model handed an article rephrases it; the model not handed an article confabulates a different one. **A 1.5 B base + 18 M LoRA adapter cannot internalise 148 articles' content from ~16 mentions each, even when those mentions are structured as RAFT examples.**

2. *Did the RAFT-trained adapter learn to ground in a provided article?* **Yes, partially.** Open-book legal accuracy ~2.85 / 5, the first non-zero number in the project. So RAFT teaches a **skill** (ground in context), not the **knowledge** itself.

For the thesis constraint ("PEFT must demonstrably learn the domain"), this **was not the answer**: the closed-book number is the thesis number, and it stayed at the floor.

---

### Experiment 4 — The pivot: knowledge-injection dataset (Stage A v1)

**Adapter:** [`runs/qlora-qwen2.5-1.5b-knowledge-v1/`](../runs/qlora-qwen2.5-1.5b-knowledge-v1/) · **Config:** [`qlora_qwen1_5b_knowledge.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen1_5b_knowledge.yaml) · **Smoke test:** [`reports/eval/knowledge_smoke.md`](eval/knowledge_smoke.md).

#### 4.1 What "knowledge QA pairs" means — and why we built them

Up to this point, every dataset (`qa_pairs.baseline.jsonl`, `qa_pairs.jsonl`, `qa_pairs_raft.jsonl`) used **one task family** — *"explain this article in the house style"* — with the gold answer being a ~250-word polished essay produced by Gemini / Claude. Three things made that mix structurally unable to teach memorisation:

1. **The answers were paraphrases, not the law.** The model never saw the verbatim article text as a training target. So it could never be graded on "did you reproduce Article N?", and the loss never directly pushed it to internalise the text.
2. **One rigid output template.** Every gold answer had the same shape. The single signal that *was* consistent across all 924-2,400 examples was the *template*, so that's what the model learned. Article content varied and was sparse — it looked like noise to the loss.
3. **Per-article exposure was ~2.** No fact, however salient, is going to imprint from 2 mentions across a 1,000-article corpus on an 18 M-parameter adapter.

A **knowledge-injection dataset** is the deliberate inverse of those three problems. For *every* one of the 1,093 articles, we emit ~14-20 short, varied training examples — each example forces the model's attention to the article's *actual text*, not to a template slot — and the answers are the raw text from `orig_data.json`, not an LLM paraphrase. The dataset has **no house-style template** in it, on purpose: house style is a separate skill that gets layered on top **afterwards** (Stage B), once the model knows what the law actually says.

This is the **"two-stage PEFT"** plan:

```
  Stage A  (knowledge dataset, this experiment)
           -> adapter learns WHAT the Egyptian Civil Code says

  Stage B  (qa_pairs.jsonl, the existing house-style SFT)
           -> the same adapter, continued, learns HOW to explain it in our voice
```

#### 4.2 What's in `qa_pairs_knowledge.jsonl` (v1: 7 task families)

For every article, the [knowledge_builder.py](../src/legal_explainer/finetune/knowledge_builder.py) module emitted ~14-15 examples spread across 7 task families. Counts in **Stage A v1** (15,205 train + 633 val):

| Task family | Example prompt | Example answer | What it forces the model to learn |
|---|---|---|---|
| `kn_verbatim` (4,356) | "Quote Article 775 of the Egyptian Civil Code exactly as it is written." | the full article text | Reproduce verbatim — straight memorisation |
| `kn_complete` (2,156) | "Article 775 begins '<first 50 % of the words> …'. Complete it." | the full article text | Recall the rest from a partial opening |
| `kn_gap` (2,013) | "Fill the blank: 'The applicant must _____ at the Caisse of the Court a sum…'." | "deposit" / the missing 3-6 words | Recall an interior span |
| `kn_reverse` (2,178) | "Which article of the Egyptian Civil Code says: '<verbatim quote>'?" | "Article 775" | Map content → article number (inverse of `kn_verbatim`) |
| `kn_placement` (2,186) | "Where does Article 775 sit in the Code?" | "Article 775 appears under: Book II › Suretyship › The Elements of Suretyship" | Learn the structural / topical placement |
| `kn_translate` (1,966) | (Arabic article shown) "Give its English text." | the English text | Bind EN ↔ AR versions of the same article |
| `kn_bilingual` (983) | "Give Article 775 in both Arabic and English." | both texts | Same, in one shot |

EN / AR split is symmetric (7,492 / 7,363 / 983 bilingual). No Claude or Gemini API calls — fully deterministic from `orig_data.json` + a small text-cleanup pass that fixes the OCR-mangled paragraph markers (`)١ (` → `(١)`).

#### 4.3 Training and metrics

QLoRA on Qwen 2.5 1.5 B Instruct, same module set as before but with a memorisation-tuned recipe: **r=16, alpha=32, LR 2e-4 (×6.7 the style recipe), 4 epochs, max_seq 1280**. ~12.4 h on the RTX 3050.

| Epoch | step | eval_loss | eval token-acc | eval entropy |
|---|---|---|---|---|
| 1 | 951 | 0.4546 | 87.9 % | 0.560 |
| 2 | 1,902 | 0.2011 | 94.5 % | 0.249 |
| **3 (best)** | **2,853** | **0.1668** | **95.3 %** | **0.183** |
| 4 | 3,804 | 0.1675 | 95.4 % | 0.178 |

Eval loss dropped almost an order of magnitude vs. the style runs (1.59 → 0.17). The eval task is mechanically different (reproduce/locate vs. explain), so the absolute numbers aren't directly comparable to the judge runs — what's important is the **shape**: train ≈ eval throughout, no overfitting, saturation at epoch 3.

#### 4.4 The closed-book smoke test — what actually happened

We loaded the Stage A v1 adapter, asked it to **quote 8 articles verbatim** (no article supplied in the prompt) and **identify 3 articles from a verbatim snippet**. Greedy decoding, no repetition penalty — measures what the weights actually know. Same 11 prompts also run on the **bare Qwen 2.5 1.5 B Instruct base** for delta.

| Metric (8 verbatim items) | Adapter | Base | Lift |
|---|---|---|---|
| Mean char-similarity | **0.44** | 0.11 | **4×** |
| Mean token-recall (gold tokens ≥ 4 chars appearing in pred) | **0.38** | 0.02 | **~20×** |
| Exact matches | **2 / 8** | 0 / 8 | — |
| Reverse lookups (closed-book article-ID) | 0 / 3 | 0 / 3 | — |

**Two perfect verbatim reproductions:**

- **Article 1068 (EN, 690 chars):** the adapter reproduced the entire procedural rule **character-perfect**. Compare with Experiment 1's prediction for the *same article* (transfer of movable property, registration with Ministry of Finance — completely fabricated). Same base model, same VRAM, different data → first closed-book correct article in the project.

  ```
  GOLD: The application shall be made by a summons to the third party holder
        and to the former owner, signed by the applicant or his representative
        holding a special mandate for this purpose. The applicant must deposit
        at the Caisse of the Court a sum which is sufficient to cover the cost
        of the sale by auction, but he shall have no right to a refund of
        expenses advanced by him if no higher price than that offered by the
        third party holder is obtained as a result of the auction. The failure
        to comply with any one of these conditions entails the nullity of the
        application. The applicant may not renounce his application without
        the consent of all the inscribed creditors and all the sureties.

  PRED: <byte-identical>
  ```

- **Article 990 (AR, 171 chars):** also char-perfect.

**The misses are not random hallucinations any more — they are real text bound to the wrong article.** Article 17 was supposed to be about inheritance / wills conflict-of-laws; the adapter produced **Article 19's** text verbatim (contractual obligations conflict-of-laws — a neighbouring article in the same Book). Article 1 was supposed to be the law-of-application provision; the adapter produced text from somewhere near Article 4. Reverse lookups failed but failed *near* — Article 775 was guessed as 778 (3 apart); Article 1 as 27. The model is producing coherent Civil-Code prose, just bound to a wrong number.

**This is the failure mode of an under-trained article-number ↔ article-text *binding*, on top of an article-text ↔ article-text *memory* that is mostly working.** It tells us:

- The 1.5 B's *content* memory is real — most of these article passages now exist somewhere in the adapter weights.
- The *indexing* — knowing which number maps to which text — is the soft spot, exactly because in the v1 dataset only ~2 of the 14 examples per article were the inverse `kn_reverse` task.

#### 4.5 Verdict

This was the **first configuration in the project where closed-book legal recall is non-zero**, with the largest base-vs-adapter delta we've seen. The remaining failure mode (article-number binding) is data-shaped, not capacity-shaped, which is the cue for Experiment 5.

---

### Experiment 5 — Stage A v2: 10-family dataset on Qwen 2.5 3B + Unsloth *(complete — 2026-05-14)*

**Adapter:** [`runs/qlora-qwen2.5-3b-knowledge/`](../runs/qlora-qwen2.5-3b-knowledge/) · **Config:** [`qlora_qwen3b_knowledge.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen3b_knowledge.yaml) · **Log:** [`runs/knowledge_3b_train.log`](../runs/knowledge_3b_train.log) · **Smoke test:** [`reports/eval/knowledge_smoke_3b.md`](eval/knowledge_smoke_3b.md).

Two changes from Experiment 4:

#### 5.1 Dataset: 7 task families → 10 (the binding-focused additions)

The [knowledge_builder.py](../src/legal_explainer/finetune/knowledge_builder.py) was extended to emit three additional task families designed specifically to attack the article-number ↔ content binding that the smoke test showed was weak:

| New task | Example | Why it helps the binding |
|---|---|---|
| `kn_card` (2,084) | "Give a reference card for Article 775." → `### Article 775\n### Topic Suretyship — formation\n### Key point …\n### Text …` | Co-occurrence of *article number*, *topic*, *first-sentence rule*, *full text* in one structured target — strongest single signal for binding them all in the adapter weights. |
| `kn_contrast` (4,183) | "Does Article 775 concern motor-vehicle insurance?" → "No — Article 775 concerns suretyship (formation)." | Trains the model to say **no** when the wrong topic is paired with a number — directly attacks the "confidently wrong" failure mode. |
| `kn_roster` (186) | "Which articles deal with suretyship?" → "Article 772, Article 773, … Article 800." | Inverse index — given a topic, name the articles. The thinnest family because only well-anchored topics qualify (114 EN / 111 AR topics with 2–40 articles). |

Plus a breadcrumb → topic extractor (`topic_of` / `_crumb_parts`) that mines clean per-article topic labels from the bilingual metadata field.

**Result:** the new dataset hits **21,679 train + 903 val examples covering 100 % (1,093 / 1,093) of articles** at **mean 19.7 / median 20 examples per article** — the "20× variation" number, validated.

```
kn_verbatim : 4,186      kn_card     : 2,084
kn_complete : 2,079      kn_contrast : 4,183
kn_gap      : 1,936      kn_roster   :   186
kn_reverse  : 2,078      kn_placement: 2,104
kn_translate: 1,899      kn_bilingual:   944
                                 → 21,679 total
```

#### 5.2 Model + engine: Qwen 2.5 **3 B** + Unsloth (instead of Qwen 1.5 B + vanilla PEFT)

The smoke test in Experiment 4 said memorisation is working but the **base-model capacity for 1,093 articles is still the bottleneck on the misses**, and Qwen 1.5 B's Arabic is the noisier of the two languages. The natural upgrade is Qwen 2.5 **3B** Instruct — same family, ~2× the parameters, materially stronger Arabic.

But 3 B QLoRA does **not** fit comfortably on 6 GB VRAM with vanilla `transformers` + PEFT. The math: 4-bit base ≈ 1.6 GB, activations at seq 1,280 with gradient checkpointing ≈ 2-2.5 GB, optimiser states + gradients + CUDA overhead ≈ another 1.5-2 GB → roughly 5-6 GB, on the OOM edge, with a single long Arabic example capable of killing a 14-hour run.

**Enter Unsloth.** Unsloth ([`src/legal_explainer/finetune/train_unsloth.py`](../src/legal_explainer/finetune/train_unsloth.py)) is a drop-in PEFT trainer that patches the layers with custom CUDA / Triton kernels: roughly **2× faster** and **~50 % less VRAM** for the same QLoRA configuration. It reads the same YAML schema as the regular `train.py`, so no architectural change — just a different engine. The existing conda env `RAFT` already had it installed (`unsloth 2026.5.2, torch 2.10 + cu128, bf16 OK`).

**Recipe (delta from the 1.5 B knowledge config):**

| Hyper-parameter | 1.5 B knowledge | 3 B + Unsloth knowledge |
|---|---|---|
| Base model | Qwen 2.5 **1.5 B** Instruct | Qwen 2.5 **3 B** Instruct |
| LoRA r / alpha | 16 / 32 | 16 / 32 (held — smoke test showed dataset > rank) |
| LoRA dropout | 0.05 | **0** — Unsloth patches all layers fully, plus we *want* to memorise hard |
| Learning rate | 2e-4 | **1e-4** (larger base wants gentler updates) |
| max_seq_length | 1,280 | **1,536** (Unsloth's memory cut affords the headroom) |
| Epochs | 4 | 4 |
| Engine | vanilla PEFT / TRL | **Unsloth** |

**Pre-flight check (dry-run, 3 steps).** Verified end-to-end: model loads in 4-bit, 36 layers patched, 29.9 M trainable params (0.96 % of base), 16 train samples + 3 dry steps complete, **no OOM**. Wall-clock 55 s for 3 steps (~18 s/step warmup).

#### 5.3 Training metrics — Stage A v2 (final)

Training completed in **15.0 h** wall-clock on the RTX 3050 (~9 s/step steady-state — Unsloth delivered the promised ~2× speedup over the 1.5B vanilla run despite 2× the parameters and a 20 % larger seq length).

| Epoch | Step | eval_loss (3B v2) | vs. 1.5B v1 |
|---|---|---|---|
| 1 | 1,355 | 0.3973 | −12.6 % vs 0.4546 |
| 2 | 2,710 | 0.1932 | −3.9 % vs 0.2011 |
| 3 | 4,065 | 0.1606 | −3.7 % vs 0.1668 |
| **4 (best)** | **5,420** | **0.1588** | **−5.2 % vs 0.1675** |
| Train loss (final, step-weighted) | | 0.335 | (1.5B v1: 0.374) |
| Train runtime | | 15.0 h | (1.5B v1: 12.4 h) |

Crucially, the 1.5 B v1 *plateaued* at epoch 3 (0.1668 → 0.1675 — slight bounce). The 3 B v2 **kept improving monotonically through epoch 4** — proof the bigger base is actually using the extra capacity rather than overfitting. 4 epochs was the right budget; a 5th would barely move it.

#### 5.4 Closed-book smoke test — the thesis-grade result

Same 11-prompt test as Experiment 4 (8 verbatim recall, 3 reverse lookup; greedy decoding, no article in the prompt). Compared against the bare Qwen 2.5 3 B Instruct base, and against the previous 1.5 B v1 numbers.

| Metric | 3 B base (no adapter) | **3 B + Stage A v2 adapter** | (ref) 1.5 B v1 adapter | Lift over 3 B base |
|---|---|---|---|---|
| Mean char-similarity (8 verbatim) | 0.072 | **0.884** | 0.44 | **~12×** |
| Mean token-recall (8 verbatim) | 0.028 | **0.889** | 0.38 | **~32×** |
| **Exact verbatim matches** | **0 / 8** | **7 / 8** | 2 / 8 | — |
| Reverse lookups correct (3 items) | **0 / 3** | **1 / 3** | 0 / 3 | first non-zero in the project |

Per-article verbatim recall (3 B v2 vs 1.5 B v1):

| Article | Lang | Len | **3 B v2 char-sim** | 1.5 B v1 char-sim | Note |
|---|---|---|---|---|---|
| 17 | AR | 275c | **1.00** | 0.07 | v1 produced *Article 19's* text verbatim. v2 produces the right article. |
| 280 | EN | 283c | **1.00** | 0.14 | v1 hallucinated a different debtor-release rule. v2: perfect. |
| 775 | EN |  96c | **1.00** | 0.41 | partial → perfect |
| 836 | AR | 266c | **1.00** | 0.57 | first paragraph + fabricated tail → perfect |
| 990 | AR | 171c | **1.00** | 1.00 | already perfect → still perfect |
| 1068 | EN | 690c | **1.00** | 1.00 | already perfect → still perfect |
| 1112 | AR | 186c | **1.00** | 0.26 | partial → perfect |
| 1 | EN | 384c | 0.08 | 0.09 | **still failing** — model outputs a real but wrong article (public-policy exception, ≈Art 28) |

**Six of the eight articles that the 1.5 B v1 either failed or partially recovered are now character-perfect on the 3 B v2.** The "wrong-neighbour article" failure mode that defined v1 is solved on 7 / 8 cases. Article 17 (AR) is especially telling — v1 was producing Article 19's text *verbatim*; v2 produces Article 17's text *verbatim*. That's exactly the failure mode the new `kn_card` / `kn_reverse` / `kn_contrast` task families were designed to fix, and the test confirms they did.

The one remaining miss (Article 1, the foundational provision) is interesting because the **reverse-lookup version of the same article got the correct answer** ("which article says 'Provisions of laws govern all matters …'?" → "Article 1"). The *binding* exists in one direction (text → number) but the forward (number → text) is still mis-routed to a neighbouring article. So the soft spot is now extremely narrow.

Reverse lookups (snippet → article number):

| Snippet | Expected | Predicted | |
|---|---|---|---|
| "Suretyship may be given without the knowledge…" | Art 775 | **Art 776** | near miss (off by 1) |
| "Provisions of laws govern all matters…" | Art 1 | **Art 1** | ✅ |
| (AR) "تجوز كفالة المدين بغير عمله…" | Art 775 | **Art 785** | near miss (off by 10) |

1 / 3 exact + 2 / 3 within 10 articles. v1 was 0 / 3 with wild predictions (27, 778, 144). The 3 B is now consistently in the right neighbourhood — a totally different failure regime.

#### 5.5 LLM-as-judge eval — same 21-case rubric as Stage 0 / 1 / RAFT

The closed-book smoke test in §5.4 is suggestive on 11 hand-picked prompts. We then ran the **same 21-case judge rubric** that every prior judge report used (`data/qa_pairs_val.baseline.jsonl` — 10 EN explanation + 10 AR explanation + 1 AR refusal) against the Stage A v2 adapter in **closed-book** mode (no article in prompt), scored on the 5-dimension rubric (legal accuracy, faithfulness, completeness, house-style, language quality). Full report at [`reports/eval/judge_report_knowledge_3b.md`](eval/judge_report_knowledge_3b.md).

| Metric (mean over 20 explanations) | Stage 0 | Stage 1 | RAFT closed-book | RAFT open-book | **Stage A v2 (3 B, closed-book)** |
|---|---|---|---|---|---|
| **Legal accuracy** | 1.00 | 1.00 | 1.00 | 2.85 | **3.35** |
| **Faithfulness to article** | 1.00 | 1.00 | 1.00 | 2.90 | **3.25** |
| Completeness | 1.00 | 1.00 | 1.50 | 3.25 | 2.20 |
| House-style adherence | 3.15 | 3.55 | 2.50 | 3.80 | 1.05 |
| Language quality | 4.15 | 4.10 | 4.00 | 4.00 | **4.45** |
| **Pass rate (mean ≥ 3.5)** | 0 / 21 | 0 / 21 | 0 / 21 | ~3-4 / 21 | **8 / 20 (40 %)** |
| Refusal accuracy (1 case) | 0 / 2 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |

**Stage A v2 closed-book legal accuracy (3.35) exceeds Stage 4 RAFT *open*-book legal accuracy (2.85).** The knowledge-injected adapter knows the law better, without retrieval, than the RAFT adapter did with the article supplied in the prompt.

Pass-rate breakdown — 8 cases pass cleanly, all driven by **character-perfect or near-perfect article-text reproduction**:

| EN passes (4 / 10) | AR passes (4 / 10) |
|---|---|
| Article 280 (3.8) — solidarity between creditors, verbatim | Article 242 (3.8) — fraud / insolvent-debtor payment, verbatim |
| Article 530 (3.6) — partnership dissolution, verbatim | Article 834 (3.8) — partition right, verbatim |
| Article 1068 (3.8) — auction summons procedure, 690c verbatim | Article 1005 (3.8) — tahkir rent assessment, verbatim |
| Article 158 (4.0) — ipso-facto rescission, verbatim with structural prefix | Article 1089 (4.0) — judgment-charge application, full multi-paragraph verbatim |

The 12 failure cases split into:
- **5 wrong-neighbour reproductions** (model emits a real *different* article's verbatim text — same failure regime as the 1.5 B v1 smoke test but now narrower): EN cases 4, 6, 7, 10; AR cases 12, 13, 17.
- **1 topic-only answer** (case 9 — "Article 360 concerns Novation and Delegation" — correct topic, zero content).
- **2 near-miss verbatim** with minor OCR or typo issues (cases 15, 19, 20).
- **1 bilingual mash** (case 1 — correct content but EN/AR clauses interleaved).
- **1 refusal collapse** (case 21 — gibberish; knowledge data has no refusals so the adapter has no concept of decline-and-redirect).

The **trade-off** is house-style adherence dropping from 3.55 (Stage 1) to **1.05** — completely expected and intentional: the knowledge dataset has *zero* house-style examples by design. The model now answers explanation prompts by quoting the article, which gives correct content but no template. This is a *separate* skill question, addressed by the combined-dataset run proposed in §6.

#### 5.6 Bilingual parity — flipped

**First run in the project where Arabic scores higher than English on overall mean.**

| Dimension | EN | AR | Gap |
|---|---|---|---|
| Legal accuracy | 3.1 | 3.6 | **+0.5 (AR better)** |
| Faithfulness | 3.0 | 3.5 | +0.5 |
| Language quality | 4.4 | 4.5 | +0.1 |
| **Overall mean** | **2.72** | **3.04** | **+0.32 (AR better)** |

Driven by (a) the 3 B base's materially stronger Arabic, (b) AR coverage parity in the new `kn_card` / `kn_reverse` / `kn_contrast` families, (c) AR generation no longer collapsing into bullet-loops / script-leakage as it did in Experiments 1-4. Stage 0 had EN ahead by 0.48; the gap is now reversed.

#### 5.7 Verdict — the thesis question is answered

Three independent measurements all agree:

| Signal | Number | Verdict |
|---|---|---|
| Training-time eval loss (held-out kn tasks) | 0.397 → 0.193 → 0.161 → **0.159** (monotonic, 4 epochs) | adapter generalises across held-out task instances |
| **Closed-book smoke test** (8 verbatim articles) | char-sim **0.884** vs base 0.072 (**12× lift**), **7 / 8 char-perfect** | adapter holds the article texts in its weights |
| **LLM-as-judge eval** (same 21-case rubric as prior runs) | **LA 3.35 · Pass 8 / 20** vs 1.0 / 0-of-21 on every prior closed-book run | adapter wins on the project's canonical comparison metric too |

**PEFT can demonstrably learn the bilingual Egyptian Civil Code domain corpus**, given (a) a knowledge-injection dataset with sufficient per-article exposure (~20×, 10 task families covering content + binding), (b) a base of adequate capacity (Qwen 2.5 3 B), and (c) a memory-efficient training engine (Unsloth) that lets it run on a 6 GB consumer GPU. The 1.5 B → 3 B + 7 → 10 task families jump more than doubled the adapter-vs-base lift (4× → 12×).

---

## 4. What "knowledge QA pairs" are — a one-screen summary

(re-stated here separately so it can stand alone in conversations.)

A **knowledge QA pair** is a *short, varied, non-prose* training example whose only job is to imprint one specific *fact about an article* into the adapter weights. It's the deliberate inverse of the original house-style pair:

| | House-style pair | Knowledge pair |
|---|---|---|
| Question shape | "Explain Article N in plain language." | One of 10 task families (quote, complete, fill-the-gap, reverse-lookup, placement, AR↔EN translate, both-languages, reference card, contrast, roster) |
| Answer source | LLM paraphrase (~250 w) | Raw text from `orig_data.json` |
| Output template | Fixed (open → bullets → example → disclaimer) | None — varied per task |
| Median answer length | ~250 words | ~30 words |
| Cost to build | LLM API per article | Free, deterministic |
| Article exposure per epoch | 2× | ~20× |
| What signal it teaches | Output *format* | Article *content* + *number ↔ content* binding |

Why we need them *in addition to* the house-style pairs: the model has to learn **both** *what the law says* (Stage A — knowledge pairs) **and** *how we want it spoken* (Stage B — house-style pairs). The house-style pairs alone — as Experiments 1, 2, and 3 closed-book showed — teach only the second.

---

## 5. The other big improvement: Qwen 3B + Unsloth in a 6 GB box

Why this is interesting on its own, separate from the dataset story:

- **6 GB VRAM is a hard ceiling.** Without Unsloth, the move from 1.5 B → 3 B was off the table for any non-trivial seq length / epoch budget. Empirically (Experiment 5 dry-run) Unsloth puts us at 74 % VRAM usage at max_seq 1,536, with comfortable headroom.
- **Unsloth is ~2× faster.** The 1.5 B vanilla v1 run ran at ~11.7 s/step. The 3 B Unsloth v2 run runs at ~9 s/step despite being a **2× larger model** at a **20 %** larger seq length. That's roughly a 2.5× per-parameter throughput speedup.
- **Stronger Arabic at the base.** Qwen 2.5 3 B Instruct's Arabic generation is materially less noisy than 1.5 B's — directly addresses the recurring "AR loops / English-script leakage / mid-sentence truncation" pattern from Experiments 1-3 (`reports/eval/judge_report.md §3`, `judge_report_raft.md §5`).
- **Zero architectural changes.** `train_unsloth.py` reads the same YAML config schema as `train.py`. Swapping engines is a one-line `python` command change.

This unlocks the next operating point on this machine: **3 B QLoRA with a memorisation-shaped recipe, in a single overnight run.**

---

## 6. What comes next

The thesis question is already answered (§5.5–§5.7). What remains is **(a) optional extra-rigor for the write-up** and **(b) optional engineering for a deployable explainer**.

### 6.1 Combined-dataset run — both skills in one set of weights *(the next active experiment)*

The Stage A v2 adapter knows the law but doesn't speak it in our voice. The prior 1.5B house-style runs (Stage 0 / 1) spoke in our voice but didn't know the law. **A single training run on the union of the two datasets** gives both signals to the optimiser simultaneously — no sequential overwrite, no catastrophic forgetting.

| Dataset | Examples | Role |
|---|---|---|
| `data/qa_pairs_knowledge.jsonl` (Stage A) | 21,679 | content + number↔text binding |
| `data/qa_pairs.jsonl` (Stage 1 house-style) | 3,389 | the explanation template + refusals |
| **`data/qa_pairs_combined.jsonl`** (to build) | ~25,068 | both, in one mixture |

Built by [`scripts/build_dataset_combined.py`](../scripts/build_dataset_combined.py) (deterministic concat + dedup + shuffle, no API calls). Trained with [`src/legal_explainer/finetune/configs/qlora_qwen3b_combined.yaml`](../src/legal_explainer/finetune/configs/qlora_qwen3b_combined.yaml) — same Unsloth recipe as the knowledge run (r=16, α=32, LR 1e-4, max_seq 1536, 4 epochs), just 10 % longer at ~17 h on the 3050.

**Hypothesis (and what the success column would have to look like):**

| Dimension | Stage A v2 (knowledge-only) | **Combined (target)** | Stage 1 (style-only) |
|---|---|---|---|
| Closed-book char-sim (smoke test) | **0.884** | ≥ 0.75 (some loss expected from mixed distribution; mild) | ~0.11 (= base) |
| Judge legal accuracy | **3.35** | ≥ 3.0 | 1.00 |
| Judge faithfulness | **3.25** | ≥ 3.0 | 1.00 |
| Judge house-style | 1.05 | **≥ 3.5** (the gain) | 3.55 |
| Judge pass-rate | **8 / 20** | **≥ 10 / 20** | 0 / 21 |
| Refusal accuracy | 0 / 1 | **≥ 1 / 1** (refusals are in `qa_pairs.jsonl`) | 0 / 1 |

If those targets hold, the combined adapter is **the only one in the project where all five judge dimensions are simultaneously in their good range** — and it's the deployable answer.

### 6.2 200-prompt held-out closed-book recall eval *(stretch — extra rigor for the write-up)*

[`scripts/closed_book_recall_eval.py`](../scripts/closed_book_recall_eval.py) is already written. It samples 100 verbatim + 100 reverse-lookup prompts from `qa_pairs_knowledge_val.jsonl` (never seen in training) and grades exact-match + char-similarity vs base. Runtime ~1.7 h per model. Useful as a larger-sample backup of the 21-case judge result, but not strictly required given the judge result already moved off the floor by a wide margin.

### 6.3 7 B / 14 B on Colab *(further stretch)*

Notebooks already exist:
- [`notebooks/qlora_qwen2_5_7b_knowledge_colab.ipynb`](../notebooks/qlora_qwen2_5_7b_knowledge_colab.ipynb) — Qwen 2.5 7B + Unsloth, fits Colab T4
- [`notebooks/qlora_qwen2_5_14b_knowledge_colab.ipynb`](../notebooks/qlora_qwen2_5_14b_knowledge_colab.ipynb) — Qwen 2.5 14B + Unsloth, needs A100

Same dataset and recipe shape. Expected gains: close Article 1 (the lone failing article in the smoke test), push reverse lookups to ≥ 2 / 3, tighten the AR generation. Run these only if/when a specific gap motivates spending the Colab credit.

A separate **product track** (a different workmate's RAG over `orig_data.json`, served at the repo root) is being built in parallel, and the **RAFT-trained adapter** from Experiment 3 will plug into that retriever for the user-facing explainer. That track is not the thesis result; the closed-book numbers from §5.5–§5.7 are.

---

## 7. Quick reference — files

| Topic | Path |
|---|---|
| Corpus | [`data/orig_data.json`](../data/orig_data.json) (1,093 bilingual articles) |
| Baseline dataset (Stage 0) | [`data/qa_pairs.baseline.jsonl`](../data/qa_pairs.baseline.jsonl) |
| Stage-1 dataset | [`data/qa_pairs.jsonl`](../data/qa_pairs.jsonl) |
| RAFT dataset | [`data/qa_pairs_raft.jsonl`](../data/qa_pairs_raft.jsonl) |
| **Knowledge dataset (v2 — current)** | [`data/qa_pairs_knowledge.jsonl`](../data/qa_pairs_knowledge.jsonl) |
| Dataset builders | [`src/legal_explainer/finetune/dataset_builder.py`](../src/legal_explainer/finetune/dataset_builder.py) (house-style + RAFT), [`knowledge_builder.py`](../src/legal_explainer/finetune/knowledge_builder.py) (knowledge) |
| Trainers | [`src/legal_explainer/finetune/train.py`](../src/legal_explainer/finetune/train.py) (vanilla), [`train_unsloth.py`](../src/legal_explainer/finetune/train_unsloth.py) (Unsloth) |
| **Configs** | `qlora_qwen1_5b_local.yaml`, `..._stage1.yaml`, `..._raft.yaml`, `..._knowledge.yaml`, `qlora_qwen3b_knowledge.yaml` (all under [`src/legal_explainer/finetune/configs/`](../src/legal_explainer/finetune/configs/)) |
| Judge eval harness | [`scripts/run_judge_inference.py`](../scripts/run_judge_inference.py) |
| Closed-book smoke test | [`scripts/smoke_test_knowledge_closed_book.py`](../scripts/smoke_test_knowledge_closed_book.py) |
| **Judge reports** | [`reports/eval/judge_report.md`](eval/judge_report.md) (Stage 0), [`judge_report_stage1.md`](eval/judge_report_stage1.md), [`judge_report_raft.md`](eval/judge_report_raft.md) |
| **Smoke-test reports** | [`reports/eval/knowledge_smoke.md`](eval/knowledge_smoke.md) (Stage A v1, 1.5B), [`knowledge_smoke_3b.md`](eval/knowledge_smoke_3b.md) (Stage A v2, 3B) |
| **Judge reports (closed-book)** | Stage 0: [`judge_report.md`](eval/judge_report.md) · Stage 1: [`judge_report_stage1.md`](eval/judge_report_stage1.md) · RAFT: [`judge_report_raft.md`](eval/judge_report_raft.md) · **Stage A v2 (3B): [`judge_report_knowledge_3b.md`](eval/judge_report_knowledge_3b.md) — thesis-grade comparison vs. all prior runs** |
| Per-case prompts/gold/preds | [`reports/eval/cases/`](eval/cases/), [`cases_stage1/`](eval/cases_stage1/), [`cases_raft/`](eval/cases_raft/), [`cases_compare/`](eval/cases_compare/) |
| Stage-A v2 final adapter | [`runs/qlora-qwen2.5-3b-knowledge/`](../runs/qlora-qwen2.5-3b-knowledge/) (114 MB, r=16, best=epoch 4) |
| 7B / 14B Colab notebooks | [`notebooks/qlora_qwen2_5_7b_knowledge_colab.ipynb`](../notebooks/qlora_qwen2_5_7b_knowledge_colab.ipynb), [`qlora_qwen2_5_14b_knowledge_colab.ipynb`](../notebooks/qlora_qwen2_5_14b_knowledge_colab.ipynb) |
| Training report (Stage 0) | [`reports/training/REPORT.md`](training/REPORT.md) |

---

*Document author: in-session Claude (Opus 4.7), 2026-05-13. Reflects the project state at the start of the Stage-A v2 (3 B + Unsloth) training run.*
