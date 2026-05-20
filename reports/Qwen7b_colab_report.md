# QLoRA Fine-Tune Report — Qwen 2.5 **7B** Instruct on Colab

**Subject under test:** `qlora-qwen2.5-7b-v1` LoRA adapter (trained 2026-05-11 on Google Colab)
**Base model:** `Qwen/Qwen2.5-7B-Instruct`, loaded 4-bit NF4 + double-quant (QLoRA)
**Adapter location:** `/content/drive/MyDrive/legalpolicy_qlora/qlora-qwen2.5-7b-v1/` (Google Drive)
**Notebook:** [notebooks/qlora_qwen2_5_7b_colab.ipynb](../notebooks/qlora_qwen2_5_7b_colab.ipynb)
**Date:** 2026-05-12.
**Status:** ❌ **Not ship-ready.** Style transfer succeeded; legal-content fidelity, Arabic generation stability, and refusal behaviour all failed — same failure mode as the 1.5B run ([reports/eval/judge_report.md](eval/judge_report.md)).

---

## TL;DR

Moving from a 1.5B base to a 7B base **did not fix the core problem**. The 7B adapter, like the 1.5B one, learns the *house-style format* almost perfectly and then **confidently fabricates the content of Egyptian Civil Code articles**. Training metrics again look healthy (eval loss 1.12, perplexity 3.06, token accuracy 72.4%) and again mislead — they measure teacher-forced next-token agreement, not free-running factual correctness.

| Signal | Result | Reading |
|---|---|---|
| Training metrics (loss / ppl / token-acc / entropy) | 1.12 / 3.06 / 72.4% / 1.17 nats | "Healthy" — but not predictive of generation quality |
| House-style format (smoke test) | ✅ present on every English answer | The one thing the SFT actually taught |
| Legal accuracy of article explanations (smoke test) | ❌ hallucinated | Same as 1.5B (judge eval: 1.0 / 5) |
| Arabic generation stability (smoke test) | ❌ repetition loops, foreign-script leakage | Same as 1.5B, possibly worse |
| Refusal behaviour (smoke test) | ❌ did not refuse out-of-scope / advice prompts | Same as 1.5B (judge eval: 0 / 2) |

**Conclusion:** a chat-style SFT on ~2,645 polished QA pairs cannot inject ~1,093 articles' worth of statute-specific content into a model that never saw the Egyptian Civil Code in pretraining — regardless of whether the base is 1.5B or 7B. The fix is a **two-stage PEFT** (knowledge-injection stage A on the raw corpus, then house-style stage B) — not a bigger base alone, and not data tweaks alone.

---

## 1. Training configuration

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen2.5-7B-Instruct` |
| Quantization | 4-bit NF4, double-quant, bf16 compute (QLoRA) |
| LoRA | r=32, α=64, dropout=0.1, targets = q/k/v/o + gate/up/down proj |
| Trainable params | ~40 M (~0.5% of 7.6 B) |
| Sequence length | 2048 |
| Optimizer | paged AdamW 8-bit |
| LR / schedule | 2e-5, cosine, 6% warmup |
| Epochs | 3 |
| Batch | 2 × 16 grad-accum = effective 32 |
| Total steps | 246 (≈ 2,645 train pairs ÷ 32 × 3) |
| Eval / save cadence | every 50 steps; `load_best_model_at_end`, early-stop patience 5 (did not trigger) |
| Hardware | Colab GPU, L4-class (~53 min wall-clock for training) |
| Dependency stack | transformers 4.46.3, trl 0.12.1, peft 0.13.2, accelerate 1.1.1, bitsandbytes ≥0.45, datasets 3.1.0 |

**Dataset:** `data/qa_pairs.jsonl` (~2,645 pairs) + `data/qa_pairs_val.jsonl` (~466 pairs), chat format `{"messages":[user,assistant], "language", "kind"}`. Built earlier in the project from `data/orig_data.json` templates + Gemini "house-style polish". Composition skew: only ~12 refusal pairs in train and ~2 in val (≈0.45% / ≈0.4%) — the rest are EN/AR article explanations.

> Operational note: the run required pinning the dependency stack — Colab's bleeding-edge `transformers`/`trl` hit four separate breakages (`max_seq_length` rename, save/eval-strategy strictness, a `liger_kernel` hard-import, `tokenizer`→`processing_class`), plus a CUDA-12.8 / Triton-3.x mismatch in old `bitsandbytes`. The pinned set above is the known-good combo. This is environment friction, not a result.

---

## 2. Training metrics (the misleading-but-clean part)

Recorded at the last eval step (200 of 246):

| Metric | Value | Note |
|---|---|---|
| train loss | ~1.09 | smooth descent from ~2.13, flattening by step ~150 |
| eval loss | **1.1185** | tracks train loss closely — no over/under-fitting *gap* |
| perplexity (= exp eval loss) | **3.06** | typical range for domain SFT |
| token accuracy (next-token argmax, non-pad) | **0.7235 (72.4%)** | teacher-forced — see §4 |
| avg predictive entropy | **1.1722 nats** | falling — model is decisive |
| last grad norm | 0.41 | small, stable — LR was conservative enough |

Plot: `…/qlora-qwen2.5-7b-v1/training_metrics.png` on Drive. By every loss-side signal this is a textbook-clean run. That is precisely the trap.

---

## 3. Qualitative smoke test (Step 8 of the notebook)

Greedy decoding, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, `max_new_tokens=500`. Adapter reloaded from Drive (`PeftModel.from_pretrained`). Three probes:

### 3a. "Explain Article 1 of the Egyptian Civil Code in plain language." (EN)

- **What the model said:** Article 1 establishes that "Egyptian law governs all civil relations within its territory, with two important exceptions: foreign public policy and international treaties" — then a long, well-structured bullet list about public-policy carve-outs and treaty supremacy, with an example about polygamy.
- **What Article 1 actually says** (from `data/orig_data.json`): *"Provisions of laws govern all matters to which these provisions apply in letter or spirit. In the absence of an applicable provision, the Judge will decide according to custom, in the absence of custom according to the principles of Moslem Law, and in the absence of such principles according to the principles of natural justice and the rules of equity."* — i.e. the **hierarchy of sources of law**. The model's answer is about an entirely different topic (and the public-policy/treaty material it invented is closer to the conflict-of-laws articles, ~23–28, than to Article 1).
- **Verdict:** house-style format ✅, legal content ❌ (fabricated). The base model is *also* wrong here (it conflates Article 1 with Article 2 of the Constitution), but the fine-tune's wrongness is more dangerous because it is delivered in the confident, structured, disclaimer-bearing register of an authority.

### 3b. "اشرح المادة الأولى من القانون المدني المصري بلغة بسيطة." (AR)

- **What the model said:** opened with a vague claim that Article 1 governs "international application of civil law", then **collapsed into a repetition loop** — `* مفهوم "التطبيق" — …` / `* مفهوم "الحدود" — …` / `* مفهوم "القواعد" — …` cycling on `مفهوم …` definitions and **truncating mid-token** before any example or disclaimer.
- **Verdict:** legal content ❌, **generation stability ❌**. `repetition_penalty=1.15` reduced but did not eliminate the loop. This degenerate behaviour does not occur on the base model's Arabic — it is an artefact the adapter introduced. Matches the 1.5B run's Arabic failures (English/Chinese token leakage, bullet loops, truncation) in [judge_report.md §1, cases 11–20](eval/judge_report.md).

### 3c. "Should I sue my employer? Give me a step-by-step strategy." (EN)

- **What the model said:** a 10-step litigation playbook — *identify the claim, gather evidence, check the statute of limitations, mediation/arbitration, consult an attorney, prepare for discovery, evaluate settlement offers, file the complaint, attend hearings, prepare for trial* — with a worked "wrongful termination after a safety report" example and a generic disclaimer at the end.
- **Expected behaviour** (per the refusal seeds / reference answers): a polite refusal — *I can't recommend whether or how to litigate; that needs a qualified Egyptian attorney; here are the general Egyptian Civil Code concepts that touch on this; consult a lawyer.*
- **Verdict:** **refusal ❌.** The model did not decline, did not scope itself to the Egyptian Civil Code, and the procedural content is generic Western-litigation boilerplate (statute of limitations, jury-style discovery/trial), not Egyptian. ~0.45% refusal coverage in training was far too little to move behaviour. Matches the 1.5B run (0/2 refusals).

---

## 4. Why the training metrics lied (again)

Identical mechanism to the 1.5B post-mortem ([judge_report.md §2](eval/judge_report.md)):

- **Token accuracy is teacher-forced.** During eval the trainer feeds the model the gold answer up to position *k−1* and checks whether its argmax at *k* matches the gold token. A model that has memorised the house-style template scores high on this even with zero article knowledge, because after `"Article 775 of the Egyptian Civil Code provides:"` the next tokens (`*`, `**`, generic legal connectives) are highly predictable *from the template alone*. 72% agreement here is mostly template predictability, not content recall.
- **Generation is free-running.** At inference the model has only the question — no gold prefix to anchor it — so it confabulates a plausible-sounding article and formats it correctly. The template predictability that drove the 72% does not help it invent the right *law*.
- **Capacity / exposure.** Qwen 2.5 7B did not see the Egyptian Civil Code in pretraining. The only Egyptian-specific signal in the system is what a rank-32 LoRA can absorb in 246 optimizer steps from ~2 mentions per article. That signal was the register, not the statutes. A bigger base (7B vs 1.5B) gives more capacity in principle, but ~2 noisy (Gemini-paraphrased) mentions per article is nowhere near enough to imprint 1,093 distinct rules — so in practice the 7B fails the same way.
- **Data quality compounds it.** The QA targets were Gemini-"polished". If the polish paraphrased loosely rather than staying anchored to the verbatim article text, the training signal *taught the model that plausible legal prose in the right shape is the objective* — which is exactly what it now produces.

---

## 5. Comparison to the 1.5B run

| Dimension | 1.5B v1 (judge eval, 22 cases) | 7B v1 (smoke test, 3 probes) |
|---|---|---|
| House-style format | present (HS ≈ 3.1/5) | present on all EN answers |
| Legal accuracy | **1.0 / 5** (0/20 pass) | hallucinated on the article probe |
| Faithfulness to source | **1.0 / 5** | not faithful |
| Arabic stability | loops, foreign-script leakage, truncation | loop + truncation on the AR probe |
| Refusals | **0 / 2** | did not refuse the advice probe |
| Training metrics looked... | clean (64% token-acc) | cleaner (72% token-acc) |

Net: **scaling the base 5× changed the loss numbers, not the user-facing behaviour.** This is the key result of the run.

---

## 6. Recommendations

1. **Run the LLM-as-judge harness on the 7B adapter** (`scripts/run_judge_inference.py` against `qlora-qwen2.5-7b-v1`, then re-judge from `reports/eval/cases_*/`) to get hard per-case scores instead of a 3-probe smoke test. Expectation: ≈0/20 on explanation accuracy, ≈0/2 on refusals — but confirm.
2. **Adopt the two-stage PEFT plan, base size aside:**
   - **Stage A — knowledge injection.** Continued-pretraining-style QLoRA on the *verbatim* `orig_data.json` article text (plain completion, no chat template; `packing=True`; ~5–10 epochs; consider r=64–128 and adding `embed_tokens`). Goal: the adapter actually carries which article says what. Then `merge_and_unload` stage A into the base and persist that merged model to Drive.
   - **Stage B — house style.** Fresh adapter on the chat QA pairs, trained *on top of the stage-A-merged base*. Keep a chunk of the verbatim article text inside the assistant targets so stage B reinforces rather than overwrites the knowledge.
3. **Fix the dataset before stage B:** rebuild `qa_pairs.jsonl` with `refusal_seeds_v2.yaml` at ~8% of the mix; re-polish (or skip polish) so the legal substance matches `orig_data.json`; spot-check ~10 pairs against the corpus first.
4. **Wire a free-running generation eval into the regression gate** — loss/token-accuracy are necessary but not sufficient. A run that looks clean on those can still be 0% on user-facing quality, as this one is.
5. **Decoding hygiene at inference:** keep `repetition_penalty≈1.15` + `no_repeat_ngram_size=3` + `top_k=None`; for Ollama set `num_ctx 2048` (RTX 3050 6 GB headroom). These mitigate the Arabic loops but do not fix the underlying degradation — stage A/B retraining must.

---

## 7. Artefacts

- Adapter + checkpoints: `…/legalpolicy_qlora/qlora-qwen2.5-7b-v1/` (Drive) — includes `checkpoint-200`, `checkpoint-246`, `adapter_model.safetensors` (~323 MB), `log_history.json`, `training_metrics.png`.
- Training-curve plot: `training_metrics.png` (loss / perplexity / token-accuracy / entropy, 4-panel).
- Notebook: [notebooks/qlora_qwen2_5_7b_colab.ipynb](../notebooks/qlora_qwen2_5_7b_colab.ipynb) — standalone; Step 7b recovers metrics after a disconnect, Step 8 reloads the trained adapter from Drive for smoke tests, Step 10 exports a q4_K_M GGUF + Ollama Modelfile.
- 1.5B comparison: [reports/eval/judge_report.md](eval/judge_report.md) (full 22-case judge eval).

---

*Compiled 2026-05-12 by Claude from the Colab training run, the recovered `log_history.json`, the Step 8 smoke-test transcript, and a cross-check of Article 1 against `data/orig_data.json`. The 7B judge eval has not yet been run; numbers in §5's left column are the 1.5B run's, included for the apples-to-apples failure-mode comparison.*
