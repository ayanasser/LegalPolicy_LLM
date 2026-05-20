"""Pull the QLoRA 7B adapter from the HuggingFace Hub and regenerate predictions
on the SAME 22 cases the 1.5B run was judged on, so the two are directly comparable.

What it does
------------
1. Logs in to HF (token via --hf-token, or HF_TOKEN env var, or `huggingface-cli login`).
2. Downloads the adapter repo (default: AyaNasser/legalpolicy-qwen2.5-7b-qlora) and the
   base model it points at (Qwen/Qwen2.5-7B-Instruct), loading the base 4-bit (QLoRA).
3. Reads the existing reports/eval/judge_predictions.json for the case list
   (id / language / kind / article_key / prompt / reference) — does NOT touch it.
4. Re-generates `prediction` for each case with greedy decoding + repetition_penalty
   (the settings that curb the Arabic loops), writing:
       reports/eval/judge_predictions_7b.json   (incremental)
       reports/eval/cases_7b/case_NN.md         (one markdown file per case)
5. A separate LLM-as-judge pass (Claude, in-session) then reads judge_predictions_7b.json
   and writes reports/eval/judge_report_7b.md.

Run it in Colab (L4/A100) — a 7B in 4-bit won't run usefully on a 6 GB laptop GPU.

Colab usage:
    !git clone https://github.com/<you>/LegalPolicy_LLM /content/repo   # or just copy this file + reports/eval/judge_predictions.json
    %pip install -q "transformers==4.46.3" "peft==0.13.2" "bitsandbytes>=0.45.0" "accelerate==1.1.1" "huggingface_hub>=0.25"
    !python /content/repo/scripts/eval_7b_from_hub.py --hf-token $HF_TOKEN

Local usage (only if you have a >=16 GB GPU):
    HF_TOKEN=hf_xxx python scripts/eval_7b_from_hub.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "AyaNasser/legalpolicy-qwen2.5-7b-qlora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CASES_JSON = PROJECT_ROOT / "reports" / "eval" / "judge_predictions.json"
DEFAULT_OUT_JSON = PROJECT_ROOT / "reports" / "eval" / "judge_predictions_7b.json"
DEFAULT_CASES_DIR = PROJECT_ROOT / "reports" / "eval" / "cases_7b"


def hf_login(token: str | None) -> str | None:
    """Log in to HF if a token is available; return the resolved username (or None)."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        token = token.strip()
    from huggingface_hub import login, whoami
    if token:
        login(token=token, add_to_git_credential=False)
    try:
        info = whoami(token=token)
        role = info.get("auth", {}).get("accessToken", {}).get("role")
        print(f"HF auth: user={info['name']}  token_role={role}")
        return info["name"]
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: HF whoami failed ({e}). If the adapter repo is private this will fail to download.")
        return None


def load_model(hf_repo: str, base_model: str | None, token: str | None):
    """Load base model 4-bit + the LoRA adapter from the Hub. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel

    token = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip() or None

    print(f"Reading adapter config from {hf_repo} ...")
    peft_cfg = PeftConfig.from_pretrained(hf_repo, token=token)
    base_id = base_model or peft_cfg.base_model_name_or_path or DEFAULT_BASE_MODEL
    print(f"Base model: {base_id}")

    tok = AutoTokenizer.from_pretrained(hf_repo, token=token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading base model (4-bit) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print("Attaching LoRA adapter ...")
    model = PeftModel.from_pretrained(base, hf_repo, token=token)
    model.eval()
    return model, tok


def generate(model, tok, user_msg: str, max_new_tokens: int) -> tuple[str, float]:
    import torch
    chat = [{"role": "user", "content": user_msg}]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                       # greedy — deterministic eval
            temperature=None, top_p=None, top_k=None,
            repetition_penalty=1.15,               # curbs the Arabic bullet-loops
            no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = tok.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()
    return text, elapsed


def write_case_md(cases_dir: Path, rec: dict) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    p = cases_dir / f"case_{rec['id']:02d}.md"
    body = (
        f"# Case {rec['id']:02d} — {rec['language'].upper()} / {rec['kind']}"
        f"{' / ' + rec['article_key'] if rec.get('article_key') else ''}\n\n"
        f"## Prompt\n\n{rec['prompt']}\n\n"
        f"## Reference (gold)\n\n{rec['reference']}\n\n"
        f"## Prediction — qlora-qwen2.5-7b-v1 (from {rec.get('source_repo', '?')})\n\n"
        f"{rec['prediction']}\n\n"
        f"---\n*gen_seconds: {rec['gen_seconds']}*\n"
    )
    p.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="Adapter repo on the Hub.")
    ap.add_argument("--base-model", default=None, help="Override base model id (default: from adapter_config).")
    ap.add_argument("--hf-token", default=None, help="HF token (else HF_TOKEN env var, else prior `huggingface-cli login`).")
    ap.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON,
                    help="Existing predictions JSON to take the case list (prompt/reference) from.")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Where to write the 7B predictions.")
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR, help="Where to write per-case markdown.")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only do the first N cases (smoke test).")
    args = ap.parse_args()

    if not args.cases_json.exists():
        sys.exit(f"Case list not found: {args.cases_json}. Run scripts/run_judge_inference.py first, "
                 f"or point --cases-json at an existing predictions file.")
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Cases to evaluate: {len(cases)} "
          f"(EN={sum(c['language']=='en' and c['kind']=='explanation' for c in cases)}, "
          f"AR={sum(c['language']=='ar' and c['kind']=='explanation' for c in cases)}, "
          f"refusals={sum(c['kind']=='refusal' for c in cases)})")

    username = hf_login(args.hf_token)
    if username and "/" in args.hf_repo and not args.hf_repo.lower().startswith(username.lower() + "/"):
        print(f"NOTE: adapter repo namespace ({args.hf_repo.split('/')[0]}) != your username ({username}). "
              f"That's fine if it's public or shared with you; just flagging it.")

    model, tok = load_model(args.hf_repo, args.base_model, args.hf_token)
    print("Model ready. Generating ...\n")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for i, c in enumerate(cases, 1):
        prompt = c["prompt"]
        pred, secs = generate(model, tok, prompt, args.max_new_tokens)
        rec = {
            "id": c["id"],
            "language": c["language"],
            "kind": c["kind"],
            "article_key": c.get("article_key"),
            "prompt": prompt,
            "reference": c["reference"],
            "prediction": pred,
            "gen_seconds": round(secs, 2),
            "source_repo": args.hf_repo,
            "model": "qlora-qwen2.5-7b-v1",
            "decoding": "greedy, repetition_penalty=1.15, no_repeat_ngram_size=3",
        }
        out.append(rec)
        write_case_md(args.cases_dir, rec)
        # incremental save
        args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(cases)}] id={c['id']} lang={c['language']} kind={c['kind']} "
              f"art={c.get('article_key')} time={secs:.1f}s chars={len(pred)}")

    print(f"\nWrote {len(out)} predictions -> {args.out_json}")
    print(f"Per-case markdown -> {args.cases_dir}/case_*.md")
    print("\nNext: hand judge_predictions_7b.json (and/or cases_7b/) to the LLM judge "
          "to produce reports/eval/judge_report_7b.md.")


if __name__ == "__main__":
    main()
