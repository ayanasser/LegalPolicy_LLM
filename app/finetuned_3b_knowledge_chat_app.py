"""Gradio chat UI for the Legal Policy explainer — Stage A v2 (our best model).

Model: QLoRA adapter `runs/qlora-qwen2.5-3b-knowledge` over Qwen2.5-3B-Instruct,
the "knowledge-injection" stage that memorised the Egyptian Civil Code. This is a
*closed-book* model: it answers from internalised corpus knowledge, with NO live
retrieval. To stay in-distribution with how it was trained and evaluated
(see scripts/closed_book_recall_eval.py), each turn is sent as a single-turn
user message through the chat template — prior turns are not threaded into the prompt.

Run (use the env that has gradio + the CUDA model stack):
    /home/aya/miniconda3/envs/legalpolicy/bin/python -m app.finetuned_3b_knowledge_chat_app
Optional env vars:
    LP_ADAPTER_DIR   override adapter directory
    LP_SHARE=1       create a public Gradio share link
    LP_PORT=7860     server port
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import gradio as gr
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = PROJECT_ROOT / "runs" / "qlora-qwen2.5-3b-knowledge"
ADAPTER_DIR = Path(os.environ.get("LP_ADAPTER_DIR", str(DEFAULT_ADAPTER)))

# Lazy singleton — load the 4-bit model once, on the first request.
_TOK = None
_MODEL = None
_LOAD_LOCK = threading.Lock()


def _bnb_4bit() -> BitsAndBytesConfig:
    # Identical quantisation to the eval harness so the UI reproduces thesis behaviour.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model():
    """Load tokenizer + adapter once and cache them. Thread-safe."""
    global _TOK, _MODEL
    if _MODEL is not None:
        return _TOK, _MODEL
    with _LOAD_LOCK:
        if _MODEL is None:  # re-check inside the lock
            if not ADAPTER_DIR.exists():
                raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")
            tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(ADAPTER_DIR),
                quantization_config=_bnb_4bit(),
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            _TOK, _MODEL = tok, model
    return _TOK, _MODEL


def respond(message, history, max_new_tokens, temperature, top_p, repetition_penalty):
    """Stream a closed-book answer token-by-token.

    `history` is intentionally ignored: the model is a single-turn recall model.
    """
    message = (message or "").strip()
    if not message:
        yield "Please type a question about the Egyptian Civil Code."
        return

    tok, model = load_model()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    do_sample = float(temperature) > 0.0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        repetition_penalty=float(repetition_penalty),
        pad_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    acc = ""
    for chunk in streamer:
        acc += chunk
        yield acc
    thread.join()


# --- demo extras: real, in-distribution example prompts (EN + AR) -------------
EXAMPLE_PROMPTS = [
    "Give me the English text of Article 376 of the Egyptian Civil Code.",
    "Identify the Egyptian Civil Code article that reads: \"The assurance of the "
    "life of a third party is void unless such third party consents thereto in "
    "writing prior to the issue of the policy.\"",
    "What part of the Egyptian Civil Code is Article 319 in, and what subject does "
    "it deal with?",
    "Summarise where Article 553 of the Egyptian Civil Code belongs and what it "
    "says, as a labelled card.",
    "اذكر نص المادة 551 من القانون المدني المصري حرفياً.",
    "إلى أي قسم من القانون المدني المصري تنتمي المادة 149، وما الموضوع الذي تتناوله؟",
    "Give me Article 644 of the Egyptian Civil Code — Arabic original and English translation.",
]

DESCRIPTION = f"""\
**Stage A v2 — our best model.** QLoRA adapter `{ADAPTER_DIR.name}` over
**Qwen2.5-3B-Instruct**, fine-tuned to memorise the **Egyptian Civil Code**
(EN + AR). This is a **closed-book** demo: answers come from the model's
internalised knowledge — there is **no live retrieval**. Try verbatim recall,
reverse lookup ("which article says…"), placement, or bilingual prompts, in
English or Arabic. The model is single-turn, so each message is answered on its own.
"""


# --- blue look & feel ---------------------------------------------------------
BLUE_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%)",
    block_title_text_color="#1d4ed8",
    button_primary_background_fill="linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%)",
    button_primary_text_color="#ffffff",
)

CUSTOM_CSS = """
#lp-header {
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    color: #ffffff;
    padding: 20px 24px;
    border-radius: 14px;
    margin-bottom: 12px;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);
}
#lp-header h1 { margin: 0; color: #ffffff; font-size: 1.7rem; }
#lp-header p  { margin: 6px 0 0; color: #dbeafe; font-size: 0.95rem; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Legal Policy Chat — Egyptian Civil Code",
        fill_height=True,
    ) as demo:
        gr.HTML(
            "<div id='lp-header'>"
            "<h1>⚖️ Legal Policy Chat — Egyptian Civil Code</h1>"
            "<p>Stage A v2 · closed-book Qwen2.5-3B knowledge adapter · EN / العربية</p>"
            "</div>"
        )
        gr.Markdown(DESCRIPTION)

        with gr.Accordion("Decoding settings", open=False):
            with gr.Row():
                max_new = gr.Slider(
                    64, 1024, value=600, step=8, label="Max new tokens",
                )
                temperature = gr.Slider(
                    0.0, 1.5, value=0.0, step=0.05, label="Temperature (0 = greedy / deterministic)",
                )
            with gr.Row():
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p (sampling only)")
                rep_pen = gr.Slider(1.0, 1.5, value=1.0, step=0.05, label="Repetition penalty")
            gr.Markdown(
                "_Defaults (greedy, rep-penalty 1.0, 600 tokens) reproduce the "
                "closed-book eval that produced the reported numbers._"
            )

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[max_new, temperature, top_p, rep_pen],
            # When additional inputs exist, each example must carry their values too:
            # [message, max_new_tokens, temperature, top_p, repetition_penalty].
            examples=[[p, 600, 0.0, 0.9, 1.0] for p in EXAMPLE_PROMPTS],
            cache_examples=False,
            chatbot=gr.Chatbot(height=460, rtl=False),
            textbox=gr.Textbox(
                placeholder="Ask about an article of the Egyptian Civil Code (English or العربية)…",
                submit_btn=True,
            ),
        )
    return demo


def main():
    demo = build_ui()
    demo.queue()  # serialise requests — only one model copy fits in 6 GB VRAM
    demo.launch(
        theme=BLUE_THEME,
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("LP_PORT", "7860")),
        share=os.environ.get("LP_SHARE", "0") == "1",
    )


if __name__ == "__main__":
    main()
