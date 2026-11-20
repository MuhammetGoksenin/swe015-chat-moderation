"""SWE015 - Real-Time Chat Moderation (Use Case 4). Gradio application.

Every message typed into the chat is labelled live (clean / offensive) with a confidence score by the
selected model (default: quantized BERTurk INT8). Flagged messages are masked. A side panel logs the
response time of every message and lets the user switch between models / compare all of them.

Run:  python app.py      (then open http://127.0.0.1:7860)
"""
import html
import os
import time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the xet downloader can hang on Colab

import gradio as gr
import pandas as pd

from src import config
from src.models import MODEL_KEYS, load_model, load_thresholds, set_cpu_threads

DEFAULT_KEY = "bert_int8"
DISPLAY = {
    "bert_int8": "BERTurk INT8 (quantized) - default",
    "bert_fp32": "BERTurk FP32",
    "w2v_ffnn": "word2vec + FFNN",
    "tfidf": "TF-IDF + LogReg",
    "lexicon": "Lexicon + edit distance (rule)",
}
KEY_OF = {v: k for k, v in DISPLAY.items()}
LOG_COLS = ["#", "model", "label", "P(offensive)", "latency (ms)", f"≤ {config.LATENCY_BUDGET_MS:.0f} ms"]


# ------------------------------------------------------------------ model management
def ensure_models() -> None:
    """Download trained weights from the Hugging Face Hub when models/ is empty (weights are not in git)."""
    if (config.BERT_INT8_DIR / "model_int8.pt").exists():
        return
    if not config.HF_REPO_ID:
        raise SystemExit("models/ is empty. Set HF_REPO_ID=<user>/<repo> (see README) or run the training notebook.")
    from huggingface_hub import snapshot_download
    print(f"Downloading models from https://huggingface.co/{config.HF_REPO_ID} ...")
    snapshot_download(config.HF_REPO_ID, local_dir=str(config.MODELS_DIR))


_cache: dict = {}


def get_model(key):
    if key not in _cache:
        t0 = time.perf_counter()
        _cache[key] = load_model(key)
        print(f"loaded {key} in {time.perf_counter() - t0:.1f}s")
    return _cache[key]


THRESHOLDS: dict = {}


def classify(key, text):
    m = get_model(key)
    t0 = time.perf_counter()
    p = float(m.predict_proba([text])[0])
    ms = (time.perf_counter() - t0) * 1000
    return p, p >= THRESHOLDS.get(key, 0.5), ms


# ------------------------------------------------------------------ rendering
def bubble(msg) -> str:
    off = msg["offensive"]
    color = "#c62828" if off else "#2e7d32"
    label = "OFFENSIVE" if off else "CLEAN"
    conf = msg["p"] if off else 1 - msg["p"]
    if off:
        body = (f"<details><summary style='cursor:pointer'>{html.escape(msg['masked'])}</summary>"
                f"<span style='opacity:.6'>original: {html.escape(msg['text'])}</span></details>")
    else:
        body = html.escape(msg["text"])
    extra = ""
    if msg.get("compare"):
        cells = "".join(
            f"<tr><td>{html.escape(DISPLAY[k].split(' - ')[0])}</td>"
            f"<td style='color:{'#c62828' if o else '#2e7d32'}'>{'OFF' if o else 'NOT'}</td>"
            f"<td>{p:.2f}</td><td>{ms:.1f} ms</td></tr>" for k, p, o, ms in msg["compare"])
        extra = f"<table style='font-size:12px;margin-top:6px'>{cells}</table>"
    return (f"<div style='margin:8px 0;padding:10px 12px;border-radius:12px;border-left:5px solid {color};"
            f"background:rgba(127,127,127,.08)'>"
            f"<div style='font-size:12px;opacity:.8'><b>user{msg['n']}</b> · "
            f"<span style='color:{color};font-weight:700'>{label}</span> {conf:.0%} · "
            f"{html.escape(DISPLAY[msg['key']].split(' - ')[0])} · {msg['ms']:.1f} ms</div>"
            f"<div style='margin-top:4px;font-size:15px'>{body}</div>{extra}</div>")


def render(history) -> str:
    if not history:
        return "<div style='opacity:.6;padding:20px'>No messages yet - type something below.</div>"
    return "<div style='max-height:520px;overflow-y:auto'>" + "".join(bubble(m) for m in reversed(history)) + "</div>"


def stats(log_rows) -> str:
    if not log_rows:
        return ""
    df = pd.DataFrame(log_rows, columns=LOG_COLS)
    lines = ["| model | messages | mean ms | max ms |", "|---|---|---|---|"]
    for name, g in df.groupby("model"):
        lines.append(f"| {name} | {len(g)} | {g['latency (ms)'].mean():.1f} | {g['latency (ms)'].max():.1f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ callbacks
def send(text, model_label, compare, history, log_rows):
    text = (text or "").strip()
    if not text:
        return gr.update(), render(history), history, log_rows, pd.DataFrame(log_rows, columns=LOG_COLS), stats(log_rows)
    key = KEY_OF[model_label]
    n = len(history) + 1
    p, off, ms = classify(key, text)
    masked, n_masked = get_model("lexicon").m.mask(text)
    if off and n_masked == 0:
        masked = "█" * min(len(text), 24) + "  [message hidden]"
    msg = {"n": n, "text": text, "key": key, "p": p, "offensive": off, "ms": ms, "masked": masked}
    rows = [[n, DISPLAY[key].split(" - ")[0], "OFF" if off else "NOT", round(p, 3), round(ms, 2),
             "✓" if ms <= config.LATENCY_BUDGET_MS else "✗"]]
    if compare:
        msg["compare"] = []
        for k in MODEL_KEYS:
            try:
                pk, ok, msk = (p, off, ms) if k == key else classify(k, text)
            except Exception as e:  # noqa: BLE001  (e.g. FP32 weights not downloaded)
                print(f"compare: {k} unavailable ({e})")
                continue
            msg["compare"].append((k, pk, ok, msk))
            if k != key:
                rows.append([n, DISPLAY[k].split(" - ")[0], "OFF" if ok else "NOT", round(pk, 3), round(msk, 2),
                             "✓" if msk <= config.LATENCY_BUDGET_MS else "✗"])
    history = history + [msg]
    log_rows = log_rows + rows
    log_df = pd.DataFrame(list(reversed(log_rows)), columns=LOG_COLS)
    return "", render(history), history, log_rows, log_df, stats(log_rows)


def clear():
    return render([]), [], [], pd.DataFrame(columns=LOG_COLS), ""


EXAMPLES = [
    "Maç çok güzeldi, iyi oynadınız 👏",
    "@USER sen tam bir salaksın",
    "s4l4aaak mısın nesin",
    "Bu takım bu sezon küme düşer bence",
    "a.p.t.a.l herif",
]


def build_ui():
    with gr.Blocks(title="SWE015 · Real-Time Chat Moderation") as demo:
        gr.Markdown("## 🛡️ Real-Time Chat Moderation (Turkish) — SWE015 Use Case 4\n"
                    f"Each message is classified live as **clean / offensive**. Offensive messages are masked "
                    f"(click to reveal). Latency budget: **{config.LATENCY_BUDGET_MS:.0f} ms / message on CPU**.")
        history, log_rows = gr.State([]), gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.HTML(render([]))
                with gr.Row():
                    box = gr.Textbox(placeholder="Type a chat message and press Enter…", show_label=False, scale=5)
                    btn = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=box)
            with gr.Column(scale=2):
                model_dd = gr.Dropdown([DISPLAY[k] for k in MODEL_KEYS[::-1]], value=DISPLAY[DEFAULT_KEY], label="Model")
                compare = gr.Checkbox(label="Compare all models on each message", value=False)
                stats_md = gr.Markdown()
                log = gr.Dataframe(headers=LOG_COLS, label="Response-time log (newest first)", interactive=False)
                gr.Button("Clear chat").click(clear, outputs=[chat, history, log_rows, log, stats_md])
        ins = [box, model_dd, compare, history, log_rows]
        outs = [box, chat, history, log_rows, log, stats_md]
        box.submit(send, ins, outs)
        btn.click(send, ins, outs)
    return demo


if __name__ == "__main__":
    set_cpu_threads()
    ensure_models()
    THRESHOLDS.update(load_thresholds())
    get_model("lexicon")
    get_model(DEFAULT_KEY)            # the application runs the quantized Transformer by default (R5)
    build_ui().launch(share=os.environ.get("GRADIO_SHARE", "0") == "1",
                      server_name=os.environ.get("HOST", "127.0.0.1"))
