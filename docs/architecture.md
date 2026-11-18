# Architecture (R6)

Every box below corresponds to a file in this repository.

```mermaid
flowchart LR
    U["User types a chat message<br/>(Gradio UI · app.py · runs on CPU)"] --> SEL{"Model selector<br/>(default: BERTurk INT8)"}

    subgraph PRE["Preprocessing — Area B · src/normalize.py"]
        N1["Full regex normalization<br/>URL/@user · Turkish lower-case · leetspeak<br/>a.m.k → amk · repeated letters · ASCII folding"]
        N2["Light normalization for BERT<br/>leetspeak · separators · repeated letters (case kept)"]
    end

    SEL --> N1
    SEL --> N2

    N1 --> L["Lexicon + min. edit distance<br/>src/lexicon.py · rules"]
    N1 --> T["TF-IDF (word + char n-grams)<br/>+ Logistic Regression · sklearn FP64"]
    N1 --> W["own word2vec (skip-gram, 100-d)<br/>→ averaged → FFNN 100-128-64-2<br/>SGD + backprop · PyTorch FP32"]
    N2 --> TK["BERTurk WordPiece tokenizer"]
    TK --> B32["BERTurk + classification head<br/>dbmdz/bert-base-turkish-cased · FP32"]
    TK --> B8["BERTurk dynamic quantization<br/>nn.Linear → INT8 · Area D"]

    L --> POST
    T --> POST
    W --> POST
    B32 --> POST
    B8 --> POST

    POST["Post-processing (app.py)<br/>P(OFF) ≥ threshold tuned on val → label + confidence<br/>mask offensive words (lexicon) or hide message<br/>measure latency vs. 50 ms budget"] --> UI["Chat view (masked messages)<br/>+ response-time log side panel"]
```

Training / offline (not on the request path): `train/00_prepare_data.py` → `01_baselines.py` →
`02_word2vec_ffnn.py` → `03_finetune_bert.py` (Colab GPU) → `04_quantize.py` → `05_evaluate.py`
→ `06_push_to_hub.py` (weights to Hugging Face Hub; the app downloads them on first start).
