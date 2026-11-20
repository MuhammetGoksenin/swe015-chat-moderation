# SWE015 · Real-Time Chat Moderation (Use Case 4)

A Turkish chat-moderation application for a gaming platform: every message is labelled live as
**clean / offensive** with a confidence score, offensive messages are masked, and a side panel logs the
response time of each message against a **50 ms / message CPU budget**. The app runs a **dynamically
quantized (INT8) BERTurk** by default and lets the user switch to / compare with every other model.

Author: Muhammet Çelik (2509011060) · SWE015 Introduction to Large Language Models · İstinye University

## Model ladder (all evaluated on the same held-out test set)

| # | Model | Area | Code |
|---|---|---|---|
| 1a | Lexicon + minimum edit distance (rule baseline) | B | `src/lexicon.py`, `train/01_baselines.py` |
| 1b | TF-IDF (word 1-2 + char 2-5 grams) + logistic regression | B | `train/01_baselines.py` |
| 2 | Own word2vec (trained on the training messages) → averaged → feed-forward net, SGD + backprop | A | `train/02_word2vec_ffnn.py` |
| 3 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned with a classification head | C | `train/03_finetune_bert.py` |
| 4 | BERTurk with post-training dynamic INT8 quantization (all `nn.Linear`) | D | `train/04_quantize.py` |

Regex normalization of noisy text (leetspeak, separated/repeated letters, Turkish characters) is in
`src/normalize.py`. Architecture diagram: [`docs/architecture.md`](docs/architecture.md).

## Results

Official OffensEval-TR 2020 test split, n = 3,528 (20.3 % OFF). Produced by `train/05_evaluate.py`
(full output in [`results/results.md`](results/results.md)).

| Model | Precision | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean / p95 (ms) | ≤ 50 ms |
|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | rules | 0.700 | 0.808 | 0.513 | 0.0 | 0.2 / 0.5 | ✓ |
| TF-IDF + LogReg | FP64 (sklearn) | 0.770 | 0.861 | 0.568 | 2.3 | 2.3 / 3.1 | ✓ |
| word2vec + FFNN | FP32 | 0.695 | 0.801 | 0.524 | 12.7 | 0.4 / 0.6 | ✓ |
| BERTurk FP32 | FP32 | 0.809 | 0.884 | 0.640 | 442.5 | 134.3 / 207.7 | ✗ |
| **BERTurk INT8 (dynamic quant.)** | INT8 linear / FP32 embeddings | **0.809** | 0.882 | 0.648 | **186.0** | **59.1 / 119.7** | ✗ |

Latency: batch size 1, preprocessing + tokenization included, mean over 200 test messages after 10 warm-up
runs. Hardware: Google Colab, Intel Xeon CPU @ 2.00 GHz, 2 logical CPUs, torch threads = 1, CPU only.
Size = serialized model weights (tokenizer/config excluded).

Quantization kept macro-F1 unchanged (0.809), made the model 2.4× smaller and 2.3× faster, but the mean
latency stays slightly above the 50 ms budget on the shared Colab CPU. Repeated INT8 runs on the same
hardware measured 51.1, 57.0 (max length 64) and 70.4 ms (2 threads), so Colab CPU latency varies by
roughly ±10–20 % between runs.

## Run the application (single command)

```bash
git clone https://github.com/MuhammetGoksenin/swe015-chat-moderation.git
cd swe015-chat-moderation
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On first start the trained weights (~0.6 GB) are downloaded from the
Hugging Face Hub ([GokseninMuhammet/swe015-chat-moderation-models](https://huggingface.co/GokseninMuhammet/swe015-chat-moderation-models))
into `models/` — weights are not stored in git. Set `GRADIO_SHARE=1` for a public link
(e.g. on Colab).

## Reproduce training (Google Colab, T4 GPU)

Open [`notebooks/SWE015_train_colab.ipynb`](notebooks/SWE015_train_colab.ipynb) and run all cells, or:

```bash
python train/00_prepare_data.py     # download OffensEval-TR 2020, make train/val/test splits
python train/01_baselines.py        # lexicon + edit distance, TF-IDF + LR
python train/02_word2vec_ffnn.py    # word2vec + FFNN (SGD)
python train/03_finetune_bert.py    # BERTurk fine-tuning (GPU recommended)
python train/04_quantize.py         # dynamic INT8 quantization
python train/05_evaluate.py         # results table on the test set (CPU)
HF_REPO_ID=<your-hf-user>/swe015-chat-moderation-models python train/06_push_to_hub.py   # only to re-upload
```

`bash tests/smoke_test.sh` runs the whole pipeline on synthetic data with a tiny random BERT (code check
only — its numbers are meaningless).

## Data

**OffensEval-TR 2020** — Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.*
LREC 2020. https://coltekin.github.io/offensive-turkish/ (CC BY 2.0).
Splits: official training set → 90 % train / 10 % validation (stratified, seed 42); official test set
(3,528 tweets) is the held-out test set for every model. Validation is used only for hyper-parameters
and decision thresholds.

Pretrained model: BERTurk — Schweter, S. (2020), `dbmdz/bert-base-turkish-cased`, Hugging Face Hub.

## AI usage

AI assistants (Claude) were used for code scaffolding and debugging; see the AI-usage slide.
