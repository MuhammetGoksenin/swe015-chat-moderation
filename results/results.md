| Model | Precision | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean / p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rule) | rules | 0.700 | 0.808 | 0.513 | 0.0 | 0.2 / 0.5 | ✓ |
| TF-IDF + LogReg | FP64 (sklearn) | 0.770 | 0.861 | 0.568 | 2.3 | 2.3 / 3.1 | ✓ |
| word2vec + FFNN | FP32 | 0.695 | 0.801 | 0.524 | 12.7 | 0.4 / 0.6 | ✓ |
| BERTurk FP32 | FP32 | 0.809 | 0.884 | 0.640 | 442.5 | 134.3 / 207.7 | ✗ |
| BERTurk INT8 (dynamic quant.) | INT8 linear / FP32 embeddings | 0.809 | 0.882 | 0.648 | 186.0 | 59.1 / 119.7 | ✗ |

Test set: OffensEval-TR 2020 official test split, n = 3528 (20.3% OFF).
Latency: batch size 1, preprocessing + tokenization included, mean over 200 test messages after 10 warm-up runs. Budget = 50 ms/message.
Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz | 2 logical CPUs | torch threads=1 | CPU only.
Size = serialized model weights on disk (tokenizer/config files excluded).