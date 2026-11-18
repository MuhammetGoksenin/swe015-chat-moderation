"""Step 4 - Area D: post-training dynamic INT8 quantization of the fine-tuned BERTurk (CPU inference).

All nn.Linear layers (attention Q/K/V/output, feed-forward, classifier) get INT8 weights; activations are
quantized on the fly. Embeddings and LayerNorm stay FP32. The decision threshold is re-tuned on val.
"""
import shutil

import _common
import torch
from transformers import AutoModelForSequenceClassification

from src import config
from src.data import load_split
from src.metrics import save_threshold, scores, tune_threshold
from src.models import BertModel, quantize_int8

model = AutoModelForSequenceClassification.from_pretrained(config.BERT_FP32_DIR).to("cpu").eval()
qmodel = quantize_int8(model)
out = config.BERT_INT8_DIR
out.mkdir(parents=True, exist_ok=True)
torch.save(qmodel.state_dict(), out / "model_int8.pt")
for f in config.BERT_FP32_DIR.iterdir():          # config + tokenizer files (no weights)
    if f.suffix not in {".safetensors", ".bin"}:
        shutil.copy(f, out / f.name)
print("saved", out)

va = load_split("val")
fp32, int8 = BertModel(int8=False), BertModel(int8=True)
p32, p8 = fp32.predict_proba(list(va.text)), int8.predict_proba(list(va.text))
thr = tune_threshold(va.label, p8)
save_threshold("bert_int8", thr)
print(f"size  FP32={fp32.size_mb():.1f} MB  INT8={int8.size_mb():.1f} MB")
print(f"val macro-F1  FP32={scores(va.label, p32)['macro_f1']:.4f}  INT8={scores(va.label, p8)['macro_f1']:.4f}")
print(f"prediction agreement FP32 vs INT8 (thr 0.5): {((p32 >= 0.5) == (p8 >= 0.5)).mean():.2%}")
print(f"INT8 threshold={thr:.2f}")
