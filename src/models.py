"""One common interface for the whole model ladder.

Every model exposes  predict_proba(list[str]) -> np.ndarray of P(offensive)  and  size_mb().
The same classes are used by the evaluation script and by the Gradio app, so the numbers in the
results table come from exactly the code that runs in the application.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config
from .normalize import normalize_for_bert, tokenize


def _mb(*paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".bin", ".pt"})
        elif p.exists():
            total += p.stat().st_size
    return total / 1e6


# --------------------------------------------------------------------------- B: rule baseline
class LexiconModel:
    name = "Lexicon + edit distance (rule)"
    precision = "rules"

    def __init__(self):
        from .lexicon import LexiconMatcher
        self.m = LexiconMatcher.load(config.LEXICON_PATH)

    def predict_proba(self, texts):
        return np.asarray(self.m.predict_proba(texts))

    def size_mb(self):
        return _mb(config.LEXICON_PATH)


# --------------------------------------------------------------------------- B: tf-idf baseline
class TfidfModel:
    name = "TF-IDF + LogReg"
    precision = "FP64 (sklearn)"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(config.TFIDF_PATH)

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))[:, 1]

    def size_mb(self):
        return _mb(config.TFIDF_PATH)


# --------------------------------------------------------------------------- A: word2vec + FFNN
class FFNN(nn.Module):
    """Feed-forward network on averaged word2vec vectors: d -> 128 -> 64 -> 2."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        # input standardization statistics (computed on the training set, saved with the weights)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def average_embeddings(kv, texts) -> np.ndarray:
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(t) if w in kv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


class W2VFFNNModel:
    name = "word2vec + FFNN"
    precision = "FP32"

    def __init__(self):
        from gensim.models import KeyedVectors
        self.kv = KeyedVectors.load(str(config.W2V_PATH))
        self.net = FFNN(self.kv.vector_size)
        self.net.load_state_dict(torch.load(config.FFNN_PATH, map_location="cpu"))
        self.net.eval()

    @torch.inference_mode()
    def predict_proba(self, texts):
        x = torch.from_numpy(average_embeddings(self.kv, list(texts)))
        return torch.softmax(self.net(x), dim=-1)[:, 1].numpy()

    def size_mb(self):
        return _mb(config.W2V_PATH, str(config.W2V_PATH) + ".vectors.npy", config.FFNN_PATH)


# --------------------------------------------------------------------------- C / D: BERTurk
def quantize_int8(model: nn.Module) -> nn.Module:
    """Area D - post-training dynamic INT8 quantization of all nn.Linear layers (CPU)."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:  # older torch
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


class BertModel:
    def __init__(self, int8: bool = False):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.int8 = int8
        self.dir = config.BERT_INT8_DIR if int8 else config.BERT_FP32_DIR
        self.name = "BERTurk INT8 (dynamic quant.)" if int8 else "BERTurk FP32"
        self.precision = "INT8 linear / FP32 embeddings" if int8 else "FP32"
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        if int8:
            cfg = AutoConfig.from_pretrained(self.dir)
            model = AutoModelForSequenceClassification.from_config(cfg)
            model = quantize_int8(model.eval())
            state = torch.load(self.dir / "model_int8.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.dir)
        self.model = model.to("cpu").eval()

    @torch.inference_mode()
    def predict_proba(self, texts, batch_size: int = 32):
        probs = []
        texts = [normalize_for_bert(t) for t in texts]
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=config.BERT_MAX_LEN,
                           padding=True, return_tensors="pt")
            probs.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].numpy())
        return np.concatenate(probs)

    def size_mb(self):
        return _mb(self.dir)


# --------------------------------------------------------------------------- registry
MODEL_KEYS = ["lexicon", "tfidf", "w2v_ffnn", "bert_fp32", "bert_int8"]


def load_model(key: str):
    return {
        "lexicon": LexiconModel,
        "tfidf": TfidfModel,
        "w2v_ffnn": W2VFFNNModel,
        "bert_fp32": lambda: BertModel(int8=False),
        "bert_int8": lambda: BertModel(int8=True),
    }[key]()


def load_thresholds() -> dict:
    if config.THRESHOLDS_PATH.exists():
        return json.loads(config.THRESHOLDS_PATH.read_text())
    return {}


def set_cpu_threads():
    n = int(os.environ.get("TORCH_THREADS", 0))
    if n > 0:
        torch.set_num_threads(n)
