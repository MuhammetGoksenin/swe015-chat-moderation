"""Central configuration. Every path/setting can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))          # raw + processed data
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))    # trained models (NOT in git)
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

# Pretrained Transformer (BERTurk, cased). Override e.g. for a quick test with a tiny model.
BERT_MODEL = os.environ.get("BERT_MODEL", "dbmdz/bert-base-turkish-cased")
BERT_MAX_LEN = int(os.environ.get("BERT_MAX_LEN", 96))
BERT_EPOCHS = int(os.environ.get("BERT_EPOCHS", 3))
BERT_BATCH = int(os.environ.get("BERT_BATCH", 32))
BERT_LR = float(os.environ.get("BERT_LR", 2e-5))

# Hugging Face Hub repo that hosts the trained weights (the app downloads from here if models/ is empty)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "GokseninMuhammet/swe015-chat-moderation-models")

SEED = 42
VAL_FRACTION = 0.10           # validation split carved out of the official training set
LATENCY_BUDGET_MS = float(os.environ.get("LATENCY_BUDGET_MS", 50))
LATENCY_RUNS = int(os.environ.get("LATENCY_RUNS", 200))   # >= 100 as required by R4

LABELS = ["NOT", "OFF"]       # 0 = clean, 1 = offensive

# Sub-folders / files inside MODELS_DIR
LEXICON_PATH = MODELS_DIR / "lexicon.json"
TFIDF_PATH = MODELS_DIR / "tfidf_lr.joblib"
W2V_PATH = MODELS_DIR / "word2vec.kv"
FFNN_PATH = MODELS_DIR / "w2v_ffnn.pt"
BERT_FP32_DIR = MODELS_DIR / "bert_fp32"
BERT_INT8_DIR = MODELS_DIR / "bert_int8"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
