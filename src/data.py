"""Download and prepare OffensEval-TR 2020 (Çöltekin, 2020).

Splits used by EVERY model (same held-out test set, requirement R3):
  train : official training set minus a stratified 10 % validation part
  val   : 10 % of the official training set (threshold / hyper-parameter tuning only)
  test  : official OffensEval-TR 2020 test set (3,528 tweets) -- never used for training/tuning
"""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from . import config

ZIP_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
HF_FALLBACK_REPO = "strombergnlp/offenseval_2020"   # also contains the Turkish files


def _find(raw_dir: Path, *needles):
    for p in raw_dir.rglob("*"):
        name = p.name.lower()
        if p.is_file() and all(n in name for n in needles):
            return p
    return None


def download_raw(raw_dir: Path = config.RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _find(raw_dir, "tr", "train"):
        print(f"Raw files already in {raw_dir}")
        return
    try:
        print(f"Downloading {ZIP_URL}")
        r = requests.get(ZIP_URL, timeout=60)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(raw_dir)
        return
    except Exception as e:  # noqa: BLE001
        print(f"Primary download failed ({e}); trying Hugging Face Hub mirror {HF_FALLBACK_REPO}")
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(HF_FALLBACK_REPO, repo_type="dataset") if "-tr-" in f]
    for f in files:
        hf_hub_download(HF_FALLBACK_REPO, f, repo_type="dataset", local_dir=raw_dir)
    if not files:
        raise RuntimeError("Could not download the dataset. Put the OffensEval-TR files into " + str(raw_dir))


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def _read_labels(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    sep = "\t" if "\t" in text[0] else ","
    rows = [line.split(sep)[:2] for line in text if line.strip()]
    if not rows[0][0].strip().isdigit():          # header line
        rows = rows[1:]
    return pd.DataFrame(rows, columns=["id", "label"])


def load_raw(raw_dir: Path = config.RAW_DIR):
    train_f = _find(raw_dir, "tr", "train", ".tsv")
    test_f = _find(raw_dir, "tr", "test", ".tsv") or _find(raw_dir, "tr", "test")
    lab_f = _find(raw_dir, "tr", "labela")
    if not (train_f and test_f and lab_f):
        raise FileNotFoundError(f"Missing raw files in {raw_dir}: train={train_f}, test={test_f}, labels={lab_f}")
    tr = _read_tsv(train_f)
    tr = tr.rename(columns={tr.columns[1]: "text", tr.columns[2]: "label"})[["id", "text", "label"]]
    te = _read_tsv(test_f)
    te = te.rename(columns={te.columns[1]: "text"})[["id", "text"]]
    labels = _read_labels(lab_f)
    te = te.merge(labels, on="id", how="inner")
    for df in (tr, te):
        df["label"] = df["label"].str.strip().map({"NOT": 0, "OFF": 1})
    tr = tr.dropna(subset=["label"]).astype({"label": int})
    te = te.dropna(subset=["label"]).astype({"label": int})
    return tr, te


def prepare() -> None:
    download_raw()
    tr, te = load_raw()
    tr, va = train_test_split(tr, test_size=config.VAL_FRACTION, stratify=tr["label"], random_state=config.SEED)
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", tr), ("val", va), ("test", te)):
        df.to_csv(config.PROC_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(df):6d} messages, OFF = {df['label'].mean():.1%}")


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.PROC_DIR / f"{name}.csv", dtype={"text": str}, keep_default_na=False)
    return df
