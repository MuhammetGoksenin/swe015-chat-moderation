"""Metrics, threshold tuning and latency benchmarking shared by all training/evaluation scripts."""
import json
import platform
import statistics
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from . import config


def scores(y_true, probs, thr: float = 0.5) -> dict:
    y_pred = (np.asarray(probs) >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "off_precision": float(p[0]), "off_recall": float(r[0]), "off_f1": float(f[0]),
    }


def tune_threshold(y_val, probs_val) -> float:
    """Pick the decision threshold that maximizes macro-F1 on the VALIDATION split (never on test)."""
    grid = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 91), np.asarray(probs_val)]), 4))
    best = max(grid, key=lambda t: f1_score(y_val, (np.asarray(probs_val) >= t).astype(int), average="macro"))
    return float(best)


def save_threshold(key: str, thr: float) -> None:
    d = json.loads(config.THRESHOLDS_PATH.read_text()) if config.THRESHOLDS_PATH.exists() else {}
    d[key] = thr
    config.THRESHOLDS_PATH.write_text(json.dumps(d, indent=2))


def latency_ms(model, texts, runs: int = config.LATENCY_RUNS, warmup: int = 10) -> dict:
    """Per-message latency (batch size 1, preprocessing included), averaged over `runs` messages."""
    for t in texts[:warmup]:
        model.predict_proba([t])
    times = []
    for i in range(runs):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"latency_mean_ms": statistics.mean(times), "latency_p95_ms": times[int(0.95 * len(times)) - 1], "latency_runs": runs}


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    import os
    return f"{cpu} | {os.cpu_count()} logical CPUs | torch threads={torch.get_num_threads()} | CPU only"
