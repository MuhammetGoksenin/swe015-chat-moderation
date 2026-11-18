"""Step 5 - evaluate the whole model ladder on the SAME held-out test set (R3) and build the results
table (R4): macro-F1, accuracy, OFF precision/recall, model size (MB) and CPU latency per message
(batch size 1, preprocessing included, averaged over >= 100 runs). Everything runs on CPU.

Outputs: results/results.csv, results/results.md, results/test_predictions.csv, results/errors_bert_int8.csv
"""
import json

import _common
import pandas as pd
import torch

from src import config
from src.data import load_split
from src.metrics import hardware, latency_ms, scores
from src.models import MODEL_KEYS, load_model, load_thresholds, set_cpu_threads

_common.seed_everything()
set_cpu_threads()
te = load_split("test")
texts, y = list(te.text), te.label.values
thresholds = load_thresholds()
hw = hardware()
print("Hardware:", hw)

rows, preds = [], pd.DataFrame({"id": te.id, "text": te.text, "label": y})
for key in MODEL_KEYS:
    try:
        m = load_model(key)
    except Exception as e:  # noqa: BLE001
        print(f"skip {key}: {e}")
        continue
    probs = m.predict_proba(texts)
    thr = thresholds.get(key, 0.5)
    s = scores(y, probs, thr)
    lat = latency_ms(m, texts)
    preds[f"p_{key}"] = probs
    rows.append({"model": m.name, "key": key, "precision": m.precision, **s, "threshold": thr,
                 "size_mb": m.size_mb(), **lat,
                 "within_budget": lat["latency_mean_ms"] <= config.LATENCY_BUDGET_MS})
    print(f"{m.name:32s} macro-F1={s['macro_f1']:.4f}  size={m.size_mb():7.1f} MB  "
          f"latency={lat['latency_mean_ms']:7.2f} ms (p95 {lat['latency_p95_ms']:.2f})")
    del m

res = pd.DataFrame(rows)
res.to_csv(config.RESULTS_DIR / "results.csv", index=False)
preds.to_csv(config.RESULTS_DIR / "test_predictions.csv", index=False)

md = ["| Model | Precision | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean / p95 (ms) | ≤ budget |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | {r['off_recall']:.3f} | "
              f"{r['size_mb']:.1f} | {r['latency_mean_ms']:.1f} / {r['latency_p95_ms']:.1f} | "
              f"{'✓' if r['within_budget'] else '✗'} |")
md += ["", f"Test set: OffensEval-TR 2020 official test split, n = {len(te)} ({y.mean():.1%} OFF).",
       f"Latency: batch size 1, preprocessing + tokenization included, mean over {config.LATENCY_RUNS} test messages "
       f"after 10 warm-up runs. Budget = {config.LATENCY_BUDGET_MS:.0f} ms/message.",
       f"Hardware: {hw}.", "Size = serialized model weights on disk (tokenizer/config files excluded)."]
(config.RESULTS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
(config.RESULTS_DIR / "hardware.json").write_text(json.dumps({"hardware": hw, "torch": torch.__version__}, indent=2))
print("\n".join(md))

if "p_bert_int8" in preds:
    thr = thresholds.get("bert_int8", 0.5)
    p = preds["p_bert_int8"]
    fn = preds[(preds.label == 1) & (p < thr)].assign(error="false negative (missed offensive)").sort_values("p_bert_int8")
    fp = preds[(preds.label == 0) & (p >= thr)].assign(error="false positive (clean flagged)").sort_values("p_bert_int8", ascending=False)
    pd.concat([fn.head(25), fp.head(25)]).to_csv(config.RESULTS_DIR / "errors_bert_int8.csv", index=False)
    print(f"\nINT8 errors: {len(fn)} false negatives, {len(fp)} false positives -> results/errors_bert_int8.csv")
