#!/usr/bin/env bash
# End-to-end smoke test of the pipeline on SYNTHETIC data with a tiny randomly initialised BERT.
# It only checks that the code runs; the numbers it prints are meaningless and must NOT be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
export DATA_DIR=/tmp/swe015_smoke/data MODELS_DIR=/tmp/swe015_smoke/models RESULTS_DIR=/tmp/swe015_smoke/results
export BERT_MODEL=/tmp/swe015_smoke/tiny_bert BERT_EPOCHS=1 LATENCY_RUNS=100
rm -rf /tmp/swe015_smoke && mkdir -p /tmp/swe015_smoke
python tests/make_fake_data.py
python train/00_prepare_data.py
python train/01_baselines.py
python train/02_word2vec_ffnn.py
python train/03_finetune_bert.py
python train/04_quantize.py
python train/05_evaluate.py
echo "SMOKE TEST OK"
