#!/usr/bin/env bash
# CUHK-X Large Model Track - team "Nabid Nur" - single inference entry point.
#   bash inference.sh <data_dir> <output_csv> [submission_id]
# <data_dir>: the official label-free Testing data (test_qa.csv + the clip videos, zipped or extracted).
# Two arguments reproduce the primary verification submission (Kaggle 56255655).
# A third argument selects another Selected submission by Kaggle Submission ID (56261134).
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <data_dir> <output_csv> [submission_id]" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_CSV="$2"
TARGET_SUBMISSION_ID="${3:-primary}"
PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ ! -d "$DATA_DIR" ]; then
  echo "Input data directory does not exist: $DATA_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$OUTPUT_CSV")"

# Deterministic run: pure-Python standard library, no randomness, no network, no GPU.
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1

exec python3 "$PACKAGE_ROOT/src/predict.py" \
  --data-dir "$DATA_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --submission-id "$TARGET_SUBMISSION_ID" \
  --config "$PACKAGE_ROOT/configs/final.yaml"
