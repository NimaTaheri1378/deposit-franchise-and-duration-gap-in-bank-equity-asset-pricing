#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
ARTIFACTS_DIR="${2:-artifacts/full_live}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

STATUS_FILE="logs/refresh_interpretability.status"
DRIVER_LOG="logs/refresh_interpretability.driver.log"
echo RUNNING > "$STATUS_FILE"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "$rc" > "$STATUS_FILE"; fi' EXIT

{
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "artifacts_dir=$ARTIFACTS_DIR"
  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/refresh_interpretability.log interpret \
    --features "$ARTIFACTS_DIR/features.parquet" \
    --out-dir "$ARTIFACTS_DIR/interpretability"
  echo "finished_at=$(date -Is)"
} > "$DRIVER_LOG" 2>&1

echo 0 > "$STATUS_FILE"
