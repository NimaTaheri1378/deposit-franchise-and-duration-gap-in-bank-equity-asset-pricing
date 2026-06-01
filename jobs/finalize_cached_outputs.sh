#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
ARTIFACTS_DIR="${2:-artifacts/full_live}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

STATUS_FILE="logs/finalize_cached_outputs.status"
DRIVER_LOG="logs/finalize_cached_outputs.driver.log"
echo RUNNING > "$STATUS_FILE"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "$rc" > "$STATUS_FILE"; fi' EXIT

{
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "artifacts_dir=$ARTIFACTS_DIR"

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_interpret.log interpret \
    --features "$ARTIFACTS_DIR/features.parquet" \
    --out-dir "$ARTIFACTS_DIR/interpretability"

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_ipca.log ipca \
    --features "$ARTIFACTS_DIR/features.parquet" \
    --out-dir "$ARTIFACTS_DIR/ipca"

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_backtest.log backtest \
    --predictions "$ARTIFACTS_DIR/models/predictions.parquet" \
    --out-dir "$ARTIFACTS_DIR/backtest" \
    --model lightgbm

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_implementation_scenarios.log implementation-scenarios \
    --predictions "$ARTIFACTS_DIR/models/predictions.parquet" \
    --out-dir "$ARTIFACTS_DIR/implementation_scenarios" \
    --model lightgbm

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_factor_alpha.log factor-alpha \
    --returns "$ARTIFACTS_DIR/backtest/portfolio_returns.parquet" \
    --factors "$ARTIFACTS_DIR/raw/factors_monthly.parquet" \
    --out-dir "$ARTIFACTS_DIR/backtest"

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_figures.log figures \
    --features "$ARTIFACTS_DIR/features.parquet" \
    --predictions "$ARTIFACTS_DIR/models/predictions.parquet" \
    --backtest-dir "$ARTIFACTS_DIR/backtest" \
    --out-dir "$ARTIFACTS_DIR/figures" \
    --event-dir "$ARTIFACTS_DIR/events" \
    --robustness-dir "$ARTIFACTS_DIR/robustness"

  "$PYTHON_BIN" -m deposit_duration.cli --log-file logs/finalize_visual_audit.log visual-audit \
    --figures-dir "$ARTIFACTS_DIR/figures/static" \
    --out-dir "$ARTIFACTS_DIR/figures" \
    --contact-sheet "$ARTIFACTS_DIR/figures/static_contact_sheet.png"

  echo "finished_at=$(date -Is)"
} > "$DRIVER_LOG" 2>&1

echo 0 > "$STATUS_FILE"
