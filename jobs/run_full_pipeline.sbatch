#!/bin/bash
#SBATCH --job-name=ddgap_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT="/scratch/nt612/Github/Deposit Franchise and Duration Gap in Bank Equity Asset Pricing"
PYTHON_BIN="$HOME/.conda/envs/ml_core/bin/python"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m deposit_duration.cli --log-file logs/schema_audit.log schema-audit
"$PYTHON_BIN" -m deposit_duration.cli --log-file logs/smoke.log smoke --out-dir artifacts/smoke --use-gpu
"$PYTHON_BIN" -m deposit_duration.cli --log-file logs/full_live.log full-live --out-dir artifacts/full_live --start 2004-01-01 --end 2025-12-31 --use-gpu
