#!/bin/bash
#SBATCH --job-name=ddgap_schema
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT="/scratch/nt612/Github/Deposit Franchise and Duration Gap in Bank Equity Asset Pricing"
PYTHON_BIN="$HOME/.conda/envs/ml_core/bin/python"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m deposit_duration.cli --log-file logs/schema_audit.log schema-audit

