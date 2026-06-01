#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/scratch/nt612/Github/Deposit Franchise and Duration Gap in Bank Equity Asset Pricing}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p data/raw data/processed artifacts logs data_manifest figures/static figures/interactive

"$PYTHON_BIN" -m pip install -e ".[dev,wrds]" --only-binary=:all:
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
p = Path.home() / ".pgpass"
print("pgpass_exists", p.exists())
print("pgpass_permissions", oct(p.stat().st_mode)[-3:] if p.exists() else "missing")
PY

