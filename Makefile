.PHONY: install test lint extract synthetic features econometrics events robustness lag-robustness public-gates train interpret ipca backtest implementation-scenarios factor-alpha figures visual-audit smoke public-summary public-artifacts paper docs ci

install:
	python -m pip install -e ".[all]"

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

extract:
	ddgap extract-live --out-dir artifacts/live

synthetic:
	ddgap synthetic --out-dir artifacts/synthetic

features:
	ddgap features --raw-dir artifacts/synthetic/raw --out artifacts/synthetic/features.parquet

econometrics:
	ddgap econometrics --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/econometrics

events:
	ddgap events --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/events

robustness:
	ddgap robustness --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/robustness

lag-robustness:
	ddgap lag-robustness --raw-dir artifacts/synthetic/raw --out-dir artifacts/synthetic/lag_robustness

public-gates:
	ddgap public-gates --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/public_gates

train:
	ddgap train --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/models

interpret:
	ddgap interpret --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/interpretability

ipca:
	ddgap ipca --features artifacts/synthetic/features.parquet --out-dir artifacts/synthetic/ipca

backtest:
	ddgap backtest --predictions artifacts/synthetic/models/predictions.parquet --out-dir artifacts/synthetic/backtest

implementation-scenarios:
	ddgap implementation-scenarios --predictions artifacts/synthetic/models/predictions.parquet --out-dir artifacts/synthetic/implementation_scenarios

factor-alpha:
	ddgap factor-alpha --returns artifacts/synthetic/backtest/portfolio_returns.parquet --factors artifacts/synthetic/raw/factors_monthly.parquet --out-dir artifacts/synthetic/backtest

figures:
	ddgap figures --features artifacts/synthetic/features.parquet --predictions artifacts/synthetic/models/predictions.parquet --backtest-dir artifacts/synthetic/backtest --out-dir artifacts/synthetic/figures --event-dir artifacts/synthetic/events --robustness-dir artifacts/synthetic/robustness

visual-audit:
	ddgap visual-audit --figures-dir artifacts/synthetic/figures/static --out-dir artifacts/synthetic/figures --contact-sheet artifacts/synthetic/figures/static_contact_sheet.png

smoke:
	ddgap smoke --out-dir artifacts/smoke

public-summary:
	python scripts/summarize_artifacts.py --artifacts-dir artifacts/full_live --out data_manifest/full_live_public_summary.json

public-artifacts: public-summary
	python scripts/publish_public_artifacts.py --artifacts-dir artifacts/full_live

paper:
	python scripts/build_report.py --summary data_manifest/full_live_public_summary.json --out artifacts/reports/paper.html

docs:
	python scripts/build_docs.py --out-dir artifacts/site

ci: lint test smoke docs
