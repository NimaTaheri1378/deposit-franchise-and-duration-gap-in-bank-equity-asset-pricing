# Deposit Franchise and Duration Gap Documentation

## Research Object

The project studies whether bank-equity returns price the interaction between
asset-side duration exposure and liability-side deposit fragility. The pipeline
is designed around public information timing: regulatory exposures are measured
at quarter end and become usable only after the configured reporting lag.

## Data Contract

Raw licensed extracts are written only under ignored artifact or data folders.
Public files contain code, SQL templates, configs, tests, documentation, and
aggregate summaries. The live schema audit freezes WRDS physical table names
into `configs/schema_map.yml`, while extraction manifests record table names,
row counts, date ranges, and output paths.

## Main Artifacts

| Layer | Command | Output |
|---|---|---|
| Schema audit | `ddgap schema-audit` | `configs/schema_map.yml` |
| Raw extraction | `ddgap extract-live` | ignored raw Parquet and manifest |
| Feature store | `ddgap features` | monthly point-in-time feature panel |
| Econometrics | `ddgap econometrics` | Fama-MacBeth and sort tables |
| Events | `ddgap events` | March 2023 CARs and local projections |
| Robustness | `ddgap robustness` | placebo, state, subperiod, double-sort checks |
| Public gates | `ddgap public-gates` | large/liquid robustness and incremental prediction checks |
| ML | `ddgap train` | walk-forward predictions and metrics |
| Interpretation | `ddgap interpret`, `ddgap ipca` | SHAP, PDP, family importance, factor layer |
| Backtest | `ddgap backtest` | portfolio returns, weights, costs, factor alpha |
| Visuals | `ddgap figures`, `ddgap visual-audit` | static and interactive figure pack |
| Claim ledger | `docs/claim_ledger.md` | evidence-bound public claims and forbidden wording |

## Empirical Discipline

- Unit of observation: listed-bank month.
- Return target: next-month excess return.
- Rebalance frequency: monthly.
- Main sample: 2004 through 2025.
- Untouched demo holdout: 2026 YTD by configuration.
- Default public-information lag: 45 days.
- Robustness lag variants: 30, 45, and 60 days.
- Main inference: time-series standard errors over monthly slopes or spreads.
- Pre-public evidence gates: large/liquid bank-month sample and fixed nonlinear group-ablation models comparing macro/market, bank-balance, and full feature sets.

## Visual QA

The visual pipeline writes PNG, PDF, and SVG versions of each static figure and
then audits image dimensions, blankness, and edge-darkness heuristics. A contact
sheet is generated for manual review, because figure readability cannot be
trusted to an automated check alone.

## Public Release Checklist

- `pytest` and `ruff` pass.
- CI synthetic smoke builds features, models, backtest, figures, docs, and report
  scaffold.
- `data_manifest/*_public_summary.json` contains only aggregate row counts and
  summary metrics.
- `figures/static/*` contains reviewed static PNG/PDF/SVG outputs.
- `figures/interactive/*` contains aggregate public-safe interactive outputs,
  not row-level licensed data.
- Secrets and raw licensed data remain ignored.
