# Claim Ledger and Closeout Audit

This ledger maps empirical claims to current artifacts and states the wording
discipline for public release. It is intentionally conservative: the project
can document evidence, but it should not claim causal proof or a finished
tradable alpha until stronger implementation and factor-spanning gates pass.

## Claim Ledger

| Claim | Evidence strength | Source artifacts | Allowed wording | Forbidden wording | Caveats and next gates |
|---|---|---|---|---|---|
| Bank duration exposure is priced in the cross-section of listed bank returns. | Strong descriptive asset-pricing evidence. | `artifacts/full_live/econometrics/fama_macbeth_summary.csv`; `data_manifest/full_live_public_summary.json` | Larger duration-gap proxy loads negatively in monthly Fama-MacBeth tests, with a Newey-West t-stat around -4.18 in the current full sample. | Duration exposure causes returns; the premium is structurally identified. | Uses proxy duration variables from base Bank Regulatory data; mechanism remains event-time/descriptive. |
| Deposit-fragility interactions matter for stress-state return behavior. | Moderate, state-dependent evidence. | `artifacts/full_live/robustness/state_dependence_summary.csv`; `artifacts/full_live/events/event_did_summary.csv`; `figures/static/state_dependent_spreads.png`; `figures/static/march_2023_event_car.png` | The signal is concentrated in stress and rate-shock states, consistent with a funding-fragility channel. | Deposit runs are proven to be the causal channel for all return effects. | March 2023 is an economic validation event, not a randomized experiment. |
| The ML layer improves out-of-sample ranking relative to simple baselines. | Strong predictive benchmark evidence. | `artifacts/full_live/models/walk_forward_metrics.csv`; `artifacts/full_live/models/walk_forward_tuning.json`; `figures/static/prediction_decile_returns.png` | LightGBM has the best mean rank IC in the current walk-forward run, about 0.15. | The model is universally stable across all future market states. | Hyperparameters are tuned within historical walk-forward windows; 2026 remains held out. |
| The post-cost strategy is positive but not yet a finished investable-alpha result. | Mixed economic evidence. | `artifacts/full_live/backtest/portfolio_summary.csv`; `artifacts/full_live/backtest/factor_alpha.csv`; `figures/static/strategy_cumulative_return.png`; `figures/static/turnover_cost_frontier.png` | The baseline post-cost long-short portfolio is positive, but factor-adjusted alpha is not yet strong enough for a finished alpha claim. | The strategy is production-ready, scalable, arbitrage, or fully orthogonal. | Needs stronger capacity, borrow, benchmark, and 2026 holdout evidence before tradability language. |
| Results are robust to key timing and implementation variants. | Good first-release robustness evidence. | `artifacts/full_live/lag_robustness/lag_variant_signal_spreads.csv`; `artifacts/full_live/implementation_scenarios/implementation_scenarios.csv`; `artifacts/full_live/robustness/subperiod_fama_macbeth_summary.csv` | The main signal survives alternative public-information lags and is reported under equal, value, and risk-scaled cost scenarios. | All robustness checks pass uniformly or remove all model risk. | Optional Group Lasso/CatBoost challengers are documented as optional and not part of the release claim. |
| The main duration evidence is not only a microcap artifact. | Good pre-public gate evidence. | `artifacts/full_live/public_gates/large_liquid/large_liquid_fama_macbeth_summary.csv`; `artifacts/full_live/public_gates/large_liquid/large_liquid_signal_spread_summary.csv` | The duration-gap result remains negative in a large/liquid half-sample, with a Fama-MacBeth t-stat around -2.97 and equal-weight spread t-stat around -3.34. | The result is capacity-proven or insensitive to all liquidity definitions. | The filter is a median market-cap and dollar-volume screen, not a full institutional capacity model. |
| Bank-balance variables add incremental ranking information beyond macro/market variables. | Moderate pre-public gate evidence. | `artifacts/full_live/public_gates/incremental_prediction/incremental_prediction_summary.csv` | In a fixed nonlinear walk-forward gate, the full feature set has higher mean rank IC than macro/market-only features, about 0.038 versus 0.027. | Bank variables dominate all macro predictors or guarantee higher portfolio spreads. | The full model's decile spread is not uniformly stronger than the macro/market-only benchmark, so this is an incremental-ranking claim only. |
| The repo is public-safe and reproducible from synthetic data. | Strong release-engineering evidence. | `.github/workflows/ci.yml`; `tests/integration/test_synthetic_pipeline.py`; `tests/regression/test_public_release_contract.py`; `figures/static/*`; `figures/interactive/*` | The public package includes CI, synthetic end-to-end tests, docs, SQL templates, and public-safe aggregate outputs. | Public files include licensed raw data or credentials. | Actual GitHub push is not performed in this workspace because no remote is configured. |

## Manifest Catalog

Current full-run manifests cover the required stages:

- `artifacts/full_live/live_extract_manifest.json`
- `artifacts/full_live/features.manifest.json`
- `artifacts/full_live/econometrics/fama_macbeth_manifest.json`
- `artifacts/full_live/robustness/robustness_manifest.json`
- `artifacts/full_live/lag_robustness/lag_variant_manifest.json`
- `artifacts/full_live/models/model_manifest.json`
- `artifacts/full_live/interpretability/interpretability_manifest.json`
- `artifacts/full_live/ipca/ipca_manifest.json`
- `artifacts/full_live/backtest/backtest_manifest.json`
- `artifacts/full_live/implementation_scenarios/implementation_scenarios_manifest.json`
- `artifacts/full_live/public_gates/public_push_gates_manifest.json`
- `artifacts/full_live/figures/figures_manifest.json`
- `artifacts/full_live/figures/visual_audit_manifest.json`

## Visual and Release Gates

| Gate | Status | Evidence |
|---|---|---|
| Full-run static visual audit | Pass | `artifacts/full_live/figures/visual_audit.csv`: 11 of 11 pass |
| Public static visual audit | Pass | `artifacts/public_visual_audit/visual_audit.csv`: 18 of 18 pass |
| Public static figure pack | Pass | `figures/static/*.png`, `*.pdf`, and `*.svg` |
| Public interactive outputs | Pass | `figures/interactive/scenario_rate_shock_explorer.html`; `figures/interactive/strategy_cumulative_return.html` |
| Lint | Pass | `python -m ruff check src tests scripts` |
| Tests | Pass | `python -m pytest -q`: 7 passed |
| Secret scan | Pass | Generic source/public artifact scan found 0 findings |

## Known Boundaries

- Bank Regulatory Premium is not required for the current run. Premium-only
  concepts are proxied with base Bank Regulatory fields and documented as
  proxies.
- The current release is not a manuscript. `paper/paper.qmd` is a report
  scaffold because manuscript writing was explicitly excluded.
- No GitHub push is recorded because the local repository has no remote
  configured. The public-safe worktree is ready for a selective commit and
  push once a target repository is provided.
