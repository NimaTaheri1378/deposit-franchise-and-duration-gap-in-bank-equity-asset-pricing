from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from deposit_duration.features.build import core_feature_columns
from deposit_duration.utils.manifest import write_manifest


def fama_macbeth(
    features_path: str | Path,
    out_dir: str | Path,
    predictors: list[str] | None = None,
    target: str = "next_month_excess_ret",
    min_obs_per_month: int = 25,
    nw_lags: int = 6,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictors is None:
        predictors = [
            "duration_gap_proxy",
            "deposit_fragility_index",
            "duration_x_uninsured",
            "log_market_cap",
            "capital_ratio",
            "cre_to_assets",
            "market_beta",
        ]
    rows = []
    for month, g in df.groupby("month"):
        cols = [c for c in predictors if c in g.columns]
        sample = g[[target, *cols]].replace([np.inf, -np.inf], np.nan).copy()
        for col in [target, *cols]:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
        sample = sample.dropna()
        if len(sample) < max(min_obs_per_month, len(cols) + 5):
            continue
        x = sm.add_constant(sample[cols].astype(float), has_constant="add")
        y = sample[target].astype(float)
        res = sm.OLS(y, x).fit()
        row = {"month": month, "nobs": len(sample)}
        row.update(res.params.to_dict())
        rows.append(row)
    betas = pd.DataFrame(rows).sort_values("month")
    if betas.empty:
        raise ValueError("No Fama-MacBeth months had enough observations.")

    summary_rows = []
    coef_cols = [c for c in betas.columns if c not in {"month", "nobs"}]
    for col in coef_cols:
        y = betas[col].dropna()
        if len(y) < 3:
            continue
        mean = float(y.mean())
        x = np.ones((len(y), 1))
        fit = sm.OLS(y.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        se = float(fit.bse[0])
        summary_rows.append(
            {
                "term": col,
                "mean_coef": mean,
                "nw_se": se,
                "t_stat": mean / se if se else np.nan,
                "months": len(y),
            }
        )
    summary = pd.DataFrame(summary_rows)
    betas.to_parquet(out_dir / "fama_macbeth_monthly_betas.parquet", index=False)
    summary.to_csv(out_dir / "fama_macbeth_summary.csv", index=False)
    write_manifest(
        out_dir / "fama_macbeth_manifest.json",
        {
            "kind": "fama_macbeth",
            "features": str(features_path),
            "predictors": predictors,
            "target": target,
            "months": len(betas),
            "nw_lags": nw_lags,
        },
    )
    return summary


def portfolio_sorts(
    features_path: str | Path,
    out_dir: str | Path,
    signal: str = "duration_x_uninsured",
    target: str = "next_month_excess_ret",
    buckets: int = 5,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for month, g in df.groupby("month"):
        sample = g[["month", signal, target]].replace([np.inf, -np.inf], np.nan).copy()
        sample[signal] = pd.to_numeric(sample[signal], errors="coerce")
        sample[target] = pd.to_numeric(sample[target], errors="coerce")
        sample = sample.dropna()
        if len(sample) < buckets * 5 or sample[signal].nunique() < buckets:
            continue
        sample = sample.copy()
        sample["bucket"] = pd.qcut(sample[signal], buckets, labels=False, duplicates="drop") + 1
        for bucket, bg in sample.groupby("bucket"):
            rows.append({"month": month, "bucket": int(bucket), "ret": float(bg[target].mean()), "n": len(bg)})
    rets = pd.DataFrame(rows)
    if rets.empty:
        raise ValueError("No portfolio sort returns produced.")
    wide = rets.pivot(index="month", columns="bucket", values="ret").sort_index()
    spread = wide[buckets] - wide[1]
    summary = pd.DataFrame(
        {
            "portfolio": [f"Q{buckets}-Q1"],
            "mean_monthly_ret": [float(spread.mean())],
            "vol_monthly": [float(spread.std(ddof=1))],
            "sharpe_annualized": [
                float(spread.mean() / spread.std(ddof=1) * np.sqrt(12)) if spread.std(ddof=1) else np.nan
            ],
            "months": [int(spread.notna().sum())],
        }
    )
    rets.to_parquet(out_dir / f"sorts_{signal}.parquet", index=False)
    summary.to_csv(out_dir / f"sorts_{signal}_summary.csv", index=False)
    write_manifest(
        out_dir / f"sorts_{signal}_manifest.json",
        {"kind": "portfolio_sorts", "signal": signal, "target": target, "buckets": buckets},
    )
    return summary


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    explicit = [c for c in core_feature_columns() if c in df.columns]
    ranks = [f"{c}_rank" for c in explicit if f"{c}_rank" in df.columns]
    return explicit + ranks
