from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from deposit_duration.utils.manifest import write_manifest


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _month_distance(month: pd.Series, event_month: pd.Timestamp) -> pd.Series:
    m = pd.to_datetime(month).dt.to_period("M")
    e = event_month.to_period("M")
    return (m.dt.year - e.year) * 12 + (m.dt.month - e.month)


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd


def event_study_cars(
    features_path: str | Path,
    out_dir: str | Path,
    event_month: str = "2023-03-31",
    score: str = "duration_x_uninsured",
    return_col: str = "excess_ret",
    pre_months: int = 6,
    post_months: int = 6,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_ts = pd.to_datetime(event_month).to_period("M").to_timestamp("M")
    needed = [score, return_col, "log_market_cap", "capital_ratio", "market_beta"]
    df = _coerce_numeric(df, needed)
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp("M")

    baseline = df[df["month"] < event_ts].sort_values(["permno", "month"]).groupby("permno").tail(1)
    baseline = baseline.dropna(subset=[score])
    if len(baseline) < 30:
        raise ValueError("Not enough banks with pre-event scores for event-study groups.")
    lo, hi = baseline[score].quantile([1 / 3, 2 / 3])
    groups = baseline[["permno", score]].copy()
    groups["event_group"] = np.select(
        [groups[score] <= lo, groups[score] >= hi],
        ["low_fragility_duration", "high_fragility_duration"],
        default="middle",
    )
    groups = groups[groups["event_group"] != "middle"][["permno", "event_group"]]

    window = df.merge(groups, on="permno", how="inner")
    window["relative_month"] = _month_distance(window["month"], event_ts)
    window = window[window["relative_month"].between(-pre_months, post_months)]
    window = window.dropna(subset=[return_col])
    monthly = (
        window.groupby(["event_group", "relative_month"], observed=True)[return_col]
        .mean()
        .rename("mean_excess_ret")
        .reset_index()
        .sort_values(["event_group", "relative_month"])
    )
    monthly["car"] = monthly.groupby("event_group", observed=True)["mean_excess_ret"].cumsum()
    monthly.to_csv(out_dir / "event_car.csv", index=False)

    did = window[window["relative_month"].between(-pre_months, post_months)].copy()
    did["treated"] = (did["event_group"] == "high_fragility_duration").astype(float)
    did["post"] = (did["relative_month"] >= 0).astype(float)
    did["treated_post"] = did["treated"] * did["post"]
    controls = [c for c in ["log_market_cap", "capital_ratio", "market_beta"] if c in did.columns]
    did = did.dropna(subset=[return_col, "treated", "post", "treated_post", *controls])
    x = sm.add_constant(did[["treated", "post", "treated_post", *controls]].astype(float), has_constant="add")
    fit = sm.OLS(did[return_col].astype(float), x).fit(cov_type="HC1")
    did_summary = pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "se": fit.bse.values,
            "t_stat": fit.tvalues.values,
            "nobs": int(fit.nobs),
        }
    )
    did_summary.to_csv(out_dir / "event_did_summary.csv", index=False)
    return monthly


def local_projections(
    features_path: str | Path,
    out_dir: str | Path,
    exposure: str = "duration_x_uninsured",
    shock: str = "rate_2y_change",
    return_col: str = "excess_ret",
    horizons: tuple[int, ...] = (1, 3, 6),
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = ["log_market_cap", "capital_ratio", "cre_to_assets", "market_beta", "term_spread", "vix"]
    needed = [exposure, shock, return_col, *controls]
    df = _coerce_numeric(df, needed)
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp("M")
    df = df.sort_values(["permno", "month"])
    df["exposure_z"] = df.groupby("month")[exposure].transform(_zscore)
    df["shock_z"] = _zscore(df[shock])
    df["exposure_x_shock"] = df["exposure_z"] * df["shock_z"]

    rows = []
    for horizon in horizons:
        y = pd.Series(0.0, index=df.index)
        valid = pd.Series(True, index=df.index)
        for step in range(1, horizon + 1):
            shifted = df.groupby("permno")[return_col].shift(-step)
            y = y + shifted.fillna(0)
            valid &= shifted.notna()
        sample = df.loc[valid].copy()
        sample["cum_future_excess_ret"] = y.loc[valid]
        rhs = ["exposure_z", "shock_z", "exposure_x_shock", *[c for c in controls if c in sample.columns]]
        sample = sample.dropna(subset=["cum_future_excess_ret", *rhs])
        if len(sample) < max(100, len(rhs) * 10):
            continue
        x = sm.add_constant(sample[rhs].astype(float), has_constant="add")
        fit = sm.OLS(sample["cum_future_excess_ret"].astype(float), x).fit(cov_type="HC1")
        for term in fit.params.index:
            rows.append(
                {
                    "horizon": horizon,
                    "term": term,
                    "coef": float(fit.params[term]),
                    "se": float(fit.bse[term]),
                    "t_stat": float(fit.tvalues[term]),
                    "nobs": int(fit.nobs),
                    "exposure": exposure,
                    "shock": shock,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No local-projection horizons had enough observations.")
    summary.to_csv(out_dir / "local_projection_summary.csv", index=False)
    return summary


def run_event_identification(features_path: str | Path, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_study_cars(features_path, out_dir)
    local_projections(features_path, out_dir)
    manifest = {
        "kind": "event_identification",
        "features": str(features_path),
        "event_month": "2023-03-31",
        "event_score": "duration_x_uninsured",
        "local_projection_exposure": "duration_x_uninsured",
        "local_projection_shock": "rate_2y_change",
        "outputs": {
            "event_car": str(out_dir / "event_car.csv"),
            "event_did_summary": str(out_dir / "event_did_summary.csv"),
            "local_projection_summary": str(out_dir / "local_projection_summary.csv"),
        },
    }
    write_manifest(out_dir / "event_identification_manifest.json", manifest)
    return manifest["outputs"]
