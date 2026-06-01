from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from deposit_duration.features.build import build_features
from deposit_duration.models.econometrics import fama_macbeth
from deposit_duration.utils.manifest import write_manifest


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _hac_mean(series: pd.Series, lags: int = 6) -> tuple[float, float, float, int]:
    y = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(y) < 3:
        return np.nan, np.nan, np.nan, int(len(y))
    fit = sm.OLS(y.to_numpy(), np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean = float(y.mean())
    se = float(fit.bse[0])
    return mean, se, mean / se if se else np.nan, int(len(y))


def _monthly_spread(
    df: pd.DataFrame,
    signal: str,
    target: str,
    buckets: int,
    weight_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    cols = ["month", signal, target, *(["market_cap"] if weight_col == "market_cap" else [])]
    for month, g in df[cols].dropna().groupby("month", sort=True):
        if len(g) < buckets * 8 or g[signal].nunique() < buckets:
            continue
        sample = g.copy()
        sample["bucket"] = pd.qcut(sample[signal].rank(method="first"), buckets, labels=False) + 1
        bucket_returns = {}
        for bucket, bg in sample.groupby("bucket", observed=True):
            if weight_col and weight_col in bg.columns:
                weights = bg[weight_col].clip(lower=0)
                if weights.sum() > 0:
                    ret = float(np.average(bg[target], weights=weights))
                else:
                    ret = float(bg[target].mean())
            else:
                ret = float(bg[target].mean())
            bucket_returns[int(bucket)] = ret
        if 1 in bucket_returns and buckets in bucket_returns:
            rows.append(
                {
                    "month": month,
                    "signal": signal,
                    "weighting": "value" if weight_col else "equal",
                    "low": bucket_returns[1],
                    "high": bucket_returns[buckets],
                    "spread": bucket_returns[buckets] - bucket_returns[1],
                }
            )
    return pd.DataFrame(rows)


def signal_spread_robustness(
    features: pd.DataFrame,
    out_dir: Path,
    target: str = "next_month_excess_ret",
    signals: tuple[str, ...] = (
        "duration_gap_proxy",
        "deposit_fragility_index",
        "duration_x_uninsured",
        "duration_x_fragility",
    ),
    buckets: int = 5,
) -> pd.DataFrame:
    rows = []
    monthly_blocks = []
    for signal in signals:
        if signal not in features.columns:
            continue
        for weight_col in [None, "market_cap"]:
            monthly = _monthly_spread(features, signal, target, buckets, weight_col=weight_col)
            if monthly.empty:
                continue
            monthly_blocks.append(monthly)
            mean, se, t_stat, months = _hac_mean(monthly["spread"])
            rows.append(
                {
                    "test": "signal_spread",
                    "signal": signal,
                    "weighting": "value" if weight_col else "equal",
                    "mean_monthly_spread": mean,
                    "nw_se": se,
                    "t_stat": t_stat,
                    "months": months,
                }
            )
    monthly_spreads = pd.concat(monthly_blocks, ignore_index=True) if monthly_blocks else pd.DataFrame()
    if not monthly_spreads.empty:
        monthly_spreads.to_parquet(out_dir / "signal_spread_monthly.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "signal_spread_summary.csv", index=False)
    return summary


def double_sort_table(
    features: pd.DataFrame,
    out_dir: Path,
    row_signal: str = "deposit_fragility_index",
    col_signal: str = "duration_gap_proxy",
    target: str = "next_month_excess_ret",
    buckets: int = 5,
) -> pd.DataFrame:
    rows = []
    for month, g in features[["month", row_signal, col_signal, target]].dropna().groupby("month", sort=True):
        if len(g) < buckets * buckets * 3:
            continue
        sample = g.copy()
        sample["row_bucket"] = pd.qcut(sample[row_signal].rank(method="first"), buckets, labels=False) + 1
        sample["col_bucket"] = pd.qcut(sample[col_signal].rank(method="first"), buckets, labels=False) + 1
        cell = (
            sample.groupby(["row_bucket", "col_bucket"], observed=True)[target]
            .mean()
            .rename("mean_return")
            .reset_index()
        )
        cell["month"] = month
        rows.append(cell)
    monthly = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if monthly.empty:
        raise ValueError("No double-sort cells produced.")
    monthly.to_parquet(out_dir / "double_sort_monthly.parquet", index=False)
    table = (
        monthly.groupby(["row_bucket", "col_bucket"], observed=True)["mean_return"]
        .mean()
        .reset_index()
        .pivot(index="row_bucket", columns="col_bucket", values="mean_return")
        .sort_index()
    )
    table.to_csv(out_dir / "double_sort_mean_returns.csv")
    return table


def state_dependence(
    features: pd.DataFrame,
    out_dir: Path,
    signal: str = "duration_x_uninsured",
    target: str = "next_month_excess_ret",
    buckets: int = 5,
) -> pd.DataFrame:
    df = features.copy()
    states: dict[str, pd.Series] = {}
    if "rate_2y_change" in df.columns:
        states["rising_rates"] = df["rate_2y_change"] > df["rate_2y_change"].median()
        states["falling_rates"] = df["rate_2y_change"] <= df["rate_2y_change"].median()
    if "vix" in df.columns:
        states["high_vix"] = df["vix"] >= df["vix"].quantile(0.75)
        states["low_vix"] = df["vix"] <= df["vix"].quantile(0.25)

    rows = []
    for state, mask in states.items():
        monthly = _monthly_spread(df.loc[mask], signal, target, buckets)
        if monthly.empty:
            continue
        mean, se, t_stat, months = _hac_mean(monthly["spread"])
        rows.append(
            {
                "state": state,
                "signal": signal,
                "mean_monthly_spread": mean,
                "nw_se": se,
                "t_stat": t_stat,
                "months": months,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "state_dependence_summary.csv", index=False)
    return summary


def placebo_signal_test(
    features: pd.DataFrame,
    out_dir: Path,
    signal: str = "duration_x_uninsured",
    target: str = "next_month_excess_ret",
    buckets: int = 5,
    n_placebos: int = 100,
    seed: int = 1378,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _monthly_spread(features, signal, target, buckets)
    realized, _, realized_t, realized_months = _hac_mean(base["spread"])
    placebo_rows = []
    cols = ["month", signal, target]
    source = features[cols].dropna().copy()
    for draw in range(n_placebos):
        shuffled = []
        for _, g in source.groupby("month", sort=False):
            tmp = g.copy()
            tmp[signal] = rng.permutation(tmp[signal].to_numpy())
            shuffled.append(tmp)
        placebo = pd.concat(shuffled, ignore_index=True)
        monthly = _monthly_spread(placebo, signal, target, buckets)
        mean, _, t_stat, months = _hac_mean(monthly["spread"])
        placebo_rows.append({"draw": draw, "placebo_mean_spread": mean, "placebo_t_stat": t_stat, "months": months})
    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df["realized_mean_spread"] = realized
    placebo_df["realized_t_stat"] = realized_t
    placebo_df["realized_months"] = realized_months
    if placebo_df["placebo_mean_spread"].notna().any() and np.isfinite(realized):
        placebo_df["empirical_p_two_sided"] = (
            placebo_df["placebo_mean_spread"].abs() >= abs(realized)
        ).mean()
    else:
        placebo_df["empirical_p_two_sided"] = np.nan
    placebo_df.to_csv(out_dir / "placebo_signal_summary.csv", index=False)
    return placebo_df


def subperiod_fama_macbeth(
    features_path: str | Path,
    out_dir: Path,
    periods: tuple[tuple[str, str, str], ...] = (
        ("pre_covid", "2004-01-01", "2019-12-31"),
        ("covid_era", "2020-01-01", "2021-12-31"),
        ("tightening_era", "2022-01-01", "2025-12-31"),
    ),
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    df["month"] = pd.to_datetime(df["month"])
    rows = []
    for name, start, end in periods:
        block = df[df["month"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if block["month"].nunique() < 12 or len(block) < 300:
            continue
        tmp = out_dir / f"_subperiod_{name}.parquet"
        block.to_parquet(tmp, index=False)
        subdir = out_dir / f"fama_macbeth_{name}"
        try:
            summary = fama_macbeth(tmp, subdir, min_obs_per_month=20)
        finally:
            tmp.unlink(missing_ok=True)
        summary = summary.copy()
        summary["subperiod"] = name
        summary["start"] = start
        summary["end"] = end
        rows.append(summary)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.to_csv(out_dir / "subperiod_fama_macbeth_summary.csv", index=False)
    return combined


def lag_variant_robustness(
    raw_dir: str | Path,
    out_dir: str | Path,
    lags: tuple[int, ...] = (30, 45, 60),
    target: str = "next_month_excess_ret",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    outputs = {}
    for lag_days in lags:
        lag_dir = out_dir / f"lag_{lag_days}d"
        lag_dir.mkdir(parents=True, exist_ok=True)
        features_path = lag_dir / "features.parquet"
        manifest_path = lag_dir / "features.manifest.json"
        features = build_features(raw_dir, features_path, lag_days=lag_days, manifest_path=manifest_path)
        summary = signal_spread_robustness(features, lag_dir, target=target)
        summary = summary.copy()
        summary["lag_days"] = lag_days
        rows.append(summary)
        outputs[f"lag_{lag_days}d"] = {
            "features": str(features_path),
            "summary": str(lag_dir / "signal_spread_summary.csv"),
        }
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.to_csv(out_dir / "lag_variant_signal_spreads.csv", index=False)
    write_manifest(
        out_dir / "lag_variant_manifest.json",
        {
            "kind": "lag_variant_robustness",
            "raw_dir": str(raw_dir),
            "target": target,
            "lags": list(lags),
            "outputs": outputs,
        },
    )
    return combined


def run_robustness(
    features_path: str | Path,
    out_dir: str | Path,
    target: str = "next_month_excess_ret",
    n_placebos: int = 100,
    seed: int = 1378,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path)
    numeric_cols = [
        target,
        "market_cap",
        "duration_gap_proxy",
        "deposit_fragility_index",
        "duration_x_uninsured",
        "duration_x_fragility",
        "rate_2y_change",
        "vix",
    ]
    df = _coerce_numeric(df, [c for c in numeric_cols if c in df.columns])
    signal_spread_robustness(df, out_dir, target=target)
    double_sort_table(df, out_dir, target=target)
    state_dependence(df, out_dir, target=target)
    placebo_signal_test(df, out_dir, target=target, n_placebos=n_placebos, seed=seed)
    subperiod_fama_macbeth(features_path, out_dir)
    outputs = {
        "signal_spread_summary": str(out_dir / "signal_spread_summary.csv"),
        "double_sort_mean_returns": str(out_dir / "double_sort_mean_returns.csv"),
        "state_dependence_summary": str(out_dir / "state_dependence_summary.csv"),
        "placebo_signal_summary": str(out_dir / "placebo_signal_summary.csv"),
        "subperiod_fama_macbeth_summary": str(out_dir / "subperiod_fama_macbeth_summary.csv"),
    }
    write_manifest(
        out_dir / "robustness_manifest.json",
        {
            "kind": "robustness",
            "features": str(features_path),
            "target": target,
            "n_placebos": n_placebos,
            "seed": seed,
            "outputs": outputs,
        },
    )
    return outputs
