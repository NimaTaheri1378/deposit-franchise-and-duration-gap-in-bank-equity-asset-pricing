from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from deposit_duration.models.econometrics import fama_macbeth
from deposit_duration.models.robustness import signal_spread_robustness
from deposit_duration.utils.manifest import write_manifest


MODEL_GROUPS = {
    "macro_market": [
        "log_market_cap",
        "market_beta",
        "amihud",
        "turnover",
        "reversal_1m",
        "momentum_12_1",
        "bank_relative_return",
        "idiosyncratic_vol_proxy",
        "rate_2y_change",
        "term_spread",
        "vix",
    ],
    "bank_balance": [
        "duration_gap_proxy",
        "deposit_fragility_index",
        "duration_x_uninsured",
        "securities_to_assets",
        "htm_to_assets",
        "unrealized_loss_to_capital",
        "hedge_offset",
        "capital_ratio",
        "cre_to_assets",
        "loan_to_deposit",
    ],
    "full": [
        "log_market_cap",
        "market_beta",
        "amihud",
        "turnover",
        "reversal_1m",
        "momentum_12_1",
        "bank_relative_return",
        "idiosyncratic_vol_proxy",
        "rate_2y_change",
        "term_spread",
        "vix",
        "duration_gap_proxy",
        "deposit_fragility_index",
        "duration_x_uninsured",
        "duration_x_fragility",
        "fragility_x_vix",
        "cre_x_fragility",
        "hedge_x_duration",
        "htm_x_rate_shock",
        "securities_to_assets",
        "htm_to_assets",
        "unrealized_loss_to_capital",
        "hedge_offset",
        "capital_ratio",
        "cre_to_assets",
        "loan_to_deposit",
    ],
}


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _spearman(y: pd.Series, pred: pd.Series | np.ndarray) -> float:
    return float(pd.Series(y).corr(pd.Series(pred, index=y.index), method="spearman"))


def _hac_mean(series: pd.Series, lags: int = 6) -> tuple[float, float, float, int]:
    y = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(y) < 3:
        return np.nan, np.nan, np.nan, int(len(y))
    fit = sm.OLS(y.to_numpy(), np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean = float(y.mean())
    se = float(fit.bse[0])
    return mean, se, mean / se if se else np.nan, int(len(y))


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and not df[c].isna().all()]


def _large_liquid_filter(df: pd.DataFrame, quantile: float) -> pd.Series:
    market_cut = df.groupby("month")["market_cap"].transform(lambda s: s.quantile(quantile))
    volume_cut = df.groupby("month")["dollar_volume"].transform(lambda s: s.quantile(quantile))
    return (df["market_cap"] >= market_cut) & (df["dollar_volume"] >= volume_cut)


def run_large_liquid_gate(
    features_path: str | Path,
    out_dir: str | Path,
    target: str = "next_month_excess_ret",
    quantile: float = 0.50,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path).replace([np.inf, -np.inf], np.nan)
    df["month"] = pd.to_datetime(df["month"])
    df = _coerce_numeric(
        df,
        [
            target,
            "market_cap",
            "dollar_volume",
            "duration_gap_proxy",
            "deposit_fragility_index",
            "duration_x_uninsured",
            "duration_x_fragility",
        ],
    )
    liquid = df.loc[_large_liquid_filter(df, quantile)].copy()
    sample_summary = pd.DataFrame(
        [
            {
                "gate": "large_liquid",
                "market_cap_quantile": quantile,
                "dollar_volume_quantile": quantile,
                "rows": len(liquid),
                "months": int(liquid["month"].nunique()),
                "permnos": int(liquid["permno"].nunique()),
            }
        ]
    )
    sample_summary.to_csv(out_dir / "large_liquid_sample_summary.csv", index=False)

    tmp = out_dir / "_large_liquid_features.parquet"
    liquid.to_parquet(tmp, index=False)
    try:
        fm = fama_macbeth(tmp, out_dir / "large_liquid_fama_macbeth", min_obs_per_month=20)
    finally:
        tmp.unlink(missing_ok=True)
    spread_dir = out_dir / "large_liquid_signal_spreads"
    spread_dir.mkdir(parents=True, exist_ok=True)
    spreads = signal_spread_robustness(liquid, spread_dir, target=target)
    fm.to_csv(out_dir / "large_liquid_fama_macbeth_summary.csv", index=False)
    spreads.to_csv(out_dir / "large_liquid_signal_spread_summary.csv", index=False)
    write_manifest(
        out_dir / "large_liquid_gate_manifest.json",
        {
            "kind": "large_liquid_gate",
            "features": str(features_path),
            "target": target,
            "quantile": quantile,
            "rows": len(liquid),
            "months": int(liquid["month"].nunique()),
            "permnos": int(liquid["permno"].nunique()),
            "outputs": {
                "sample_summary": str(out_dir / "large_liquid_sample_summary.csv"),
                "fama_macbeth": str(out_dir / "large_liquid_fama_macbeth_summary.csv"),
                "signal_spreads": str(out_dir / "large_liquid_signal_spread_summary.csv"),
            },
        },
    )
    return {
        "sample_summary": str(out_dir / "large_liquid_sample_summary.csv"),
        "fama_macbeth": str(out_dir / "large_liquid_fama_macbeth_summary.csv"),
        "signal_spreads": str(out_dir / "large_liquid_signal_spread_summary.csv"),
    }


def _make_hgb(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_iter=180,
                    learning_rate=0.035,
                    l2_regularization=0.05,
                    min_samples_leaf=50,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_incremental_prediction_gate(
    features_path: str | Path,
    out_dir: str | Path,
    first_test_year: int = 2017,
    last_test_year: int = 2025,
    target: str = "next_month_excess_ret",
    seed: int = 1378,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path).replace([np.inf, -np.inf], np.nan)
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    feature_cols = sorted({c for cols in MODEL_GROUPS.values() for c in cols})
    df = _coerce_numeric(df, [target, *feature_cols])
    df = df.dropna(subset=[target])

    metrics_rows = []
    prediction_blocks = []
    for test_year in range(first_test_year, last_test_year + 1):
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        for group, columns in MODEL_GROUPS.items():
            cols = _available_columns(df, columns)
            if not cols:
                continue
            model = _make_hgb(seed)
            model.fit(train[cols], train[target])
            pred = model.predict(test[cols])
            block = test[["permno", "month", target]].copy()
            block["feature_group"] = group
            block["prediction"] = pred
            block["test_year"] = test_year
            prediction_blocks.append(block)
            metrics_rows.append(
                {
                    "feature_group": group,
                    "test_year": test_year,
                    "nobs": len(test),
                    "spearman_ic": _spearman(test[target], pd.Series(pred, index=test.index)),
                    "feature_count": len(cols),
                }
            )

    if not prediction_blocks:
        raise ValueError("No incremental prediction-gate predictions were produced.")
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    spread_rows = []
    for (group, month), g in predictions.groupby(["feature_group", "month"], sort=True):
        if len(g) < 50:
            continue
        sample = g.copy()
        sample["decile"] = pd.qcut(sample["prediction"].rank(method="first"), 10, labels=False) + 1
        decile_returns = sample.groupby("decile", observed=True)[target].mean()
        if 1 in decile_returns.index and 10 in decile_returns.index:
            spread_rows.append(
                {
                    "feature_group": group,
                    "month": month,
                    "decile_10_minus_1": float(decile_returns.loc[10] - decile_returns.loc[1]),
                }
            )
    spreads = pd.DataFrame(spread_rows)
    summary_rows = []
    for group, g in metrics.groupby("feature_group", observed=True):
        group_spreads = spreads.loc[spreads["feature_group"] == group, "decile_10_minus_1"]
        mean_spread, se_spread, t_spread, months = _hac_mean(group_spreads)
        summary_rows.append(
            {
                "feature_group": group,
                "mean_spearman_ic": float(g["spearman_ic"].mean()),
                "median_spearman_ic": float(g["spearman_ic"].median()),
                "test_years": int(g["test_year"].nunique()),
                "mean_decile_10_minus_1": mean_spread,
                "decile_spread_nw_se": se_spread,
                "decile_spread_t_stat": t_spread,
                "decile_spread_months": months,
                "mean_feature_count": float(g["feature_count"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_spearman_ic", ascending=False)
    predictions.to_parquet(out_dir / "incremental_predictions.parquet", index=False)
    metrics.to_csv(out_dir / "incremental_prediction_metrics.csv", index=False)
    spreads.to_csv(out_dir / "incremental_prediction_decile_spreads.csv", index=False)
    summary.to_csv(out_dir / "incremental_prediction_summary.csv", index=False)
    write_manifest(
        out_dir / "incremental_prediction_gate_manifest.json",
        {
            "kind": "incremental_prediction_gate",
            "features": str(features_path),
            "target": target,
            "first_test_year": first_test_year,
            "last_test_year": last_test_year,
            "model": "hist_gradient_boosting",
            "feature_groups": MODEL_GROUPS,
            "outputs": {
                "metrics": str(out_dir / "incremental_prediction_metrics.csv"),
                "summary": str(out_dir / "incremental_prediction_summary.csv"),
                "decile_spreads": str(out_dir / "incremental_prediction_decile_spreads.csv"),
            },
        },
    )
    return {
        "metrics": str(out_dir / "incremental_prediction_metrics.csv"),
        "summary": str(out_dir / "incremental_prediction_summary.csv"),
        "decile_spreads": str(out_dir / "incremental_prediction_decile_spreads.csv"),
    }


def run_public_push_gates(
    features_path: str | Path,
    out_dir: str | Path,
    first_test_year: int = 2017,
    last_test_year: int = 2025,
    target: str = "next_month_excess_ret",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    large_liquid = run_large_liquid_gate(features_path, out_dir / "large_liquid", target=target)
    incremental = run_incremental_prediction_gate(
        features_path,
        out_dir / "incremental_prediction",
        first_test_year=first_test_year,
        last_test_year=last_test_year,
        target=target,
    )
    outputs = {
        "large_liquid": large_liquid,
        "incremental_prediction": incremental,
    }
    write_manifest(
        out_dir / "public_push_gates_manifest.json",
        {
            "kind": "public_push_gates",
            "features": str(features_path),
            "target": target,
            "outputs": outputs,
        },
    )
    return {
        "large_liquid_fama_macbeth": large_liquid["fama_macbeth"],
        "large_liquid_signal_spreads": large_liquid["signal_spreads"],
        "incremental_prediction_summary": incremental["summary"],
    }
