from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deposit_duration.utils.manifest import write_manifest
from deposit_duration.utils.validation import assert_unique_keys


QUARTERLY_REQUIRED = [
    "report_date",
    "total_assets",
    "total_deposits",
    "uninsured_deposits",
    "brokered_deposits",
    "large_time_deposits",
    "noninterest_deposits",
    "securities_afs",
    "securities_htm",
    "unrealized_losses",
    "tier1_capital",
    "cre_loans",
    "total_loans",
    "interest_rate_derivatives",
]

MONTHLY_REQUIRED = [
    "permno",
    "month",
    "ret",
    "excess_ret",
    "price",
    "shares_out",
    "volume",
    "market_beta",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _rank01(s: pd.Series) -> pd.Series:
    if s.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    return s.rank(pct=True)


def _winsor_by_month(df: pd.DataFrame, columns: list[str], month_col: str = "month") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        q = out.groupby(month_col)[col].quantile([0.01, 0.99]).unstack()
        q.columns = ["lo", "hi"]
        out = out.join(q, on=month_col)
        out[col] = out[col].clip(out["lo"], out["hi"])
        out = out.drop(columns=["lo", "hi"])
    return out


def build_features(
    raw_dir: str | Path,
    out: str | Path,
    lag_days: int = 45,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    out = Path(out)
    quarterly = pd.read_parquet(raw_dir / "bank_quarterly.parquet")
    monthly = pd.read_parquet(raw_dir / "stock_monthly.parquet")
    macro = pd.read_parquet(raw_dir / "macro_monthly.parquet")

    for col in QUARTERLY_REQUIRED:
        if col not in quarterly.columns:
            raise ValueError(f"Missing quarterly column: {col}")
    for col in MONTHLY_REQUIRED:
        if col not in monthly.columns:
            raise ValueError(f"Missing monthly column: {col}")

    quarterly = quarterly.copy()
    monthly = monthly.copy()
    macro = macro.copy()
    quarterly["report_date"] = pd.to_datetime(quarterly["report_date"])
    quarterly["available_date"] = quarterly["report_date"] + pd.to_timedelta(lag_days, unit="D")
    quarterly["available_month"] = quarterly["available_date"].dt.to_period("M").dt.to_timestamp("M")
    monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M").dt.to_timestamp("M")
    macro["month"] = pd.to_datetime(macro["month"]).dt.to_period("M").dt.to_timestamp("M")
    link_key = "permno" if "permno" in quarterly.columns and "permno" in monthly.columns else "permco"
    if link_key not in quarterly.columns or link_key not in monthly.columns:
        raise ValueError("Quarterly and monthly data must share either permno or permco.")

    q = quarterly.sort_values([link_key, "available_month"])
    m = monthly.sort_values([link_key, "month"])
    joined = []
    for key_value, left in m.groupby(link_key, sort=False):
        right = q[q[link_key] == key_value].sort_values("available_month")
        if right.empty:
            continue
        block = pd.merge_asof(
            left.sort_values("month"),
            right.drop(columns=[link_key]).sort_values("available_month"),
            left_on="month",
            right_on="available_month",
            direction="backward",
            allow_exact_matches=True,
        )
        block[link_key] = key_value
        joined.append(block)
    if not joined:
        raise ValueError("No monthly rows could be matched to lagged quarterly reports.")
    panel = pd.concat(joined, ignore_index=True)
    panel = panel.merge(macro, on="month", how="left", validate="many_to_one")

    panel["market_cap"] = panel["price"].abs() * panel["shares_out"]
    panel["log_market_cap"] = np.log(panel["market_cap"].replace(0, np.nan))
    panel["dollar_volume"] = panel["price"].abs() * panel["volume"]
    panel["turnover"] = _safe_div(panel["volume"], panel["shares_out"])
    panel["amihud"] = _safe_div(panel["ret"].abs(), panel["dollar_volume"])
    panel = panel.sort_values(["permno", "month"])
    panel["reversal_1m"] = panel.groupby("permno")["ret"].shift(1)
    panel["momentum_12_1"] = (
        panel.groupby("permno")["ret"]
        .transform(lambda s: (1 + s.shift(2)).rolling(11, min_periods=6).apply(np.nanprod, raw=True) - 1)
    )
    if "bank_market_ret" in panel.columns:
        panel["bank_relative_return"] = panel["excess_ret"] - panel["bank_market_ret"]
        residual = panel["excess_ret"] - panel["market_beta"] * panel["bank_market_ret"]
        panel["idiosyncratic_vol_proxy"] = residual.groupby(panel["permno"]).transform(
            lambda s: s.rolling(12, min_periods=6).std()
        )
    else:
        panel["bank_relative_return"] = np.nan
        panel["idiosyncratic_vol_proxy"] = np.nan

    panel["securities_to_assets"] = _safe_div(
        panel["securities_afs"] + panel["securities_htm"], panel["total_assets"]
    ).clip(lower=0, upper=1.5)
    panel["afs_to_assets"] = _safe_div(panel["securities_afs"], panel["total_assets"]).clip(lower=0, upper=1)
    panel["htm_to_assets"] = _safe_div(panel["securities_htm"], panel["total_assets"]).clip(lower=0, upper=1)
    panel["unrealized_loss_to_capital"] = _safe_div(-panel["unrealized_losses"], panel["tier1_capital"])
    derivative_intensity = _safe_div(panel["interest_rate_derivatives"], panel["total_assets"]).clip(lower=0)
    panel["hedge_offset"] = derivative_intensity / (1 + derivative_intensity)
    loan_assets_for_duration = _safe_div(panel["total_loans"], panel["total_assets"]).clip(lower=0, upper=1.5)
    panel["asset_duration_proxy"] = (
        1.0 * panel["htm_to_assets"] + 0.55 * panel["afs_to_assets"] + 0.35 * loan_assets_for_duration
    )
    panel["liability_stickiness_proxy"] = _safe_div(
        panel["noninterest_deposits"], panel["total_deposits"]
    ).clip(lower=0, upper=1) - 0.5 * _safe_div(panel["brokered_deposits"], panel["total_deposits"]).clip(
        lower=0, upper=1
    )
    panel["duration_gap_proxy"] = panel["asset_duration_proxy"] - panel["liability_stickiness_proxy"] - 0.25 * panel["hedge_offset"]

    panel["large_time_deposit_share"] = _safe_div(panel["large_time_deposits"], panel["total_deposits"]).clip(0, 1)
    panel["uninsured_deposit_share_raw"] = _safe_div(panel["uninsured_deposits"], panel["total_deposits"])
    panel["uninsured_deposit_missing"] = panel["uninsured_deposit_share_raw"].isna().astype(float)
    panel["brokered_deposit_share"] = _safe_div(panel["brokered_deposits"], panel["total_deposits"]).clip(0, 1)
    panel["noninterest_deposit_share"] = _safe_div(panel["noninterest_deposits"], panel["total_deposits"]).clip(0, 1)
    uninsured_proxy = (panel["large_time_deposit_share"].fillna(0) + panel["brokered_deposit_share"].fillna(0)).clip(0, 1)
    panel["uninsured_deposit_share"] = panel["uninsured_deposit_share_raw"].fillna(uninsured_proxy).clip(0, 1)
    panel["deposit_fragility_index"] = (
        panel["uninsured_deposit_share"]
        + 0.6 * panel["brokered_deposit_share"]
        - 0.5 * panel["noninterest_deposit_share"]
    )

    panel["capital_ratio"] = _safe_div(panel["tier1_capital"], panel["total_assets"])
    panel["cre_to_assets"] = _safe_div(panel["cre_loans"], panel["total_assets"])
    panel["loan_to_deposit"] = _safe_div(panel["total_loans"], panel["total_deposits"])

    panel["duration_x_uninsured"] = panel["duration_gap_proxy"] * panel["uninsured_deposit_share"]
    panel["duration_x_fragility"] = panel["duration_gap_proxy"] * panel["deposit_fragility_index"]
    panel["htm_x_rate_shock"] = panel["htm_to_assets"] * panel["rate_2y_change"]
    panel["fragility_x_vix"] = panel["deposit_fragility_index"] * panel["vix"]
    panel["cre_x_fragility"] = panel["cre_to_assets"] * panel["deposit_fragility_index"]
    panel["hedge_x_duration"] = panel["hedge_offset"] * panel["duration_gap_proxy"]

    feature_cols = core_feature_columns()
    panel = _winsor_by_month(panel, [c for c in feature_cols if c in panel.columns])
    for col in feature_cols:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby("month")[col].transform(_rank01)

    panel = panel.sort_values(["permno", "month"])
    panel["next_month_excess_ret"] = panel.groupby("permno")["excess_ret"].shift(-1)
    panel["downside_next_month"] = (panel["next_month_excess_ret"] < -0.05).astype("float")
    panel.loc[panel["next_month_excess_ret"].isna(), "downside_next_month"] = np.nan

    panel = panel.dropna(subset=["next_month_excess_ret", "duration_gap_proxy", "deposit_fragility_index"])
    result = assert_unique_keys(panel, ["permno", "month"], "feature_panel")
    if not result.passed:
        raise ValueError(result.message)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    if manifest_path is None:
        manifest_path = out.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        {
            "kind": "feature_store",
            "path": str(out),
            "rows": len(panel),
            "permnos": int(panel["permno"].nunique()),
            "month_min": str(panel["month"].min().date()),
            "month_max": str(panel["month"].max().date()),
            "lag_days": lag_days,
            "feature_columns": feature_cols,
        },
    )
    return panel


def core_feature_columns() -> list[str]:
    return [
        "log_market_cap",
        "turnover",
        "amihud",
        "reversal_1m",
        "momentum_12_1",
        "bank_relative_return",
        "idiosyncratic_vol_proxy",
        "market_beta",
        "securities_to_assets",
        "afs_to_assets",
        "htm_to_assets",
        "unrealized_loss_to_capital",
        "hedge_offset",
        "asset_duration_proxy",
        "liability_stickiness_proxy",
        "duration_gap_proxy",
        "uninsured_deposit_share",
        "uninsured_deposit_missing",
        "large_time_deposit_share",
        "brokered_deposit_share",
        "noninterest_deposit_share",
        "deposit_fragility_index",
        "capital_ratio",
        "cre_to_assets",
        "loan_to_deposit",
        "rate_2y_change",
        "term_spread",
        "vix",
        "duration_x_uninsured",
        "duration_x_fragility",
        "htm_x_rate_shock",
        "fragility_x_vix",
        "cre_x_fragility",
        "hedge_x_duration",
    ]
