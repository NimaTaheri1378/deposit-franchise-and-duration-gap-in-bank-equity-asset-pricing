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


def _demean_beta(weights: pd.Series, beta: pd.Series) -> pd.Series:
    if weights.abs().sum() == 0 or beta.notna().sum() < 3:
        return weights
    exposure = float((weights * beta.fillna(beta.mean())).sum())
    beta_centered = beta.fillna(beta.mean()) - beta.mean()
    denom = float((beta_centered**2).sum())
    if denom <= 1e-12:
        return weights
    adjusted = weights - exposure * beta_centered / denom
    long_sum = adjusted[adjusted > 0].sum()
    short_sum = -adjusted[adjusted < 0].sum()
    if long_sum > 0:
        adjusted[adjusted > 0] /= long_sum
    if short_sum > 0:
        adjusted[adjusted < 0] /= short_sum
    return adjusted


def _normalize_sides(weights: pd.Series) -> pd.Series:
    out = weights.copy()
    long_sum = out[out > 0].sum()
    short_sum = -out[out < 0].sum()
    if long_sum > 0:
        out[out > 0] /= long_sum
    if short_sum > 0:
        out[out < 0] /= short_sum
    return out


def _cap_side(abs_weights: pd.Series, max_weight: float) -> pd.Series:
    if abs_weights.empty or max_weight <= 0 or abs_weights.sum() <= 0:
        return abs_weights
    if len(abs_weights) * max_weight < 1.0:
        return abs_weights / abs_weights.sum()
    out = abs_weights / abs_weights.sum()
    capped = pd.Series(False, index=out.index)
    for _ in range(len(out) + 1):
        over = (out > max_weight) & ~capped
        if not over.any():
            break
        capped |= over
        out.loc[capped] = max_weight
        remaining = 1.0 - out.loc[capped].sum()
        free = ~capped
        if remaining <= 0 or not free.any():
            break
        out.loc[free] = abs_weights.loc[free] / abs_weights.loc[free].sum() * remaining
    return out / out.sum()


def _apply_abs_cap(weights: pd.Series, max_weight: float | None) -> pd.Series:
    if max_weight is None or max_weight <= 0:
        return _normalize_sides(weights)
    out = weights.copy()
    pos = out > 0
    neg = out < 0
    if pos.any():
        out.loc[pos] = _cap_side(out.loc[pos].abs(), max_weight)
    if neg.any():
        out.loc[neg] = -_cap_side(out.loc[neg].abs(), max_weight)
    return _normalize_sides(out)


def _initial_side_weights(sample: pd.DataFrame, mask: pd.Series, weighting: str) -> pd.Series:
    idx = sample.index[mask]
    if len(idx) == 0:
        return pd.Series(dtype=float)
    if weighting == "value" and "market_cap" in sample.columns:
        raw = pd.to_numeric(sample.loc[idx, "market_cap"], errors="coerce").clip(lower=0).astype("float64")
    elif weighting == "risk_scaled" and "market_beta" in sample.columns:
        beta = pd.to_numeric(sample.loc[idx, "market_beta"], errors="coerce").abs().replace(0, np.nan)
        raw = (1.0 / beta).astype("float64")
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(raw.median())
    else:
        raw = pd.Series(1.0, index=idx)
    if raw.sum() <= 0 or raw.isna().all():
        raw = pd.Series(1.0, index=idx)
    return (raw / raw.sum()).astype("float64")


def _liquidity_cost_bps(sample: pd.DataFrame, base_bps: float) -> pd.Series:
    permno = sample["permno"].to_numpy()
    if "dollar_volume" not in sample.columns or sample["dollar_volume"].notna().sum() < 10:
        return pd.Series(base_bps, index=permno)
    rank = sample["dollar_volume"].rank(pct=True)
    cost = pd.Series(base_bps, index=sample.index, dtype=float)
    cost.loc[rank <= 0.33] = base_bps * 1.75
    cost.loc[rank >= 0.67] = base_bps * 0.50
    return pd.Series(cost.to_numpy(), index=permno)


def construct_backtest(
    predictions_path: str | Path,
    out_dir: str | Path,
    model: str = "lightgbm",
    target_return_col: str = "next_month_excess_ret",
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
    one_way_cost_bps: float = 20.0,
    beta_neutral: bool = True,
    max_name_weight: float | None = 0.05,
    holding_buffer: float = 0.0,
    weighting: str = "equal",
) -> pd.DataFrame:
    preds = pd.read_parquet(predictions_path)
    preds = _coerce_numeric(preds, ["prediction", target_return_col, "excess_ret", "ret", "market_beta"])
    if model not in set(preds["model"]):
        model = sorted(preds["model"].unique())[0]
    df = preds[preds["model"] == model].copy()
    rows = []
    weights_out = []
    prev_w: pd.Series | None = None
    for month, g in df.groupby("month"):
        sample = g.dropna(subset=["prediction", target_return_col]).copy()
        if len(sample) < 20:
            continue
        hi = sample["prediction"].quantile(long_quantile)
        lo = sample["prediction"].quantile(short_quantile)
        long = sample["prediction"] >= hi
        short = sample["prediction"] <= lo
        if holding_buffer > 0 and prev_w is not None:
            lower_keep = max(0.0, long_quantile - holding_buffer)
            upper_keep = min(1.0, short_quantile + holding_buffer)
            keep_hi = sample["prediction"].quantile(lower_keep)
            keep_lo = sample["prediction"].quantile(upper_keep)
            prev_long = set(prev_w[prev_w > 0].index)
            prev_short = set(prev_w[prev_w < 0].index)
            permnos = sample["permno"].to_numpy()
            long = long | (pd.Series(permnos, index=sample.index).isin(prev_long) & (sample["prediction"] >= keep_hi))
            short = short | (pd.Series(permnos, index=sample.index).isin(prev_short) & (sample["prediction"] <= keep_lo))
            overlap = long & short
            if overlap.any():
                long.loc[overlap] = sample.loc[overlap, "prediction"] >= sample["prediction"].median()
                short.loc[overlap] = ~long.loc[overlap]
        w = pd.Series(0.0, index=sample.index)
        if long.sum() == 0 or short.sum() == 0:
            continue
        w.loc[long] = _initial_side_weights(sample, long, weighting)
        w.loc[short] = -_initial_side_weights(sample, short, weighting)
        w = _apply_abs_cap(w, max_name_weight)
        if beta_neutral and "market_beta" in sample.columns:
            w = _demean_beta(w, sample["market_beta"])
            w = _apply_abs_cap(w, max_name_weight)
        gross_ret = float((w * sample[target_return_col]).sum())
        beta_exposure = float((w * sample["market_beta"].fillna(0)).sum()) if "market_beta" in sample.columns else np.nan

        current = pd.Series(w.to_numpy(), index=sample["permno"].to_numpy())
        cost_map = _liquidity_cost_bps(sample, one_way_cost_bps)
        if prev_w is None:
            turnover = float(current.abs().sum())
            cost = float((current.abs() * cost_map.reindex(current.index).fillna(one_way_cost_bps)).sum() / 10000.0)
        else:
            aligned = pd.concat([prev_w.rename("prev"), current.rename("cur")], axis=1).fillna(0.0)
            turnover = float((aligned["cur"] - aligned["prev"]).abs().sum())
            trade = (aligned["cur"] - aligned["prev"]).abs()
            cost = float((trade * cost_map.reindex(aligned.index).fillna(one_way_cost_bps)).sum() / 10000.0)
        net_ret = gross_ret - cost
        rows.append(
            {
                "month": month,
                "model": model,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "cost": cost,
                "avg_trade_cost_bps": float(cost / turnover * 10000.0) if turnover else np.nan,
                "net_ret": net_ret,
                "n_long": int(long.sum()),
                "n_short": int(short.sum()),
                "beta_exposure": beta_exposure,
                "max_abs_weight": float(w.abs().max()),
                "weighting": weighting,
            }
        )
        tmp = sample[[c for c in ["permno", "month", "prediction", target_return_col, "market_cap", "dollar_volume"] if c in sample.columns]].copy()
        tmp["weight"] = w.to_numpy()
        weights_out.append(tmp)
        prev_w = current

    returns = pd.DataFrame(rows).sort_values("month")
    if returns.empty:
        raise ValueError("No backtest returns produced.")
    returns["gross_cum"] = (1 + returns["gross_ret"]).cumprod() - 1
    returns["net_cum"] = (1 + returns["net_ret"]).cumprod() - 1
    returns["net_drawdown"] = (1 + returns["net_cum"]) / (1 + returns["net_cum"]).cummax() - 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(out_dir / "portfolio_returns.parquet", index=False)
    pd.concat(weights_out, ignore_index=True).to_parquet(out_dir / "portfolio_weights.parquet", index=False)
    summary = summarize_returns(returns)
    summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    write_manifest(
        out_dir / "backtest_manifest.json",
        {
            "kind": "portfolio_backtest",
            "predictions": str(predictions_path),
            "model": model,
            "months": len(returns),
            "one_way_cost_bps": one_way_cost_bps,
            "beta_neutral": beta_neutral,
            "max_name_weight": max_name_weight,
            "holding_buffer": holding_buffer,
            "weighting": weighting,
        },
    )
    return returns


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["gross_ret", "net_ret"]:
        r = returns[col].dropna()
        downside = r[r < 0]
        vol = float(r.std(ddof=1))
        downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
        cum_col = "gross_cum" if col == "gross_ret" else "net_cum"
        drawdown = (1 + returns[cum_col]) / (1 + returns[cum_col]).cummax() - 1
        max_drawdown = float(drawdown.min())
        ann_return = float((1 + r.mean()) ** 12 - 1)
        rows.append(
            {
                "series": col,
                "mean_monthly": float(r.mean()),
                "vol_monthly": vol,
                "ann_return": ann_return,
                "ann_vol": float(vol * np.sqrt(12)),
                "sharpe": float(r.mean() / vol * np.sqrt(12)) if vol else np.nan,
                "sortino": float(r.mean() / downside_vol * np.sqrt(12)) if downside_vol else np.nan,
                "max_drawdown": max_drawdown,
                "calmar": float(ann_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
                "hit_rate": float((r > 0).mean()),
                "avg_turnover": float(returns["turnover"].mean()) if "turnover" in returns else np.nan,
                "avg_cost": float(returns["cost"].mean()) if "cost" in returns else np.nan,
                "avg_trade_cost_bps": float(returns["avg_trade_cost_bps"].mean())
                if "avg_trade_cost_bps" in returns
                else np.nan,
                "avg_abs_beta_exposure": float(returns["beta_exposure"].abs().mean()) if "beta_exposure" in returns else np.nan,
                "avg_max_abs_weight": float(returns["max_abs_weight"].mean()) if "max_abs_weight" in returns else np.nan,
                "months": int(r.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def implementation_scenarios(
    predictions_path: str | Path,
    out_dir: str | Path,
    model: str = "lightgbm",
    cost_bps: tuple[float, ...] = (10.0, 20.0, 35.0),
    weightings: tuple[str, ...] = ("equal", "value", "risk_scaled"),
    holding_buffer: float = 0.05,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows = []
    for weighting in weightings:
        for cost in cost_bps:
            scenario_dir = out_dir / f"scenario_{weighting}_{int(cost)}bps"
            try:
                construct_backtest(
                    predictions_path,
                    scenario_dir,
                    model=model,
                    one_way_cost_bps=cost,
                    holding_buffer=holding_buffer,
                    weighting=weighting,
                )
                summary = pd.read_csv(scenario_dir / "portfolio_summary.csv")
            except Exception as exc:
                rows.append(
                    {
                        "weighting": weighting,
                        "one_way_cost_bps": cost,
                        "series": "net_ret",
                        "status": f"failed: {type(exc).__name__}",
                    }
                )
                continue
            block = summary.copy()
            block["weighting"] = weighting
            block["one_way_cost_bps"] = cost
            block["status"] = "ok"
            rows.extend(block.to_dict("records"))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "implementation_scenarios.csv", index=False)
    write_manifest(
        out_dir / "implementation_scenarios_manifest.json",
        {
            "kind": "implementation_scenarios",
            "predictions": str(predictions_path),
            "model": model,
            "cost_bps": list(cost_bps),
            "weightings": list(weightings),
            "holding_buffer": holding_buffer,
        },
    )
    return result


def factor_alpha(returns_path: str | Path, factors_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    returns = pd.read_parquet(returns_path)
    factors = pd.read_parquet(factors_path)
    data = returns.merge(factors, on="month", how="inner")
    factor_cols = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd"] if c in data.columns]
    if not factor_cols:
        raise ValueError("No recognized factor columns found.")
    data = _coerce_numeric(data, ["net_ret", *factor_cols]).dropna(subset=["net_ret", *factor_cols])
    x = sm.add_constant(data[factor_cols].astype(float), has_constant="add")
    fit = sm.OLS(data["net_ret"].astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    summary = pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "se": fit.bse.values,
            "t_stat": fit.tvalues.values,
        }
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "factor_alpha.csv", index=False)
    write_manifest(
        out_dir / "factor_alpha_manifest.json",
        {
            "kind": "factor_alpha",
            "returns": str(returns_path),
            "factors": str(factors_path),
            "factor_cols": factor_cols,
            "months": int(fit.nobs),
        },
    )
    return summary
