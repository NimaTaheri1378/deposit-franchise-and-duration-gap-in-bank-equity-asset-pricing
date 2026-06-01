from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from deposit_duration.utils.manifest import write_manifest


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _robust_limits(series: pd.Series, low: float = 0.01, high: float = 0.99) -> tuple[float, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    lo, hi = clean.quantile([low, high]).to_numpy(dtype=float)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        span = max(abs(float(clean.median())), 1.0) * 0.1
        return float(clean.median() - span), float(clean.median() + span)
    pad = 0.04 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _return_norm(series: pd.Series) -> TwoSlopeNorm:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)
    hi = float(clean.abs().quantile(0.98))
    hi = max(hi, 0.01)
    return TwoSlopeNorm(vmin=-hi, vcenter=0, vmax=hi)


def _polish_axis(ax) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)


def _set_date_axis(ax, dates: pd.Series) -> None:
    clean = pd.to_datetime(dates).dropna()
    if clean.empty:
        return
    months = max(1, (clean.max().year - clean.min().year) * 12 + clean.max().month - clean.min().month + 1)
    if months > 84:
        locator = mdates.YearLocator(base=2)
    elif months > 30:
        locator = mdates.YearLocator()
    else:
        locator = mdates.MonthLocator(interval=max(1, int(np.ceil(months / 6))))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save_static(fig, static_dir: Path, stem: str) -> list[Path]:
    paths = [
        static_dir / f"{stem}.png",
        static_dir / f"{stem}.pdf",
        static_dir / f"{stem}.svg",
    ]
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    return paths


def build_figures(
    features_path: str | Path,
    predictions_path: str | Path,
    backtest_dir: str | Path,
    out_dir: str | Path,
    event_dir: str | Path | None = None,
    robustness_dir: str | Path | None = None,
) -> list[Path]:
    features = pd.read_parquet(features_path)
    predictions = pd.read_parquet(predictions_path)
    backtest_dir = Path(backtest_dir)
    returns = pd.read_parquet(backtest_dir / "portfolio_returns.parquet")
    features = _coerce_numeric(
        features,
        [
            "deposit_fragility_index",
            "duration_gap_proxy",
            "uninsured_deposit_share",
            "next_month_excess_ret",
        ],
    )
    predictions = _coerce_numeric(predictions, ["prediction", "next_month_excess_ret", "test_year"])
    returns = _coerce_numeric(
        returns, ["gross_ret", "net_ret", "gross_cum", "net_cum", "net_drawdown", "turnover", "cost"]
    )
    out_dir = Path(out_dir)
    event_dir = Path(event_dir) if event_dir else None
    robustness_dir = Path(robustness_dir) if robustness_dir else None
    static_dir = out_dir / "static"
    interactive_dir = out_dir / "interactive"
    static_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
        },
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(returns["month"], returns["net_cum"], label="Net cumulative return", color="#1b6f5a", linewidth=2.2)
    ax.fill_between(returns["month"], returns["net_drawdown"], 0, color="#b23a48", alpha=0.18, label="Drawdown")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Bank Fragility Strategy: Net Performance")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative return / drawdown")
    _set_date_axis(ax, returns["month"])
    ax.legend(loc="lower left", frameon=True, framealpha=0.94)
    _polish_axis(ax)
    paths = _save_static(fig, static_dir, "strategy_cumulative_return")
    plt.close(fig)
    made.extend(paths)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sc = ax.scatter(
        returns["turnover"],
        returns["net_ret"],
        c=returns["cost"],
        cmap="crest",
        s=42,
        alpha=0.80,
        linewidths=0.2,
        edgecolors="#ffffff",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Implementation Cost and Turnover")
    ax.set_xlabel("Monthly turnover")
    ax.set_ylabel("Net monthly return")
    _polish_axis(ax)
    fig.colorbar(sc, ax=ax, label="Transaction cost", shrink=0.9, pad=0.02)
    paths = _save_static(fig, static_dir, "turnover_cost_frontier")
    plt.close(fig)
    made.extend(paths)

    latest_month = features["month"].max()
    latest = features[features["month"] == latest_month].dropna(
        subset=["deposit_fragility_index", "duration_gap_proxy", "next_month_excess_ret"]
    ).copy()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    if len(latest) >= 25:
        latest["fragility_bucket"] = pd.qcut(
            latest["deposit_fragility_index"].rank(method="first"), 10, labels=False, duplicates="drop"
        )
        latest["duration_bucket"] = pd.qcut(
            latest["duration_gap_proxy"].rank(method="first"), 10, labels=False, duplicates="drop"
        )
        heat = latest.pivot_table(
            index="fragility_bucket", columns="duration_bucket", values="next_month_excess_ret", aggfunc="mean"
        )
        heat = pd.DataFrame(heat.to_numpy(dtype=float, na_value=np.nan), index=heat.index, columns=heat.columns)
        sns.heatmap(
            heat,
            ax=ax,
            cmap="vlag",
            center=0,
            mask=heat.isna(),
            linewidths=0.35,
            linecolor="#f4f4f4",
            cbar_kws={"label": "Mean next-month excess return", "shrink": 0.86},
        )
        ax.set_xlabel("Duration gap bucket")
        ax.set_ylabel("Deposit fragility bucket")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
    else:
        ax.text(0.5, 0.5, "Insufficient latest-month observations", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(f"Fragility x Duration Heatmap ({latest_month:%Y-%m})")
    paths = _save_static(fig, static_dir, "fragility_duration_heatmap")
    plt.close(fig)
    made.extend(paths)

    sample = features.dropna(subset=["duration_gap_proxy", "uninsured_deposit_share", "next_month_excess_ret"])
    if len(sample) > 100:
        sample = sample.sample(min(20000, len(sample)), random_state=1378)
    xlim = _robust_limits(sample["duration_gap_proxy"])
    ylim = (0.0, min(1.0, _robust_limits(sample["uninsured_deposit_share"], 0.005, 0.995)[1]))
    norm = _return_norm(sample["next_month_excess_ret"])
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    sc = ax.scatter(
        sample["duration_gap_proxy"],
        sample["uninsured_deposit_share"],
        c=sample["next_month_excess_ret"],
        s=10,
        alpha=0.48,
        cmap="vlag",
        norm=norm,
        linewidths=0,
    )
    ax.set_title("Next-Month Return Across Duration and Uninsured Deposits")
    ax.set_xlabel("Duration gap proxy")
    ax.set_ylabel("Uninsured deposit share")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _polish_axis(ax)
    fig.colorbar(sc, ax=ax, label="Next-month excess return", shrink=0.92, pad=0.02)
    paths = _save_static(fig, static_dir, "duration_uninsured_contour_proxy")
    plt.close(fig)
    made.extend(paths)

    pred_model = "lightgbm" if "lightgbm" in set(predictions["model"]) else predictions["model"].iloc[0]
    p = predictions[predictions["model"] == pred_model].dropna(subset=["prediction", "next_month_excess_ret"]).copy()
    p["prediction_decile"] = p.groupby("month")["prediction"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )
    deciles = (
        p.groupby("prediction_decile", observed=True)["next_month_excess_ret"]
        .mean()
        .rename("mean_next_month_excess_ret")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = ["#6f8fb7" if v < 0 else "#1b6f5a" for v in deciles["mean_next_month_excess_ret"]]
    ax.bar(deciles["prediction_decile"].astype(int), deciles["mean_next_month_excess_ret"], color=colors, width=0.72)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Realized Return by Prediction Decile ({pred_model})")
    ax.set_xlabel("Prediction decile")
    ax.set_ylabel("Mean next-month excess return")
    ax.set_xticks(deciles["prediction_decile"].astype(int))
    _polish_axis(ax)
    paths = _save_static(fig, static_dir, "prediction_decile_returns")
    plt.close(fig)
    made.extend(paths)

    if event_dir and (event_dir / "event_car.csv").exists():
        cars = pd.read_csv(event_dir / "event_car.csv")
        cars = _coerce_numeric(cars, ["relative_month", "car", "mean_excess_ret"])
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for group, block in cars.sort_values("relative_month").groupby("event_group", observed=True):
            label = str(group).replace("_", " ")
            color = "#b23a48" if "high" in label else "#1b6f5a"
            ax.plot(block["relative_month"], block["car"], marker="o", linewidth=1.9, label=label, color=color)
        ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title("March 2023 Event-Time Cumulative Abnormal Return")
        ax.set_xlabel("Months from event")
        ax.set_ylabel("Cumulative excess return")
        ax.legend(loc="best", frameon=True, framealpha=0.94)
        _polish_axis(ax)
        paths = _save_static(fig, static_dir, "march_2023_event_car")
        plt.close(fig)
        made.extend(paths)

    if event_dir and (event_dir / "local_projection_summary.csv").exists():
        lp = pd.read_csv(event_dir / "local_projection_summary.csv")
        lp = _coerce_numeric(lp, ["horizon", "coef", "se"])
        lp = lp[lp["term"] == "exposure_x_shock"].sort_values("horizon")
        if not lp.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            lower = lp["coef"] - 1.96 * lp["se"]
            upper = lp["coef"] + 1.96 * lp["se"]
            ax.plot(lp["horizon"], lp["coef"], color="#1b6f5a", marker="o", linewidth=2.0)
            ax.fill_between(lp["horizon"], lower, upper, color="#1b6f5a", alpha=0.18)
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_title("Rate-Shock Local Projection")
            ax.set_xlabel("Horizon in months")
            ax.set_ylabel("Interaction coefficient")
            ax.set_xticks(lp["horizon"].astype(int))
            _polish_axis(ax)
            paths = _save_static(fig, static_dir, "local_projection_rate_shock")
            plt.close(fig)
            made.extend(paths)

    if robustness_dir and (robustness_dir / "double_sort_mean_returns.csv").exists():
        double_sort = pd.read_csv(robustness_dir / "double_sort_mean_returns.csv", index_col=0)
        double_sort = double_sort.apply(pd.to_numeric, errors="coerce")
        fig, ax = plt.subplots(figsize=(7.8, 5.8))
        sns.heatmap(
            double_sort,
            ax=ax,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2%",
            linewidths=0.35,
            linecolor="#f4f4f4",
            cbar_kws={"label": "Mean next-month excess return", "shrink": 0.84},
        )
        ax.set_title("Double Sort: Fragility x Duration")
        ax.set_xlabel("Duration gap quintile")
        ax.set_ylabel("Deposit fragility quintile")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        paths = _save_static(fig, static_dir, "double_sort_fragility_duration")
        plt.close(fig)
        made.extend(paths)

    if robustness_dir and (robustness_dir / "state_dependence_summary.csv").exists():
        states = pd.read_csv(robustness_dir / "state_dependence_summary.csv")
        states = _coerce_numeric(states, ["mean_monthly_spread", "nw_se", "t_stat"])
        states = states.dropna(subset=["mean_monthly_spread"])
        if not states.empty:
            fig, ax = plt.subplots(figsize=(8.6, 5.0))
            order = states.sort_values("mean_monthly_spread")["state"]
            tmp = states.set_index("state").loc[order].reset_index()
            colors = ["#b23a48" if v < 0 else "#1b6f5a" for v in tmp["mean_monthly_spread"]]
            ax.barh(tmp["state"].str.replace("_", " "), tmp["mean_monthly_spread"], color=colors)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_title("State-Dependent Signal Spread")
            ax.set_xlabel("Mean monthly high-minus-low return")
            ax.set_ylabel("")
            _polish_axis(ax)
            paths = _save_static(fig, static_dir, "state_dependent_spreads")
            plt.close(fig)
            made.extend(paths)

    if robustness_dir and (robustness_dir / "placebo_signal_summary.csv").exists():
        placebo = pd.read_csv(robustness_dir / "placebo_signal_summary.csv")
        placebo = _coerce_numeric(placebo, ["placebo_mean_spread", "realized_mean_spread"])
        placebo = placebo.dropna(subset=["placebo_mean_spread"])
        if not placebo.empty:
            realized = float(placebo["realized_mean_spread"].iloc[0])
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            ax.hist(placebo["placebo_mean_spread"], bins=24, color="#6f8fb7", alpha=0.72)
            ax.axvline(realized, color="#b23a48", linewidth=2.0, label="realized")
            ax.axvline(-realized, color="#b23a48", linewidth=1.2, linestyle="--", alpha=0.75)
            ax.set_title("Placebo Signal Test")
            ax.set_xlabel("Placebo high-minus-low return")
            ax.set_ylabel("Count")
            ax.legend(loc="best", frameon=True, framealpha=0.94)
            _polish_axis(ax)
            paths = _save_static(fig, static_dir, "placebo_signal_test")
            plt.close(fig)
            made.extend(paths)

    alpha_path = backtest_dir / "factor_alpha.csv"
    if alpha_path.exists():
        alpha = pd.read_csv(alpha_path)
        alpha = _coerce_numeric(alpha, ["coef", "se", "t_stat"])
        alpha = alpha.dropna(subset=["coef"])
        if not alpha.empty:
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            labels = alpha["term"].astype(str).str.replace("const", "alpha", regex=False)
            colors = ["#1b6f5a" if v >= 0 else "#b23a48" for v in alpha["coef"]]
            ax.barh(labels, alpha["coef"], color=colors)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_title("Factor Alpha Attribution")
            ax.set_xlabel("Monthly coefficient")
            ax.set_ylabel("")
            _polish_axis(ax)
            paths = _save_static(fig, static_dir, "factor_alpha_attribution")
            plt.close(fig)
            made.extend(paths)

    p["rank"] = p.groupby("month")["prediction"].rank(pct=True)
    fig = px.scatter(
        p.sample(min(len(p), 25000), random_state=1378),
        x="prediction",
        y="next_month_excess_ret",
        color="test_year",
        opacity=0.45,
        title=f"Predicted vs Realized Returns ({pred_model})",
    )
    path = interactive_dir / "prediction_scatter.html"
    fig.write_html(path, include_plotlyjs="cdn")
    made.append(path)

    fig = px.line(
        returns,
        x="month",
        y=["gross_cum", "net_cum"],
        title="Gross and Net Cumulative Returns",
    )
    path = interactive_dir / "strategy_cumulative_return.html"
    fig.write_html(path, include_plotlyjs="cdn")
    made.append(path)

    latest_for_scenario = features[features["month"] == latest_month].dropna(
        subset=["duration_gap_proxy", "deposit_fragility_index", "uninsured_deposit_share"]
    ).copy()
    if not latest_for_scenario.empty:
        scenario_rows = []
        for shock_bp in [0, 25, 50, 100]:
            block = latest_for_scenario.copy()
            block["rate_shock_bp"] = shock_bp
            block["scenario_vulnerability"] = (
                block["duration_gap_proxy"] * block["uninsured_deposit_share"] * shock_bp / 100
                + 0.05 * block["deposit_fragility_index"]
            )
            scenario_rows.append(block)
        scenario = pd.concat(scenario_rows, ignore_index=True)
        fig = px.scatter(
            scenario,
            x="duration_gap_proxy",
            y="deposit_fragility_index",
            color="scenario_vulnerability",
            animation_frame="rate_shock_bp",
            hover_data=["permno", "uninsured_deposit_share"],
            color_continuous_scale="RdBu_r",
            title="Scenario Explorer: Rate Shock Vulnerability",
        )
        path = interactive_dir / "scenario_rate_shock_explorer.html"
        fig.write_html(path, include_plotlyjs="cdn")
        made.append(path)

    write_manifest(
        out_dir / "figures_manifest.json",
        {
            "kind": "figures",
            "features": str(features_path),
            "predictions": str(predictions_path),
            "event_dir": str(event_dir) if event_dir else None,
            "robustness_dir": str(robustness_dir) if robustness_dir else None,
            "files": [str(p) for p in made],
        },
    )
    return made
