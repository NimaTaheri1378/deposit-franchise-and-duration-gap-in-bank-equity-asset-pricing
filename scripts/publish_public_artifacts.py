from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px


STATIC_STEMS = [
    "double_sort_fragility_duration",
    "duration_uninsured_contour_proxy",
    "factor_alpha_attribution",
    "fragility_duration_heatmap",
    "local_projection_rate_shock",
    "march_2023_event_car",
    "placebo_signal_test",
    "prediction_decile_returns",
    "state_dependent_spreads",
    "strategy_cumulative_return",
    "turnover_cost_frontier",
]

INTERPRETABILITY_STEMS = [
    "partial_dependence_core",
    "partial_dependence_duration_uninsured",
    "permutation_importance",
    "shap_duration_uninsured",
    "shap_interaction_duration_uninsured",
    "shap_summary",
]


def _copy_static_pack(artifacts: Path, static_out: Path) -> None:
    static_src = artifacts / "figures" / "static"
    for stem in STATIC_STEMS:
        for ext in ["png", "pdf", "svg"]:
            src = static_src / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, static_out / f"{stem}.{ext}")

    interpret_src = artifacts / "interpretability"
    for stem in INTERPRETABILITY_STEMS:
        for ext in ["png", "pdf", "svg"]:
            src = interpret_src / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, static_out / f"interpretability_{stem}.{ext}")

    for ext in ["png", "pdf", "svg"]:
        src = static_src / f"duration_uninsured_contour_proxy.{ext}"
        if src.exists():
            shutil.copy2(src, static_out / f"hero_figure.{ext}")


def _write_public_strategy(artifacts: Path, interactive_out: Path) -> None:
    returns_path = artifacts / "backtest" / "portfolio_returns.parquet"
    if not returns_path.exists():
        return
    returns = pd.read_parquet(returns_path)
    keep = ["month", "gross_cum", "net_cum", "turnover", "cost"]
    returns = returns[[c for c in keep if c in returns.columns]].copy()
    returns["month"] = pd.to_datetime(returns["month"])
    long = returns.melt("month", value_vars=["gross_cum", "net_cum"], var_name="series", value_name="cumulative_return")
    fig = px.line(
        long,
        x="month",
        y="cumulative_return",
        color="series",
        title="Aggregate Bank-Fragility Strategy Performance",
        labels={"month": "Month", "cumulative_return": "Cumulative return", "series": "Series"},
    )
    fig.write_html(interactive_out / "strategy_cumulative_return.html", include_plotlyjs="cdn")


def _write_public_scenario(artifacts: Path, interactive_out: Path) -> None:
    features_path = artifacts / "features.parquet"
    if not features_path.exists():
        return
    cols = ["month", "duration_gap_proxy", "deposit_fragility_index", "uninsured_deposit_share"]
    features = pd.read_parquet(features_path, columns=cols)
    features["month"] = pd.to_datetime(features["month"])
    latest = features[features["month"] == features["month"].max()].dropna().copy()
    if len(latest) < 25:
        return
    latest["duration_bin"] = pd.qcut(latest["duration_gap_proxy"], 8, labels=False, duplicates="drop")
    latest["fragility_bin"] = pd.qcut(latest["deposit_fragility_index"], 8, labels=False, duplicates="drop")
    grouped = (
        latest.groupby(["duration_bin", "fragility_bin"], observed=True)
        .agg(
            mean_duration_gap=("duration_gap_proxy", "mean"),
            mean_fragility=("deposit_fragility_index", "mean"),
            mean_uninsured_share=("uninsured_deposit_share", "mean"),
            bank_count=("duration_gap_proxy", "size"),
        )
        .reset_index()
    )
    grouped["scenario_vulnerability"] = (
        grouped["mean_duration_gap"].rank(pct=True)
        * grouped["mean_fragility"].rank(pct=True)
        * (1 + grouped["mean_uninsured_share"])
    )
    fig = px.scatter(
        grouped,
        x="mean_duration_gap",
        y="mean_fragility",
        size="bank_count",
        color="scenario_vulnerability",
        color_continuous_scale="RdBu_r",
        title="Aggregate Rate-Shock Scenario Explorer",
        labels={
            "mean_duration_gap": "Mean duration gap proxy",
            "mean_fragility": "Mean deposit fragility",
            "bank_count": "Banks in bin",
            "scenario_vulnerability": "Scenario vulnerability",
        },
        hover_data={
            "mean_duration_gap": ":.3f",
            "mean_fragility": ":.3f",
            "mean_uninsured_share": ":.3f",
            "bank_count": True,
            "duration_bin": False,
            "fragility_bin": False,
        },
    )
    fig.write_html(interactive_out / "scenario_rate_shock_explorer.html", include_plotlyjs="cdn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/full_live")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    figures = Path(args.figures_dir)
    static_out = figures / "static"
    interactive_out = figures / "interactive"
    static_out.mkdir(parents=True, exist_ok=True)
    interactive_out.mkdir(parents=True, exist_ok=True)

    _copy_static_pack(artifacts, static_out)
    _write_public_scenario(artifacts, interactive_out)
    _write_public_strategy(artifacts, interactive_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
