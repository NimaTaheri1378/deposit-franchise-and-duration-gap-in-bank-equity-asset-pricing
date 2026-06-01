from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_summary(artifacts_dir: Path) -> dict:
    feature_manifest = json.loads((artifacts_dir / "features.manifest.json").read_text())
    extract_path = artifacts_dir / "live_extract_manifest.json"
    extract_manifest = json.loads(extract_path.read_text()) if extract_path.exists() else {}
    visual_audit = _read_csv(artifacts_dir / "figures" / "visual_audit.csv")
    metrics = _read_csv(artifacts_dir / "models" / "walk_forward_metrics.csv")
    if metrics.empty and (artifacts_dir / "walk_forward_metrics.csv").exists():
        metrics = _read_csv(artifacts_dir / "walk_forward_metrics.csv")
    portfolio = _read_csv(artifacts_dir / "backtest" / "portfolio_summary.csv")
    fm = _read_csv(artifacts_dir / "econometrics" / "fama_macbeth_summary.csv")
    robustness = _read_csv(artifacts_dir / "robustness" / "signal_spread_summary.csv")
    large_liquid_fm = _read_csv(
        artifacts_dir / "public_gates" / "large_liquid" / "large_liquid_fama_macbeth_summary.csv"
    )
    large_liquid_spreads = _read_csv(
        artifacts_dir / "public_gates" / "large_liquid" / "large_liquid_signal_spread_summary.csv"
    )
    incremental_prediction = _read_csv(
        artifacts_dir / "public_gates" / "incremental_prediction" / "incremental_prediction_summary.csv"
    )

    model_metrics = {}
    if not metrics.empty:
        grouped = metrics.groupby("model", observed=True)["spearman_ic"]
        model_metrics = {
            name: {
                "mean_spearman_ic": float(group.mean()),
                "median_spearman_ic": float(group.median()),
                "test_years": int(metrics.loc[metrics["model"] == name, "test_year"].nunique()),
            }
            for name, group in grouped
        }

    return {
        "kind": "public_result_summary",
        "feature_store": {
            "rows": int(feature_manifest["rows"]),
            "permnos": int(feature_manifest["permnos"]),
            "month_min": feature_manifest["month_min"],
            "month_max": feature_manifest["month_max"],
            "lag_days": int(feature_manifest["lag_days"]),
        },
        "extract_rows": extract_manifest.get("rows", {}),
        "wrds_tables": extract_manifest.get("wrds_tables", []),
        "visual_audit": {
            "figures": int(len(visual_audit)),
            "passed": int(visual_audit["pass"].sum()) if "pass" in visual_audit else 0,
        },
        "model_metrics": model_metrics,
        "portfolio_summary": portfolio.to_dict("records"),
        "fama_macbeth": fm.to_dict("records"),
        "signal_spread_robustness": robustness.to_dict("records"),
        "public_push_gates": {
            "large_liquid_fama_macbeth": large_liquid_fm.to_dict("records"),
            "large_liquid_signal_spreads": large_liquid_spreads.to_dict("records"),
            "incremental_prediction": incremental_prediction.to_dict("records"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = build_summary(Path(args.artifacts_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
