from __future__ import annotations

import argparse
from pathlib import Path

from deposit_duration.backtest.portfolio import construct_backtest, factor_alpha, implementation_scenarios
from deposit_duration.data.live_extract import extract_live_raw
from deposit_duration.data.schema_audit import run_schema_audit
from deposit_duration.data.synthetic import make_synthetic_raw
from deposit_duration.features.build import build_features
from deposit_duration.models.econometrics import fama_macbeth, portfolio_sorts
from deposit_duration.models.event_study import run_event_identification
from deposit_duration.models.interpretability import run_interpretability
from deposit_duration.models.ipca import run_ipca_layer
from deposit_duration.models.robustness import lag_variant_robustness, run_robustness
from deposit_duration.models.train import train_walk_forward
from deposit_duration.utils.config import project_config
from deposit_duration.utils.logging import setup_logging
from deposit_duration.visuals.figures import build_figures
from deposit_duration.visuals.qa import audit_static_figures, build_contact_sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddgap")
    parser.add_argument("--config", default="configs/project.yml")
    parser.add_argument("--log-file", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("schema-audit")
    p.add_argument("--out-yaml", default="configs/schema_map.yml")
    p.add_argument("--manifest", default="data_manifest/schema_audit.json")

    p = sub.add_parser("synthetic")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1378)

    p = sub.add_parser("features")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lag-days", type=int, default=None)

    p = sub.add_parser("extract-live")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p = sub.add_parser("full-live")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--first-test-year", type=int, default=None)
    p.add_argument("--last-test-year", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--tuning-trials", type=int, default=None)

    p = sub.add_parser("econometrics")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)

    p = sub.add_parser("events")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)

    p = sub.add_parser("robustness")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-placebos", type=int, default=100)

    p = sub.add_parser("lag-robustness")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lags", default="30,45,60")

    p = sub.add_parser("train")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--first-test-year", type=int, default=None)
    p.add_argument("--last-test-year", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--tuning-trials", type=int, default=None)

    p = sub.add_parser("interpret")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)

    p = sub.add_parser("ipca")
    p.add_argument("--features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-components", type=int, default=3)

    p = sub.add_parser("backtest")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="lightgbm")
    p.add_argument("--weighting", choices=["equal", "value", "risk_scaled"], default="equal")
    p.add_argument("--holding-buffer", type=float, default=0.0)
    p.add_argument("--one-way-cost-bps", type=float, default=20.0)
    p.add_argument("--max-name-weight", type=float, default=0.05)

    p = sub.add_parser("implementation-scenarios")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="lightgbm")

    p = sub.add_parser("factor-alpha")
    p.add_argument("--returns", required=True)
    p.add_argument("--factors", required=True)
    p.add_argument("--out-dir", required=True)

    p = sub.add_parser("figures")
    p.add_argument("--features", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--backtest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--event-dir", default=None)
    p.add_argument("--robustness-dir", default=None)

    p = sub.add_parser("visual-audit")
    p.add_argument("--figures-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--contact-sheet", default=None)

    p = sub.add_parser("smoke")
    p.add_argument("--out-dir", default="artifacts/smoke")
    p.add_argument("--use-gpu", action="store_true")

    args = parser.parse_args(argv)
    logger = setup_logging(args.log_file)
    cfg = project_config(args.config)

    if args.cmd == "schema-audit":
        result = run_schema_audit(args.config, args.out_yaml, args.manifest)
        logger.info("Schema audit complete: %s", args.out_yaml)
        logger.info("Available target libraries: %s", [k for k, v in result["libraries"].items() if v["available"]])
        return 0

    if args.cmd == "synthetic":
        make_synthetic_raw(args.out_dir, seed=args.seed)
        logger.info("Synthetic raw data written under %s", args.out_dir)
        return 0

    if args.cmd == "features":
        lag_days = args.lag_days or int(cfg["project"]["public_information_lag_days"])
        build_features(args.raw_dir, args.out, lag_days=lag_days)
        logger.info("Feature store written: %s", args.out)
        return 0

    if args.cmd == "extract-live":
        start = args.start or cfg["project"]["sample_start"]
        end = args.end or cfg["project"]["main_sample_end"]
        extract_live_raw(args.out_dir, start=start, end=end)
        logger.info("Live raw extract written under %s", args.out_dir)
        return 0

    if args.cmd == "econometrics":
        fama_macbeth(args.features, args.out_dir)
        portfolio_sorts(args.features, args.out_dir)
        logger.info("Econometric outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "events":
        run_event_identification(args.features, args.out_dir)
        logger.info("Event-identification outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "robustness":
        run_robustness(args.features, args.out_dir, n_placebos=args.n_placebos)
        logger.info("Robustness outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "lag-robustness":
        lags = tuple(int(x.strip()) for x in args.lags.split(",") if x.strip())
        lag_variant_robustness(args.raw_dir, args.out_dir, lags=lags)
        logger.info("Lag-variant robustness outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "train":
        modeling = cfg["modeling"]
        train_walk_forward(
            args.features,
            args.out_dir,
            first_test_year=args.first_test_year or int(modeling["first_test_year"]),
            last_test_year=args.last_test_year or int(modeling["last_test_year"]),
            use_gpu=args.use_gpu or bool(modeling.get("use_gpu", False)),
            seed=int(modeling.get("random_seed", 1378)),
            validation_years=int(modeling.get("validation_years", 2)),
            tuning_trials=args.tuning_trials
            if args.tuning_trials is not None
            else int(modeling.get("tuning_trials", 8)),
        )
        logger.info("Model predictions written: %s", Path(args.out_dir) / "predictions.parquet")
        return 0

    if args.cmd == "interpret":
        run_interpretability(args.features, args.out_dir)
        logger.info("Interpretability outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "ipca":
        run_ipca_layer(args.features, args.out_dir, n_components=args.n_components)
        logger.info("IPCA/PCA interpretation outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "backtest":
        construct_backtest(
            args.predictions,
            args.out_dir,
            model=args.model,
            one_way_cost_bps=args.one_way_cost_bps,
            max_name_weight=args.max_name_weight,
            holding_buffer=args.holding_buffer,
            weighting=args.weighting,
        )
        logger.info("Backtest outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "implementation-scenarios":
        implementation_scenarios(args.predictions, args.out_dir, model=args.model)
        logger.info("Implementation-scenario outputs written: %s", args.out_dir)
        return 0

    if args.cmd == "factor-alpha":
        factor_alpha(args.returns, args.factors, args.out_dir)
        logger.info("Factor alpha written: %s", Path(args.out_dir) / "factor_alpha.csv")
        return 0

    if args.cmd == "figures":
        build_figures(
            args.features,
            args.predictions,
            args.backtest_dir,
            args.out_dir,
            event_dir=args.event_dir,
            robustness_dir=args.robustness_dir,
        )
        logger.info("Figures written: %s", args.out_dir)
        return 0

    if args.cmd == "visual-audit":
        audit = audit_static_figures(args.figures_dir, args.out_dir)
        if args.contact_sheet:
            build_contact_sheet(args.figures_dir, args.contact_sheet)
        logger.info("Visual audit written: %s (%s/%s pass)", args.out_dir, int(audit["pass"].sum()), len(audit))
        return 0

    if args.cmd == "smoke":
        out_dir = Path(args.out_dir)
        make_synthetic_raw(out_dir)
        features = out_dir / "features.parquet"
        build_features(out_dir / "raw", features)
        econ_dir = out_dir / "econometrics"
        fama_macbeth(features, econ_dir)
        portfolio_sorts(features, econ_dir)
        event_dir = out_dir / "events"
        run_event_identification(features, event_dir)
        robustness_dir = out_dir / "robustness"
        run_robustness(features, robustness_dir, n_placebos=25)
        lag_variant_robustness(out_dir / "raw", out_dir / "lag_robustness")
        model_dir = out_dir / "models"
        train_walk_forward(
            features,
            model_dir,
            first_test_year=2019,
            last_test_year=2021,
            use_gpu=args.use_gpu,
            tuning_trials=3,
        )
        interpret_dir = out_dir / "interpretability"
        run_interpretability(features, interpret_dir, sample_size=2500, shap_sample_size=600)
        run_ipca_layer(features, out_dir / "ipca")
        backtest_dir = out_dir / "backtest"
        construct_backtest(model_dir / "predictions.parquet", backtest_dir)
        implementation_scenarios(model_dir / "predictions.parquet", out_dir / "implementation_scenarios")
        factor_alpha(backtest_dir / "portfolio_returns.parquet", out_dir / "raw" / "factors_monthly.parquet", backtest_dir)
        build_figures(
            features,
            model_dir / "predictions.parquet",
            backtest_dir,
            out_dir / "figures",
            event_dir=event_dir,
            robustness_dir=robustness_dir,
        )
        audit_static_figures(out_dir / "figures" / "static", out_dir / "figures")
        build_contact_sheet(out_dir / "figures" / "static", out_dir / "figures" / "static_contact_sheet.png")
        logger.info("Smoke pipeline complete: %s", out_dir)
        return 0

    if args.cmd == "full-live":
        start = args.start or cfg["project"]["sample_start"]
        end = args.end or cfg["project"]["main_sample_end"]
        out_dir = Path(args.out_dir)
        if not args.skip_extract:
            extract_live_raw(out_dir, start=start, end=end)
        elif not (out_dir / "raw").exists():
            raise FileNotFoundError(f"Cannot skip extraction because raw directory is missing: {out_dir / 'raw'}")
        features = out_dir / "features.parquet"
        lag_days = int(cfg["project"]["public_information_lag_days"])
        build_features(out_dir / "raw", features, lag_days=lag_days)
        econ_dir = out_dir / "econometrics"
        fama_macbeth(features, econ_dir)
        portfolio_sorts(features, econ_dir)
        event_dir = out_dir / "events"
        run_event_identification(features, event_dir)
        robustness_dir = out_dir / "robustness"
        run_robustness(features, robustness_dir)
        lag_variant_robustness(out_dir / "raw", out_dir / "lag_robustness")
        model_dir = out_dir / "models"
        modeling = cfg["modeling"]
        train_walk_forward(
            features,
            model_dir,
            first_test_year=args.first_test_year or int(modeling["first_test_year"]),
            last_test_year=args.last_test_year or int(modeling["last_test_year"]),
            use_gpu=args.use_gpu or bool(modeling.get("use_gpu", False)),
            seed=int(modeling.get("random_seed", 1378)),
            validation_years=int(modeling.get("validation_years", 2)),
            tuning_trials=args.tuning_trials
            if args.tuning_trials is not None
            else int(modeling.get("tuning_trials", 8)),
        )
        interpret_dir = out_dir / "interpretability"
        run_interpretability(features, interpret_dir, seed=int(modeling.get("random_seed", 1378)))
        run_ipca_layer(features, out_dir / "ipca")
        backtest_dir = out_dir / "backtest"
        construct_backtest(model_dir / "predictions.parquet", backtest_dir)
        implementation_scenarios(model_dir / "predictions.parquet", out_dir / "implementation_scenarios")
        factor_alpha(backtest_dir / "portfolio_returns.parquet", out_dir / "raw" / "factors_monthly.parquet", backtest_dir)
        build_figures(
            features,
            model_dir / "predictions.parquet",
            backtest_dir,
            out_dir / "figures",
            event_dir=event_dir,
            robustness_dir=robustness_dir,
        )
        audit_static_figures(out_dir / "figures" / "static", out_dir / "figures")
        build_contact_sheet(out_dir / "figures" / "static", out_dir / "figures" / "static_contact_sheet.png")
        logger.info("Full live pipeline complete: %s", out_dir)
        return 0

    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
