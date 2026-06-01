from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from deposit_duration.backtest.portfolio import construct_backtest
from deposit_duration.backtest.portfolio import implementation_scenarios
from deposit_duration.data.synthetic import make_synthetic_raw
from deposit_duration.features.build import build_features
from deposit_duration.models.econometrics import fama_macbeth, portfolio_sorts
from deposit_duration.models.ipca import run_ipca_layer
from deposit_duration.models.public_gates import run_public_push_gates
from deposit_duration.models.robustness import lag_variant_robustness
from deposit_duration.models.train import train_walk_forward
from deposit_duration.visuals.figures import build_figures
from deposit_duration.visuals.qa import audit_static_figures, build_contact_sheet


def test_synthetic_end_to_end_pipeline(tmp_path: Path):
    make_synthetic_raw(tmp_path)
    features_path = tmp_path / "features.parquet"
    panel = build_features(tmp_path / "raw", features_path)
    assert {"duration_gap_proxy", "deposit_fragility_index", "next_month_excess_ret"}.issubset(panel.columns)

    econ_dir = tmp_path / "econometrics"
    fm = fama_macbeth(features_path, econ_dir, min_obs_per_month=20)
    assert not fm.empty
    sorts = portfolio_sorts(features_path, econ_dir)
    assert not sorts.empty
    lag_summary = lag_variant_robustness(tmp_path / "raw", tmp_path / "lag_robustness", lags=(30, 45))
    assert set(lag_summary["lag_days"]) == {30, 45}
    public_gates = run_public_push_gates(
        features_path,
        tmp_path / "public_gates",
        first_test_year=2019,
        last_test_year=2020,
    )
    assert all(Path(path).exists() for path in public_gates.values())

    model_dir = tmp_path / "models"
    train_walk_forward(features_path, model_dir, first_test_year=2019, last_test_year=2020, tuning_trials=2)
    preds = pd.read_parquet(model_dir / "predictions.parquet")
    assert not preds.empty
    assert (model_dir / "walk_forward_tuning.json").exists()
    ipca_outputs = run_ipca_layer(features_path, tmp_path / "ipca", n_components=2)
    assert all(Path(path).exists() for path in ipca_outputs.values())

    bt_dir = tmp_path / "backtest"
    rets = construct_backtest(model_dir / "predictions.parquet", bt_dir, model="elastic_net")
    assert not rets.empty
    scenarios = implementation_scenarios(
        model_dir / "predictions.parquet",
        tmp_path / "implementation_scenarios",
        model="elastic_net",
        cost_bps=(10.0,),
        weightings=("equal", "risk_scaled"),
    )
    assert set(scenarios["weighting"].dropna()) == {"equal", "risk_scaled"}

    figure_paths = build_figures(features_path, model_dir / "predictions.parquet", bt_dir, tmp_path / "figures")
    assert len(figure_paths) >= 3
    assert all(path.exists() for path in figure_paths)
    static_pngs = [path for path in figure_paths if path.suffix == ".png"]
    assert static_pngs
    for path in static_pngs:
        image = mpimg.imread(path)
        assert image.shape[0] >= 600
        assert image.shape[1] >= 800
        assert np.nanstd(image) > 0.01
    audit = audit_static_figures(tmp_path / "figures", tmp_path / "figures")
    assert audit["pass"].all()
    contact_sheet = build_contact_sheet(tmp_path / "figures", tmp_path / "figures" / "contact_sheet.png")
    assert contact_sheet.exists()
