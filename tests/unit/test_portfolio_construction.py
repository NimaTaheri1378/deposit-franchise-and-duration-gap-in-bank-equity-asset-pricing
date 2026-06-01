from __future__ import annotations

import numpy as np
import pandas as pd

from deposit_duration.backtest.portfolio import construct_backtest


def test_backtest_weights_are_long_short_and_finite(tmp_path):
    months = pd.period_range("2020-01", "2021-12", freq="M").to_timestamp("M")
    rows = []
    for month in months:
        for i in range(40):
            rows.append(
                {
                    "permno": 10000 + i,
                    "month": month,
                    "model": "elastic_net",
                    "prediction": i / 40 + np.random.default_rng(i).normal(0, 0.001),
                    "next_month_excess_ret": (i - 20) / 1000,
                    "excess_ret": 0.0,
                    "ret": 0.0,
                    "market_beta": 1.0 + i / 100,
                    "test_year": month.year,
                }
            )
    pred_path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(pred_path, index=False)
    returns = construct_backtest(pred_path, tmp_path / "bt", model="elastic_net")
    weights = pd.read_parquet(tmp_path / "bt" / "portfolio_weights.parquet")
    assert returns["net_ret"].notna().all()
    assert np.isfinite(returns["turnover"]).all()
    by_month = weights.groupby("month")["weight"].sum().abs()
    assert (by_month < 1e-8).all()

