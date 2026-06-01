from __future__ import annotations

import pandas as pd

from deposit_duration.features.build import build_features
from deposit_duration.data.synthetic import make_synthetic_raw


def test_quarterly_features_obey_public_information_lag(tmp_path):
    make_synthetic_raw(tmp_path)
    panel = build_features(tmp_path / "raw", tmp_path / "features.parquet", lag_days=45)
    sample = panel[["month", "report_date", "available_date"]].dropna()
    assert not sample.empty
    assert (pd.to_datetime(sample["available_date"]) <= pd.to_datetime(sample["month"])).all()
    assert ((pd.to_datetime(sample["available_date"]) - pd.to_datetime(sample["report_date"])).dt.days == 45).all()

