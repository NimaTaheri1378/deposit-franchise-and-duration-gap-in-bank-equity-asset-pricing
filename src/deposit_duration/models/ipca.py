from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

from deposit_duration.utils.manifest import write_manifest


def run_ipca_layer(
    features_path: str | Path,
    out_dir: str | Path,
    return_col: str = "next_month_excess_ret",
    n_components: int = 3,
) -> dict[str, str]:
    df = pd.read_parquet(features_path)
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    panel = df.pivot_table(index="month", columns="permno", values=return_col, aggfunc="mean").sort_index()
    panel = panel.dropna(axis=1, thresh=max(24, int(0.25 * len(panel))))
    panel = panel.sub(panel.mean(axis=1), axis=0)
    panel = panel.fillna(0.0)
    if panel.shape[0] < 24 or panel.shape[1] < n_components + 2:
        raise ValueError("Not enough bank-return panel coverage for the PCA/IPCA interpretation layer.")

    pca = PCA(n_components=n_components, random_state=1378)
    factors = pca.fit_transform(panel.to_numpy())
    factor_df = pd.DataFrame(
        factors,
        columns=[f"bank_pca_factor_{idx + 1}" for idx in range(n_components)],
    )
    factor_df.insert(0, "month", panel.index)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"bank_pca_loading_{idx + 1}" for idx in range(n_components)],
    )
    loadings.insert(0, "permno", panel.columns.to_numpy())

    char_cols = [
        c
        for c in [
            "duration_gap_proxy",
            "deposit_fragility_index",
            "duration_x_uninsured",
            "log_market_cap",
            "capital_ratio",
            "cre_to_assets",
            "market_beta",
        ]
        if c in df.columns
    ]
    chars = df.sort_values("month").groupby("permno", as_index=False)[char_cols].last()
    exposure_panel = loadings.merge(chars, on="permno", how="left")
    rows = []
    for loading_col in [c for c in loadings.columns if c.startswith("bank_pca_loading_")]:
        sample = exposure_panel[[loading_col, *char_cols]].replace([np.inf, -np.inf], np.nan)
        sample = sample.apply(pd.to_numeric, errors="coerce").dropna().astype("float64")
        if len(sample) < len(char_cols) + 10:
            continue
        x = sm.add_constant(sample[char_cols], has_constant="add")
        fit = sm.OLS(sample[loading_col], x).fit(cov_type="HC1")
        for term, coef in fit.params.items():
            rows.append(
                {
                    "loading": loading_col,
                    "term": term,
                    "coef": float(coef),
                    "se": float(fit.bse[term]),
                    "t_stat": float(fit.tvalues[term]),
                    "nobs": int(fit.nobs),
                }
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "factors": out_dir / "bank_pca_factors.csv",
        "loadings": out_dir / "bank_pca_loadings.csv",
        "loading_characteristics": out_dir / "loading_characteristics.csv",
        "explained_variance": out_dir / "explained_variance.csv",
    }
    factor_df.to_csv(paths["factors"], index=False)
    loadings.to_csv(paths["loadings"], index=False)
    pd.DataFrame(rows).to_csv(paths["loading_characteristics"], index=False)
    pd.DataFrame(
        {
            "component": [idx + 1 for idx in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(paths["explained_variance"], index=False)
    write_manifest(
        out_dir / "ipca_manifest.json",
        {
            "kind": "pca_ipca_interpretation_layer",
            "features": str(features_path),
            "return_col": return_col,
            "n_components": n_components,
            "months": int(panel.shape[0]),
            "permnos": int(panel.shape[1]),
            "outputs": {k: str(v) for k, v in paths.items()},
        },
    )
    return {k: str(v) for k, v in paths.items()}
