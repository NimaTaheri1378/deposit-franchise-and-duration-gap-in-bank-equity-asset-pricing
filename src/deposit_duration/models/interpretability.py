from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from deposit_duration.models.econometrics import model_feature_columns
from deposit_duration.utils.manifest import write_manifest


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _save_current_figure(out_dir: Path, stem: str) -> list[str]:
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf", out_dir / f"{stem}.svg"]
    plt.tight_layout()
    plt.savefig(paths[0], dpi=180, bbox_inches="tight")
    plt.savefig(paths[1], bbox_inches="tight")
    plt.savefig(paths[2], bbox_inches="tight")
    plt.close()
    return [str(p) for p in paths]


def _polish_axis(ax) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)


def _fit_interpretable_model(x: pd.DataFrame, y: pd.Series, seed: int):
    try:
        import lightgbm as lgb  # type: ignore

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(x, y)
        return model, "lightgbm"
    except Exception:
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.035,
                        l2_regularization=0.01,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x, y)
        return model, "hist_gradient_boosting"


def _feature_family(feature: str) -> str:
    if feature.endswith("_rank"):
        feature = feature[: -len("_rank")]
    if any(token in feature for token in ["duration", "htm", "afs", "securities", "hedge", "unrealized"]):
        return "duration_and_hedging"
    if any(token in feature for token in ["deposit", "fragility", "liability"]):
        return "deposit_franchise"
    if any(token in feature for token in ["capital", "cre", "loan"]):
        return "capital_and_loan_book"
    if any(token in feature for token in ["rate", "term", "vix"]):
        return "macro_regime"
    if any(token in feature for token in ["market", "turnover", "amihud"]):
        return "market_implementation"
    return "other"


def _feature_label(feature: str) -> str:
    labels = {
        "duration_gap_proxy": "Duration gap proxy",
        "uninsured_deposit_share": "Uninsured deposit share",
        "vix": "VIX",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def _family_permutation_importance(model, x: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_pred = model.predict(x)
    baseline_rmse = float(np.sqrt(np.mean((y.to_numpy() - baseline_pred) ** 2)))
    rows = []
    families = pd.Series({_feature: _feature_family(_feature) for _feature in x.columns})
    for family, cols in families.groupby(families).groups.items():
        shuffled = x.copy()
        for col in cols:
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        pred = model.predict(shuffled)
        rmse = float(np.sqrt(np.mean((y.to_numpy() - pred) ** 2)))
        rows.append(
            {
                "family": family,
                "feature_count": len(cols),
                "baseline_rmse": baseline_rmse,
                "family_permutation_rmse": rmse,
                "rmse_increase": rmse - baseline_rmse,
                "features": ",".join(cols),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False)


def _partial_dependence_figures(model, x: pd.DataFrame, out_dir: Path) -> list[str]:
    figure_files: list[str] = []
    candidates = [c for c in ["duration_gap_proxy", "uninsured_deposit_share", "vix"] if c in x.columns]
    if candidates:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        display = PartialDependenceDisplay.from_estimator(model, x, candidates[:2], ax=ax)
        fig.suptitle("Partial Dependence: Core Bank-Fragility Features")
        axes = np.asarray(display.axes_).ravel()
        for axis, feature in zip(axes, candidates[:2], strict=False):
            axis.set_xlabel(_feature_label(feature))
            axis.set_ylabel("Partial dependence")
        figure_files.extend(_save_current_figure(out_dir, "partial_dependence_core"))
    if {"duration_gap_proxy", "uninsured_deposit_share"}.issubset(x.columns):
        feature_pair = ("duration_gap_proxy", "uninsured_deposit_share")
        feature_indices = tuple(x.columns.get_loc(c) for c in feature_pair)
        pd_result = partial_dependence(model, x, [feature_indices], grid_resolution=36)
        average = np.asarray(pd_result["average"])[0]
        grid_values = pd_result.get("grid_values", pd_result.get("values"))
        x_grid = np.asarray(grid_values[0], dtype=float)
        y_grid = np.asarray(grid_values[1], dtype=float)

        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        vmax = float(np.nanmax(np.abs(average))) if np.isfinite(average).any() else 1.0
        mesh = ax.pcolormesh(
            x_grid,
            y_grid,
            average.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title("Partial Dependence: Duration x Uninsured Deposits")
        ax.set_xlabel(_feature_label(feature_pair[0]))
        ax.set_ylabel(_feature_label(feature_pair[1]))
        fig.colorbar(mesh, ax=ax, label="Predicted next-month excess return", shrink=0.9, pad=0.02)
        _polish_axis(ax)
        figure_files.extend(_save_current_figure(out_dir, "partial_dependence_duration_uninsured"))
    return figure_files


def run_interpretability(
    features_path: str | Path,
    out_dir: str | Path,
    target: str = "next_month_excess_ret",
    sample_size: int = 5000,
    shap_sample_size: int = 1200,
    seed: int = 1378,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 15,
        },
    )
    features = model_feature_columns(df)
    if not features:
        raise ValueError("No model features found for interpretability.")
    df = _coerce_numeric(df, [target, *features]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])
    features = [c for c in features if c in df.columns and not df[c].isna().all()]
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
    x = df[features]
    y = df[target]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_name = _fit_interpretable_model(x, y, seed)
    perm = permutation_importance(
        model,
        x,
        y,
        n_repeats=5,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "permutation_importance": perm.importances_mean,
            "permutation_importance_sd": perm.importances_std,
            "model": model_name,
        }
    ).sort_values("permutation_importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    family_importance = _family_permutation_importance(model, x, y, seed)
    family_importance.to_csv(out_dir / "family_importance.csv", index=False)

    top = importance.head(15).sort_values("permutation_importance")
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.barh(top["feature"], top["permutation_importance"], color="#1b6f5a")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Increase in RMSE when permuted")
    ax.set_ylabel("")
    _polish_axis(ax)
    figure_files = _save_current_figure(out_dir, "permutation_importance")
    figure_files.extend(_partial_dependence_figures(model, x, out_dir))

    shap_status = "not_run"
    if model_name == "lightgbm":
        try:
            import shap  # type: ignore

            shap_frame = x.sample(min(len(x), shap_sample_size), random_state=seed)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_frame)
            shap_array = np.asarray(shap_values)
            mean_abs = pd.DataFrame(
                {"feature": features, "mean_abs_shap": np.abs(shap_array).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False)
            mean_abs.to_csv(out_dir / "shap_importance.csv", index=False)

            shap.summary_plot(shap_array, shap_frame, show=False, max_display=20)
            figure_files.extend(_save_current_figure(out_dir, "shap_summary"))

            if "duration_gap_proxy" in shap_frame.columns and "uninsured_deposit_share" in shap_frame.columns:
                idx = features.index("duration_gap_proxy")
                fig, ax = plt.subplots(figsize=(8.2, 5.4))
                sc = ax.scatter(
                    shap_frame["duration_gap_proxy"],
                    shap_array[:, idx],
                    c=shap_frame["uninsured_deposit_share"],
                    cmap="viridis",
                    s=12,
                    alpha=0.65,
                    linewidths=0,
                )
                ax.axhline(0, color="#333333", linewidth=0.8)
                ax.set_title("Duration Exposure SHAP Response")
                ax.set_xlabel("Duration gap proxy")
                ax.set_ylabel("SHAP value")
                fig.colorbar(sc, ax=ax, label="Uninsured deposit share", shrink=0.9, pad=0.02)
                _polish_axis(ax)
                figure_files.extend(_save_current_figure(out_dir, "shap_duration_uninsured"))
                try:
                    interaction_frame = shap_frame.sample(min(len(shap_frame), 250), random_state=seed)
                    interactions = np.asarray(explainer.shap_interaction_values(interaction_frame))
                    if interactions.ndim == 3:
                        jdx = features.index("uninsured_deposit_share")
                        interaction_values = interactions[:, idx, jdx]
                        pd.DataFrame(
                            {
                                "duration_gap_proxy": interaction_frame["duration_gap_proxy"].to_numpy(),
                                "uninsured_deposit_share": interaction_frame["uninsured_deposit_share"].to_numpy(),
                                "shap_interaction": interaction_values,
                            }
                        ).to_csv(out_dir / "shap_interaction_duration_uninsured.csv", index=False)
                        fig, ax = plt.subplots(figsize=(8.2, 5.4))
                        sc = ax.scatter(
                            interaction_frame["duration_gap_proxy"],
                            interaction_values,
                            c=interaction_frame["uninsured_deposit_share"],
                            cmap="viridis",
                            s=12,
                            alpha=0.65,
                            linewidths=0,
                        )
                        ax.axhline(0, color="#333333", linewidth=0.8)
                        ax.set_title("SHAP Interaction: Duration x Uninsured Deposits")
                        ax.set_xlabel("Duration gap proxy")
                        ax.set_ylabel("Interaction SHAP value")
                        fig.colorbar(sc, ax=ax, label="Uninsured deposit share", shrink=0.9, pad=0.02)
                        _polish_axis(ax)
                        figure_files.extend(_save_current_figure(out_dir, "shap_interaction_duration_uninsured"))
                except Exception:
                    pass
            shap_status = "complete"
        except Exception as exc:
            shap_status = f"failed: {type(exc).__name__}"

    write_manifest(
        out_dir / "interpretability_manifest.json",
        {
            "kind": "interpretability",
            "features": str(features_path),
            "target": target,
            "model": model_name,
            "rows": len(df),
            "feature_count": len(features),
            "shap_status": shap_status,
            "figure_files": figure_files,
            "family_importance": str(out_dir / "family_importance.csv"),
        },
    )
    return importance
