from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deposit_duration.models.econometrics import model_feature_columns
from deposit_duration.utils.manifest import write_manifest


def _year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.year


def _spearman(y: pd.Series, yhat: pd.Series) -> float:
    if len(y) < 3:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(yhat), method="spearman"))


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _fit_elastic_net(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    seed: int,
    alpha: float = 0.001,
    l1_ratio: float = 0.2,
) -> Pipeline:
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "enet",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=5000),
            ),
        ]
    )
    model.fit(train[features], train[target])
    return model


def _fit_lightgbm(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    use_gpu: bool,
    seed: int,
    params_override: dict | None = None,
):
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        return None
    params = {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.001,
        "reg_lambda": 0.01,
        "random_state": seed,
        "verbosity": -1,
    }
    if params_override:
        params.update(params_override)
    if use_gpu:
        params["device_type"] = "gpu"
    model = lgb.LGBMRegressor(**params)
    try:
        model.fit(train[features], train[target])
    except Exception:
        if use_gpu:
            params.pop("device_type", None)
            model = lgb.LGBMRegressor(**params)
            model.fit(train[features], train[target])
        else:
            raise
    return model


def _validation_split(df: pd.DataFrame, test_year: int, validation_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = test_year - validation_years
    train = df[df["year"] < validation_start].copy()
    valid = df[(df["year"] >= validation_start) & (df["year"] < test_year)].copy()
    return train, valid


def _score_validation(y: pd.Series, pred: np.ndarray) -> float:
    ic = _spearman(y, pd.Series(pred, index=y.index))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    if not np.isfinite(ic):
        ic = -1.0
    return float(ic - 0.01 * rmse)


def _tune_elastic_net(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    seed: int,
) -> dict:
    if len(train) < 200 or len(valid) < 50:
        return {"alpha": 0.001, "l1_ratio": 0.2}
    best_score = -np.inf
    best = {"alpha": 0.001, "l1_ratio": 0.2}
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
        for l1_ratio in [0.1, 0.3, 0.7]:
            model = _fit_elastic_net(train, features, target, seed, alpha=alpha, l1_ratio=l1_ratio)
            pred = model.predict(valid[features])
            score = _score_validation(valid[target], pred)
            if score > best_score:
                best_score = score
                best = {"alpha": alpha, "l1_ratio": l1_ratio}
    return best


def _fallback_lgbm_candidates() -> list[dict]:
    return [
        {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 50},
        {"num_leaves": 63, "learning_rate": 0.025, "min_child_samples": 75},
        {"num_leaves": 127, "learning_rate": 0.015, "min_child_samples": 100},
        {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 150, "reg_lambda": 0.1},
    ]


def _tune_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    use_gpu: bool,
    seed: int,
    tuning_trials: int,
) -> dict:
    if len(train) < 500 or len(valid) < 50 or tuning_trials <= 0:
        return {}
    try:
        import optuna  # type: ignore

        def objective(trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
                "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
                "n_estimators": 250,
            }
            model = _fit_lightgbm(train, features, target, use_gpu=use_gpu, seed=seed, params_override=params)
            if model is None:
                return -np.inf
            pred = model.predict(valid[features])
            return _score_validation(valid[target], pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)
        params = dict(study.best_params)
        params["n_estimators"] = 500
        return params
    except Exception:
        best_score = -np.inf
        best: dict = {}
        for params in _fallback_lgbm_candidates()[: max(1, min(tuning_trials, 4))]:
            model = _fit_lightgbm(train, features, target, use_gpu=False, seed=seed, params_override=params)
            if model is None:
                continue
            pred = model.predict(valid[features])
            score = _score_validation(valid[target], pred)
            if score > best_score:
                best_score = score
                best = params
        return best


def _fit_ft_transformer(train: pd.DataFrame, features: list[str], target: str, use_gpu: bool, seed: int):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    x_df = train[features].replace([np.inf, -np.inf], np.nan)
    med = x_df.median().fillna(0)
    scale = x_df.std(ddof=0).replace(0, 1).fillna(1)
    x = ((x_df.fillna(med) - med) / scale).to_numpy(dtype=np.float32)
    y = train[target].to_numpy(dtype=np.float32).reshape(-1, 1)

    class FTTransformerRegressor(nn.Module):
        def __init__(self, n_features: int, token_dim: int = 64, depth: int = 3, heads: int = 4):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, token_dim))
            self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
            nn.init.xavier_uniform_(self.weight)
            self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=heads,
                dim_feedforward=token_dim * 4,
                dropout=0.10,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 64),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(64, 1),
            )

        def forward(self, x_in):
            tokens = x_in.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            cls = self.cls.expand(x_in.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0])

    model = FTTransformerRegressor(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(y, device=device)
    model.train()
    epochs = 60 if len(train) > 5000 else 25
    batch = min(4096, max(256, len(train) // 8))
    for _ in range(epochs):
        order = torch.randperm(len(xt), device=device)
        for idx in order.split(batch):
            pred = model(xt[idx])
            loss = loss_fn(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    class TorchWrapper:
        def __init__(self, fitted, medians, scales, cols, run_device):
            self.fitted = fitted
            self.medians = medians
            self.scales = scales
            self.cols = cols
            self.device = run_device

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            x_local = frame[self.cols].replace([np.inf, -np.inf], np.nan)
            x_local = ((x_local.fillna(self.medians) - self.medians) / self.scales).to_numpy(dtype=np.float32)
            self.fitted.eval()
            with torch.no_grad():
                out = self.fitted(torch.tensor(x_local, device=self.device)).cpu().numpy().ravel()
            return out

    return TorchWrapper(model, med, scale, features, device)


def train_walk_forward(
    features_path: str | Path,
    out_dir: str | Path,
    first_test_year: int = 2017,
    last_test_year: int = 2025,
    target: str = "next_month_excess_ret",
    use_gpu: bool = False,
    seed: int = 1378,
    validation_years: int = 2,
    tuning_trials: int = 8,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _coerce_numeric(df, [target])
    df = df.dropna(subset=[target])
    df["year"] = _year(df["month"])
    features = model_feature_columns(df)
    if not features:
        raise ValueError("No model features found.")
    df = _coerce_numeric(df, [target, "ret", "excess_ret", "market_beta", *features])
    features = [c for c in features if not df[c].isna().all()]
    if not features:
        raise ValueError("All model features are missing after numeric coercion.")

    all_preds = []
    metrics = []
    tuning_rows = []
    for test_year in range(first_test_year, last_test_year + 1):
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        if len(train) < 200 or len(test) < 20:
            continue

        tune_train, valid = _validation_split(df, test_year, validation_years)
        enet_params = _tune_elastic_net(tune_train, valid, features, target, seed)
        lgbm_params = _tune_lightgbm(
            tune_train,
            valid,
            features,
            target,
            use_gpu=use_gpu,
            seed=seed,
            tuning_trials=tuning_trials,
        )
        tuning_rows.append(
            {
                "test_year": test_year,
                "validation_years": validation_years,
                "elastic_net_params": enet_params,
                "lightgbm_params": lgbm_params,
                "validation_rows": len(valid),
                "tuning_rows": len(tune_train),
            }
        )

        models = {"elastic_net": _fit_elastic_net(train, features, target, seed, **enet_params)}
        lgbm = _fit_lightgbm(train, features, target, use_gpu=use_gpu, seed=seed, params_override=lgbm_params)
        if lgbm is not None:
            models["lightgbm"] = lgbm
        torch_model = _fit_ft_transformer(train, features, target, use_gpu=use_gpu, seed=seed)
        if torch_model is not None:
            models["ft_transformer"] = torch_model

        for model_name, model in models.items():
            pred = model.predict(test[features])
            block_cols = [
                c
                for c in [
                    "permno",
                    "month",
                    target,
                    "excess_ret",
                    "ret",
                    "market_beta",
                    "market_cap",
                    "dollar_volume",
                    "amihud",
                    "turnover",
                ]
                if c in test.columns
            ]
            block = test[block_cols].copy()
            block["model"] = model_name
            block["prediction"] = pred
            block["test_year"] = test_year
            all_preds.append(block)
            pearson_ic = float(pd.Series(block[target]).corr(pd.Series(block["prediction"]), method="pearson"))
            hit_rate = float(((block[target] >= 0) == (block["prediction"] >= 0)).mean())
            metrics.append(
                {
                    "model": model_name,
                    "test_year": test_year,
                    "nobs": len(block),
                    "r2": float(r2_score(block[target], pred)),
                    "rmse": float(np.sqrt(mean_squared_error(block[target], pred))),
                    "spearman_ic": _spearman(block[target], block["prediction"]),
                    "pearson_ic": pearson_ic,
                    "hit_rate": hit_rate,
                }
            )

    if not all_preds:
        raise ValueError("No walk-forward predictions were produced.")
    predictions = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    metrics_df.to_csv(out_dir / "walk_forward_metrics.csv", index=False)
    tuning_df = pd.DataFrame(tuning_rows)
    if not tuning_df.empty:
        tuning_df.to_json(out_dir / "walk_forward_tuning.json", orient="records", indent=2)
    write_manifest(
        out_dir / "model_manifest.json",
        {
            "kind": "walk_forward_models",
            "features": str(features_path),
            "feature_columns": features,
            "target": target,
            "first_test_year": first_test_year,
            "last_test_year": last_test_year,
            "models": sorted(predictions["model"].unique().tolist()),
            "rows": len(predictions),
            "use_gpu_requested": use_gpu,
            "deep_model": "ft_transformer",
            "validation_years": validation_years,
            "tuning_trials": tuning_trials,
            "tuning_log": str(out_dir / "walk_forward_tuning.json"),
        },
    )
    return predictions
