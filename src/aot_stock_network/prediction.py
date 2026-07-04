"""
prediction.py — AOT Closing Price Prediction & Model Comparison
=================================================================

Compares 8 model families for one-step-ahead monthly forecasting:

  ML models (multivariate):
    1. Linear Regression     — sklearn baseline
    2. Random Forest         — bagging ensemble
    3. XGBoost               — gradient boosting (tree)
    4. LightGBM              — gradient boosting (leaf-wise)
    5. CatBoost              — gradient boosting (ordered)
    6. LSTM                  — recurrent neural network (TensorFlow / Keras)

  Time-series models (univariate, target-only):
    7. ARIMA                 — autoregressive integrated moving average
    8. Prophet               — additive model with seasonality (Meta)

Pipeline
---------
  1. Chronological split: train / validation / test
  2. Feature scaling (standardisation) for ML models
  3. Hyperparameter tuning via grid/random search (TimeSeriesSplit CV)
  4. Model comparison on validation RMSE  →  best model auto-selected
  5. Final evaluation on held-out test set
  6. Publication-quality figures (SVG export)

Evaluation
-----------
  RMSE  — Root Mean Squared Error (primary)
  MAE   — Mean Absolute Error
  MAPE  — Mean Absolute Percentage Error
  R²    — Coefficient of Determination

Figures
--------
  • Residual Plot          — residuals vs predicted + histogram
  • Prediction Plot        — actual vs predicted time series + scatter
  • Feature Importance     — bar chart (tree models); coefficients (LR)
  • SHAP Summary           — SHAP beeswarm / bar plot
  • Model Comparison       — side-by-side bar chart of all metrics

Usage
-----
    from aot_stock_network.prediction import PredictionPipeline

    pipe = PredictionPipeline(
        df=feature_dataset,
        target_col="aot_close",
        test_months=12,
        val_months=6,
    )
    results = pipe.run(models=["rf", "xgb", "lstm"])
    print(results.summary())
    results.plot_all(output_dir="reports/figures")
"""

from __future__ import annotations

import itertools
import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prophet as pr

# ── SHAP ───────────────────────────────────────────────────
import shap

# ── tree boosters ──────────────────────────────────────────
import xgboost as xgb
from matplotlib.figure import Figure
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor

# ── sklearn ────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ── time series ────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA as ARIMA_MODEL

import importlib

warnings.filterwarnings("ignore", category=FutureWarning)

_keras = None
_kl = None

def _lazy_keras():
    global _keras, _kl
    if _keras is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        tf = importlib.import_module("tensorflow")
        _keras = tf.keras
        importlib.import_module("tensorflow.keras.layers")
        _kl = tf.keras.layers
    return _keras, _kl

logger = logging.getLogger("prediction")

PUBLICATION_COLORS: Dict[str, str] = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#C44E52",
    "pred": "#55A868",
    "actual": "#212121",
    "residual": "#8172B3",
    "bg": "#FAFAFA",
    "text": "#212121",
}

# ══════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@dataclass
class EvalMetrics:
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
        return EvalMetrics(
            rmse=rmse(y_true, y_pred),
            mae=float(mean_absolute_error(y_true, y_pred)),
            mape=mape(y_true, y_pred),
            r2=float(r2_score(y_true, y_pred)),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return f"RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  MAPE={self.mape:.2f}%  R²={self.r2:.4f}"


# ══════════════════════════════════════════════════════════════
# Hyper-parameter grids per model
# ══════════════════════════════════════════════════════════════

LINEAR_REGRESSION_GRID = [{}]  # no hyper-params

RANDOM_FOREST_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5],
}

XGBOOST_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
}

LIGHTGBM_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [8, 16, 31],
    "subsample": [0.8, 1.0],
}

CATBOOST_GRID = {
    "iterations": [100, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [3, 5, 7],
    "l2_leaf_reg": [1, 3, 5],
}

ARIMA_GRID = {
    "p": [0, 1, 2, 3],
    "d": [0, 1],
    "q": [0, 1, 2, 3],
}

PROPHET_GRID = {
    "changepoint_prior_scale": [0.01, 0.05, 0.5],
    "seasonality_prior_scale": [0.01, 0.1, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
}

LSTM_GRID = {
    "units": [32, 64, 128],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [0.001, 0.01],
    "epochs": [50, 100],
}

MODEL_GRIDS: Dict[str, Dict[str, Any]] = {
    "lr": LINEAR_REGRESSION_GRID,
    "rf": RANDOM_FOREST_GRID,
    "xgb": XGBOOST_GRID,
    "lgbm": LIGHTGBM_GRID,
    "cb": CATBOOST_GRID,
    "arima": ARIMA_GRID,
    "prophet": PROPHET_GRID,
    "lstm": LSTM_GRID,
}

MODEL_LABELS: Dict[str, str] = {
    "lr": "Linear Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "cb": "CatBoost",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "lstm": "LSTM",
}


# ══════════════════════════════════════════════════════════════
# Base model wrapper
# ══════════════════════════════════════════════════════════════


class _BaseModelWrapper(ABC):
    """Consistent interface for all 8 model families."""

    name: str = ""
    is_univariate: bool = False  # True for ARIMA, Prophet

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "_BaseModelWrapper": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, Any],
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """Default hyper-parameter tuning (override in subclasses)."""
        return {}


# ── 1. Linear Regression ──────────────────────────────────


class _LRWrapper(_BaseModelWrapper):
    name = "lr"

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(len(self._model.coef_))],
                "importance": np.abs(self._model.coef_),
                "direction": np.sign(self._model.coef_),
            }
        )

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        return {}


# ── 2. Random Forest ──────────────────────────────────────


class _RFWrapper(_BaseModelWrapper):
    name = "rf"

    def __init__(self, **kwargs):
        self._params = kwargs
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._model = RandomForestRegressor(
            **self._params,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        return pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(len(self._model.feature_importances_))],
                "importance": self._model.feature_importances_,
            }
        )

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        if not param_grid:
            return {}
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 5))
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info("  RF best: %s (RMSE=%.4f)", gs.best_params_, -gs.best_score_)
        return gs.best_params_


# ── 3. XGBoost ────────────────────────────────────────────


class _XGBWrapper(_BaseModelWrapper):
    name = "xgb"

    def __init__(self, **kwargs):
        self._params = kwargs
        self._model: Optional[xgb.XGBRegressor] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        evals = None
        if X_val is not None and y_val is not None:
            evals = [(X_val, y_val)]
        self._model = xgb.XGBRegressor(
            **self._params,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric="rmse",
            verbosity=0,
            enable_categorical=False,
        )
        self._model.fit(X_train, y_train, eval_set=evals, verbose=False)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        return pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(len(imp))],
                "importance": imp,
            }
        )

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        if not param_grid:
            return {}
        n = len(X_train)
        tscv = TimeSeriesSplit(n_splits=min(3, n // 5))
        gs = GridSearchCV(
            xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                eval_metric="rmse",
                verbosity=0,
                enable_categorical=False,
            ),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info("  XGB best: %s (RMSE=%.4f)", gs.best_params_, -gs.best_score_)
        return gs.best_params_


# ── 4. LightGBM ───────────────────────────────────────────


class _LGBMWrapper(_BaseModelWrapper):
    name = "lgbm"

    def __init__(self, **kwargs):
        self._params = kwargs
        self._model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        self._model = lgb.LGBMRegressor(
            **self._params,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
            ],
        )
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        return pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(len(imp))],
                "importance": imp,
            }
        )

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        if not param_grid:
            return {}
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 5))
        gs = GridSearchCV(
            lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info("  LGBM best: %s (RMSE=%.4f)", gs.best_params_, -gs.best_score_)
        return gs.best_params_


# ── 5. CatBoost ───────────────────────────────────────────


class _CBWrapper(_BaseModelWrapper):
    name = "cb"

    def __init__(self, **kwargs):
        self._params = kwargs
        self._model: Optional[cb.CatBoostRegressor] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        self._model = cb.CatBoostRegressor(
            **self._params,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=20,
        )
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        imp = self._model.get_feature_importance()
        return pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(len(imp))],
                "importance": imp,
            }
        )

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        if not param_grid:
            return {}
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 5))
        gs = GridSearchCV(
            cb.CatBoostRegressor(random_seed=42, verbose=0),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info("  CB best: %s (RMSE=%.4f)", gs.best_params_, -gs.best_score_)
        return gs.best_params_


# ── 6. ARIMA ──────────────────────────────────────────────


class _ARIMAWrapper(_BaseModelWrapper):
    name = "arima"
    is_univariate = True

    def __init__(self, **kwargs):
        self._order = (kwargs.get("p", 1), kwargs.get("d", 1), kwargs.get("q", 1))
        self._model: Optional[ARIMA_MODEL] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._y_train = y_train
        self._model = ARIMA_MODEL(y_train, order=self._order)
        self._model = self._model.fit()
        return self

    def predict(self, X):
        n = len(X)
        if self._model is None:
            return np.zeros(n)
        # One-step-ahead forecasts for the requested period
        pred = self._model.forecast(steps=n)
        return np.asarray(pred).ravel()

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        best_rmse = np.inf
        best_params = {}
        p_vals = param_grid.get("p", [1])
        d_vals = param_grid.get("d", [1])
        q_vals = param_grid.get("q", [1])
        n = len(y_train)
        split = int(n * 0.8)
        y_tr, y_va = y_train[:split], y_train[split:]
        for p, d, q in itertools.product(p_vals, d_vals, q_vals):
            try:
                m = ARIMA_MODEL(y_tr, order=(p, d, q)).fit()
                pred = m.forecast(steps=len(y_va))
                err = rmse(y_va, np.asarray(pred).ravel())
                if err < best_rmse:
                    best_rmse = err
                    best_params = {"p": p, "d": d, "q": q}
            except Exception:
                continue
        logger.info("  ARIMA best: %s (RMSE=%.4f)", best_params, best_rmse)
        return best_params


# ── 7. Prophet ────────────────────────────────────────────


class _ProphetWrapper(_BaseModelWrapper):
    name = "prophet"
    is_univariate = True

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._model: Optional[pr.Prophet] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        df = pd.DataFrame(
            {"ds": pd.date_range("2000-01-01", periods=len(y_train), freq="ME"), "y": y_train}
        )
        self._model = pr.Prophet(**self._kwargs)
        self._model.fit(df)
        return self

    def predict(self, X):
        n = len(X)
        future = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=n, freq="ME")})
        fcst = self._model.predict(future)
        return fcst["yhat"].values

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        best_rmse = np.inf
        best_params = {}
        n = len(y_train)
        split = int(n * 0.8)
        y_tr, y_va = y_train[:split], y_train[split:]

        cps = param_grid.get("changepoint_prior_scale", [0.05])
        sps = param_grid.get("seasonality_prior_scale", [0.1])
        sm_list = param_grid.get("seasonality_mode", ["additive"])

        for cp, sp, sm in itertools.product(cps, sps, sm_list):
            try:
                df_tr = pd.DataFrame(
                    {"ds": pd.date_range("2000-01-01", periods=len(y_tr), freq="ME"), "y": y_tr}
                )
                m = pr.Prophet(
                    changepoint_prior_scale=cp, seasonality_prior_scale=sp, seasonality_mode=sm
                )
                m.fit(df_tr)
                future = pd.DataFrame(
                    {"ds": pd.date_range("2000-01-01", periods=len(y_va), freq="ME")}
                )
                fcst = m.predict(future)
                err = rmse(y_va, fcst["yhat"].values)
                if err < best_rmse:
                    best_rmse = err
                    best_params = {
                        "changepoint_prior_scale": cp,
                        "seasonality_prior_scale": sp,
                        "seasonality_mode": sm,
                    }
            except Exception:
                continue
        logger.info("  Prophet best: %s (RMSE=%.4f)", best_params, best_rmse)
        return best_params


# ── 8. LSTM ───────────────────────────────────────────────


class _LSTMWrapper(_BaseModelWrapper):
    name = "lstm"

    def __init__(self, **kwargs):
        self._params = kwargs
        self._model = None
        self._scaler_y: Optional[StandardScaler] = None
        self._lookback: int = 6
        self._batch_size: int = 16

    def _build_model(self, input_dim: int):
        _k, _l = _lazy_keras()
        units = self._params.get("units", 64)
        dropout = self._params.get("dropout", 0.2)
        lr = self._params.get("learning_rate", 0.001)

        model = _k.Sequential(
            [
                _l.LSTM(
                    units,
                    return_sequences=True,
                    dropout=dropout,
                    input_shape=(self._lookback, input_dim),
                ),
                _l.LSTM(max(units // 2, 8), dropout=dropout),
                _l.Dense(16, activation="relu"),
                _l.Dense(1),
            ]
        )
        model.compile(
            optimizer=_k.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(self._lookback, len(X)):
            Xs.append(X[i - self._lookback : i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._scaler_y = StandardScaler()
        y_scaled = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        X_seq, y_seq = self._create_sequences(X_train, y_scaled)
        input_dim = X_train.shape[1]

        self._model = self._build_model(input_dim)

        val_data = None
        if X_val is not None and y_val is not None and len(X_val) > self._lookback:
            yv_scaled = self._scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            Xv_seq, yv_seq = self._create_sequences(X_val, yv_scaled)
            if len(Xv_seq) > 0:
                val_data = (Xv_seq, yv_seq)

        _k, _ = _lazy_keras()
        callbacks = [
            _k.callbacks.EarlyStopping(
                monitor="val_loss" if val_data else "loss",
                patience=15,
                restore_best_weights=True,
                verbose=0,
            ),
            _k.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_data else "loss",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        epochs = self._params.get("epochs", 100)
        self._model.fit(
            X_seq,
            y_seq,
            validation_data=val_data,
            epochs=epochs,
            batch_size=self._batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        if self._model is None or self._scaler_y is None:
            return np.zeros(len(X))
        # Roll through X one step at a time, using last lookback as context
        preds = []
        # We need a buffer of lookback observations. Use first lookback rows of X
        # as initial context, then predict one step at a time.
        context = X[: self._lookback].copy()
        for i in range(len(X)):
            # Reshape context to (1, lookback, n_features)
            x_in = context[np.newaxis, :, :]
            y_scaled = self._model.predict(x_in, verbose=0).ravel()[0]
            # Inverse-scale
            y_val = float(self._scaler_y.inverse_transform(np.array([[y_scaled]]))[0, 0])
            preds.append(y_val)
            # Slide context: drop oldest, add current X row (if available)
            if i < len(X) - 1:
                context = np.vstack([context[1:], X[i + 1 : i + 2]])
        return np.array(preds)

    def tune(self, X_train, y_train, param_grid, n_trials=20):
        best_rmse = np.inf
        best_params = {}
        n = len(X_train)
        split = int(n * 0.8)
        X_tr, X_va = X_train[:split], X_train[split:]
        y_tr, y_va = y_train[:split], y_train[split:]

        keys = ["units", "dropout", "learning_rate", "epochs"]
        grids = [param_grid.get(k, [64]) for k in keys]
        for combo in itertools.product(*grids):
            params = dict(zip(keys, combo))
            try:
                w = _LSTMWrapper(**params)
                w._lookback = self._lookback
                w.fit(X_tr, y_tr, X_va, y_va)
                pred = w.predict(X_va)
                err = rmse(y_va, pred)
                if err < best_rmse:
                    best_rmse = err
                    best_params = params
            except Exception:
                continue
        logger.info("  LSTM best: %s (RMSE=%.4f)", best_params, best_rmse)
        return best_params


# ══════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════

WRAPPER_CLASSES: Dict[str, type] = {
    "lr": _LRWrapper,
    "rf": _RFWrapper,
    "xgb": _XGBWrapper,
    "lgbm": _LGBMWrapper,
    "cb": _CBWrapper,
    "arima": _ARIMAWrapper,
    "prophet": _ProphetWrapper,
    "lstm": _LSTMWrapper,
}


# ══════════════════════════════════════════════════════════════
# Single model result
# ══════════════════════════════════════════════════════════════


@dataclass
class ModelResult:
    name: str
    label: str
    params: Dict[str, Any]
    val_metrics: EvalMetrics
    test_metrics: EvalMetrics
    y_val_true: np.ndarray
    y_val_pred: np.ndarray
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    feature_importance: Optional[pd.DataFrame] = None
    shap_values: Optional[Any] = None


# ══════════════════════════════════════════════════════════════
# PredictionPipeline
# ══════════════════════════════════════════════════════════════


class PredictionPipeline:
    """Full prediction pipeline: train, tune, compare, evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset with PeriodIndex or DatetimeIndex.
    target_col : str, default "aot_close"
        Target column name.
    feature_cols : list of str, optional
        Feature columns. If None, uses all numeric columns except target.
    test_months : int, default 12
        Number of most recent months for final test evaluation.
    val_months : int, default 6
        Number of months between train and test for validation / tuning.
    scale_features : bool, default True
        Apply StandardScaler to features.
    seed : int, default 42
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "aot_close",
        feature_cols: Optional[List[str]] = None,
        test_months: int = 12,
        val_months: int = 6,
        scale_features: bool = True,
        seed: int = 42,
    ):
        self._df = df.copy()
        self._target_col = target_col
        self._test_months = test_months
        self._val_months = val_months
        self._seed = seed

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
            ]
        self._feature_cols = feature_cols
        self._scale_features = scale_features
        self._scaler_X: Optional[StandardScaler] = None

        # ── split ──────────────────────────────────────────
        self._split_data()

        logger.info(
            "PredictionPipeline: %d features | train=%d val=%d test=%d",
            len(feature_cols),
            len(self._y_train),
            len(self._y_val),
            len(self._y_test),
        )

    # ── data splitting ─────────────────────────────────────

    def _split_data(self) -> None:
        df = self._df.dropna(subset=[self._target_col] + self._feature_cols)
        n = len(df)
        train_end = n - self._test_months - self._val_months
        val_end = n - self._test_months

        if train_end < 2:
            raise ValueError(
                f"Too few rows ({n}) for test={self._test_months}, val={self._val_months}"
            )

        X = df[self._feature_cols].values.astype(np.float64)
        y = df[self._target_col].values.astype(np.float64)

        self._X_train = X[:train_end]
        self._y_train = y[:train_end]
        self._X_val = X[train_end:val_end]
        self._y_val = y[train_end:val_end]
        self._X_test = X[val_end:]
        self._y_test = y[val_end:]

        self._train_index = df.index[:train_end]
        self._val_index = df.index[train_end:val_end]
        self._test_index = df.index[val_end:]

    # ── feature scaling ────────────────────────────────────

    def _scale(self) -> None:
        if not self._scale_features:
            return
        self._scaler_X = StandardScaler()
        self._X_train = self._scaler_X.fit_transform(self._X_train)
        self._X_val = self._scaler_X.transform(self._X_val)
        self._X_test = self._scaler_X.transform(self._X_test)

    # ── run ────────────────────────────────────────────────

    def run(
        self,
        models: Optional[List[str]] = None,
        tune: bool = True,
        calc_shap: bool = True,
    ) -> "PredictionResults":
        """Run the full pipeline for specified models.

        Parameters
        ----------
        models : list of str, optional
            Model keys (e.g., ["lr", "rf", "xgb"]). Default: all 8.
        tune : bool, default True
            Whether to perform hyper-parameter tuning.
        calc_shap : bool, default True
            Compute SHAP values for tree / linear models.

        Returns
        -------
        PredictionResults
        """
        if models is None:
            models = list(WRAPPER_CLASSES.keys())

        self._scale()

        results: List[ModelResult] = []
        best_val_rmse = np.inf
        best_result: Optional[ModelResult] = None

        for key in models:
            if key not in WRAPPER_CLASSES:
                logger.warning("Unknown model '%s', skipping", key)
                continue

            logger.info("─" * 50)
            logger.info("  %s (%s)", MODEL_LABELS.get(key, key), key)
            logger.info("─" * 50)

            wrapper_cls = WRAPPER_CLASSES[key]
            is_uni = wrapper_cls.is_univariate

            # ── tune ───────────────────────────────────────
            if is_uni:
                X_tr, X_va = self._y_train.reshape(-1, 1), self._y_val.reshape(-1, 1)
                X_te = self._y_test.reshape(-1, 1)
            else:
                X_tr, X_va, X_te = self._X_train, self._X_val, self._X_test

            if tune:
                grid = MODEL_GRIDS.get(key, {})
                dummy = wrapper_cls()
                best_params = dummy.tune(X_tr, self._y_train, grid)
            else:
                best_params = {}

            # ── train with best params ─────────────────────
            wrapper = wrapper_cls(**best_params)
            wrapper.fit(self._X_train, self._y_train, self._X_val, self._y_val)

            # ── predict ────────────────────────────────────
            if is_uni:
                y_val_pred = wrapper.predict(X_va)
                y_test_pred = wrapper.predict(X_te)
            else:
                y_val_pred = wrapper.predict(self._X_val)
                y_test_pred = wrapper.predict(self._X_test)

            val_metrics = EvalMetrics.compute(self._y_val, y_val_pred)
            test_metrics = EvalMetrics.compute(self._y_test, y_test_pred)

            logger.info("  Val:  %s", val_metrics)
            logger.info("  Test: %s", test_metrics)

            # ── feature importance ─────────────────────────
            fi = wrapper.get_feature_importance()
            if fi is not None:
                fi["feature"] = [
                    self._feature_cols[i] if i < len(self._feature_cols) else f"f{i}"
                    for i in range(len(fi))
                ]

            # ── SHAP ───────────────────────────────────────
            shap_vals = None
            if calc_shap and not is_uni and hasattr(wrapper._model, "predict"):
                try:
                    explainer = shap.Explainer(wrapper._model, self._X_train)
                    shap_vals = explainer(self._X_test[: min(50, len(self._X_test))])
                except Exception as exc:
                    logger.debug("SHAP failed for %s: %s", key, exc)

            result = ModelResult(
                name=key,
                label=MODEL_LABELS.get(key, key),
                params=best_params,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                y_val_true=self._y_val.copy(),
                y_val_pred=y_val_pred,
                y_test_true=self._y_test.copy(),
                y_test_pred=y_test_pred,
                feature_importance=fi,
                shap_values=shap_vals,
            )
            results.append(result)

            if val_metrics.rmse < best_val_rmse:
                best_val_rmse = val_metrics.rmse
                best_result = result

        if best_result:
            logger.info("═" * 50)
            logger.info("  BEST MODEL: %s (val RMSE=%.4f)", best_result.label, best_val_rmse)
            logger.info("═" * 50)

        return PredictionResults(
            results=results,
            best_model_name=best_result.name if best_result else None,
            y_train=self._y_train.copy(),
            y_val=self._y_val.copy(),
            y_test=self._y_test.copy(),
            train_index=self._train_index,
            val_index=self._val_index,
            test_index=self._test_index,
        )


# ══════════════════════════════════════════════════════════════
# PredictionResults container + visualisation
# ══════════════════════════════════════════════════════════════


@dataclass
class PredictionResults:
    """Container for all model results and visualisation methods."""

    results: List[ModelResult]
    best_model_name: Optional[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index

    # ── summary ────────────────────────────────────────────

    def summary(self) -> str:
        """Text summary of all models and their metrics."""
        lines = [
            "Prediction Results Summary",
            "==========================",
            f"Train: {len(self.y_train)} | Val: {len(self.y_val)} | Test: {len(self.y_test)}",
            "",
            f"{'Model':20s} {'Val RMSE':>10s} {'Val MAE':>10s} {'Val MAPE':>8s} {'Val R²':>8s}  |  {'Test RMSE':>10s} {'Test MAE':>10s} {'Test MAPE':>8s} {'Test R²':>8s}",
            "-" * 100,
        ]
        for r in self.results:
            v = r.val_metrics
            t = r.test_metrics
            lines.append(
                f"{r.label:20s} {v.rmse:>10.4f} {v.mae:>10.4f} {v.mape:>7.2f}% {v.r2:>8.4f}  |  "
                f"{t.rmse:>10.4f} {t.mae:>10.4f} {t.mape:>7.2f}% {t.r2:>8.4f}"
            )
        lines.append("")
        if self.best_model_name:
            best = next((r for r in self.results if r.name == self.best_model_name), None)
            if best:
                lines.append(f"BEST: {best.label} (val RMSE={best.val_metrics.rmse:.4f})")
                lines.append(f"  Test performance: {best.test_metrics}")
        return "\n".join(lines)

    def metrics_dataframe(self) -> pd.DataFrame:
        """Return all metrics as a DataFrame."""
        rows = []
        for r in self.results:
            row = {"model": r.label, "params": str(r.params)}
            for prefix, m in [("val", r.val_metrics), ("test", r.test_metrics)]:
                row[f"{prefix}_rmse"] = m.rmse
                row[f"{prefix}_mae"] = m.mae
                row[f"{prefix}_mape"] = m.mape
                row[f"{prefix}_r2"] = m.r2
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    # ── get best model ─────────────────────────────────────

    def best(self) -> Optional[ModelResult]:
        """Return the best ModelResult (lowest val RMSE)."""
        if not self.results or not self.best_model_name:
            return None
        for r in self.results:
            if r.name == self.best_model_name:
                return r
        return min(self.results, key=lambda r: r.val_metrics.rmse)

    # ── plots ──────────────────────────────────────────────

    def plot_model_comparison(
        self,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (12, 7),
    ) -> Tuple[Figure, plt.Axes]:
        """Bar chart comparing all models on validation metrics."""
        df_m = self.metrics_dataframe()
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, metric, title in zip(
            axes,
            ["val_rmse", "val_mae", "val_r2"],
            ["RMSE (lower is better)", "MAE (lower is better)", "R² (higher is better)"],
        ):
            vals = df_m[metric].values
            colors = [
                "#C44E52" if v == (vals.min() if "r2" not in metric else vals.max()) else "#4C72B0"
                for v in vals
            ]
            ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(df_m.index, fontsize=9)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.invert_yaxis()
            for i, v in enumerate(vals):
                ax.text(v + abs(v) * 0.02, i, f"{v:.4f}", va="center", fontsize=7)
            ax.grid(axis="x", alpha=0.3)

        fig.suptitle(
            "Model Comparison — Validation Metrics", fontweight="bold", fontsize=13, y=1.02
        )
        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Exported model comparison: %s", p)
        return fig, axes  # noqa: F821

    def plot_predictions(
        self,
        model_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (14, 8),
    ) -> Tuple[Figure, plt.Axes]:
        """Time-series plot of actual vs predicted (train / val / test).

        Parameters
        ----------
        model_name : str, optional
            Model to plot. If None, uses the best model.
        output_path : str or Path, optional

        Returns
        -------
        (fig, ax) tuple.
        """
        target = self._resolve_model(model_name)
        if target is None:
            return plt.subplots()

        # Build full time series
        full_idx = list(self.train_index) + list(self.val_index) + list(self.test_index)
        full_actual = np.concatenate([self.y_train, self.y_val, self.y_test])
        full_pred = np.concatenate(
            [
                np.full_like(self.y_train, np.nan),
                target.y_val_pred,
                target.y_test_pred,
            ]
        )
        train_end = len(self.train_index)
        val_end = train_end + len(self.val_index)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

        # ── Top: time series ───────────────────────────────
        x = np.arange(len(full_actual))
        ax1.plot(
            x,
            full_actual,
            color=PUBLICATION_COLORS["actual"],
            linewidth=2,
            label="Actual",
            alpha=0.85,
        )
        ax1.plot(
            x,
            full_pred,
            color=PUBLICATION_COLORS["pred"],
            linewidth=2,
            label=f"{target.label} Predicted",
            alpha=0.85,
            linestyle="--",
        )

        # Split regions
        ax1.axvline(train_end - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.axvline(val_end - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.text(
            train_end / 2,
            ax1.get_ylim()[1],
            "TRAIN",
            ha="center",
            fontsize=9,
            color=PUBLICATION_COLORS["train"],
            fontweight="bold",
        )
        mid_val = (train_end + val_end) / 2
        ax1.text(
            mid_val,
            ax1.get_ylim()[1],
            "VAL",
            ha="center",
            fontsize=9,
            color=PUBLICATION_COLORS["val"],
            fontweight="bold",
        )
        mid_test = (val_end + len(full_actual)) / 2
        ax1.text(
            mid_test,
            ax1.get_ylim()[1],
            "TEST",
            ha="center",
            fontsize=9,
            color=PUBLICATION_COLORS["test"],
            fontweight="bold",
        )

        ax1.set_ylabel("AOT Closing Price (THB)", fontsize=11)
        ax1.set_title(f"Actual vs Predicted — {target.label}", fontweight="bold", pad=12)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(alpha=0.3)

        # ── Bottom: actual vs predicted scatter (test only) ──
        ax2.scatter(
            target.y_test_true,
            target.y_test_pred,
            color=PUBLICATION_COLORS["test"],
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=40,
        )
        lims = [
            min(target.y_test_true.min(), target.y_test_pred.min()),
            max(target.y_test_true.max(), target.y_test_pred.max()),
        ]
        ax2.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Ideal")
        ax2.set_xlabel("Actual (THB)", fontsize=11)
        ax2.set_ylabel("Predicted (THB)", fontsize=11)
        ax2.set_title(
            f"Test Set Scatter  (RMSE={target.test_metrics.rmse:.2f})", fontweight="bold", pad=10
        )
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.set_aspect("equal")

        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Exported prediction plot: %s", p)

        return fig, ax1

    def plot_residuals(
        self,
        model_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (12, 5),
    ) -> Tuple[Figure, plt.Axes]:
        """Residual diagnostics: scatter + histogram."""
        target = self._resolve_model(model_name)
        if target is None:
            return plt.subplots()

        residuals = target.y_test_true - target.y_test_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── Residual vs Predicted ──────────────────────────
        ax1.scatter(
            target.y_test_pred,
            residuals,
            color=PUBLICATION_COLORS["residual"],
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
            s=35,
        )
        ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax1.set_xlabel("Predicted (THB)", fontsize=11)
        ax1.set_ylabel("Residual (THB)", fontsize=11)
        ax1.set_title("Residuals vs Predicted", fontweight="bold", pad=10)
        ax1.grid(alpha=0.3)

        # ── Residual histogram ─────────────────────────────
        ax2.hist(
            residuals,
            bins=min(20, len(residuals) // 2),
            color=PUBLICATION_COLORS["residual"],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.8,
            density=True,
        )
        # Normal curve overlay
        mu, std = residuals.mean(), residuals.std()
        x_grid = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(
            x_grid,
            sp_stats.norm.pdf(x_grid, mu, std),
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Normal fit",
        )
        ax2.axvline(0, color="#C44E52", linewidth=1.2, linestyle=":", alpha=0.7, label="Zero error")
        ax2.set_xlabel("Residual (THB)", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(
            f"Residual Distribution  (μ={mu:.2f}, σ={std:.2f})", fontweight="bold", pad=10
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        fig.suptitle(f"{target.label} — Residual Analysis", fontweight="bold", fontsize=13, y=1.02)
        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Exported residual plot: %s", p)

        return fig, ax1

    def plot_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 15,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (10, 7),
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """Horizontal bar chart of feature importance.

        Parameters
        ----------
        model_name : str, optional
            Model to plot. If None, uses the best model.
        top_n : int, default 15
            Show top N features.
        output_path : str or Path, optional

        Returns
        -------
        (fig, ax) or None if model has no feature importance.
        """
        target = self._resolve_model(model_name)
        if target is None or target.feature_importance is None:
            return None

        fi = target.feature_importance.sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(fi)),
            fi["importance"].values,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Feature Importance — {target.label}", fontweight="bold", pad=12)
        ax.grid(axis="x", alpha=0.3)

        for i, v in enumerate(fi["importance"].values):
            ax.text(v + v * 0.02, i, f"{v:.4f}", va="center", fontsize=7)

        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Exported feature importance: %s", p)

        return fig, ax

    def plot_shap(
        self,
        model_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (12, 6),
    ) -> Optional[Tuple[Figure, Any]]:
        """SHAP beeswarm summary plot.

        Parameters
        ----------
        model_name : str, optional
            If None, uses the best model.
        output_path : str or Path, optional

        Returns
        -------
        (fig, ax) or None if SHAP values not available.
        """
        target = self._resolve_model(model_name)
        if target is None or target.shap_values is None:
            return None

        feature_names = self._get_feature_cols(target)
        if not feature_names:
            feature_names = [f"f{i}" for i in range(target.shap_values.shape[1])]

        shap.summary_plot(
            target.shap_values,
            feature_names=feature_names,
            show=False,
            alpha=0.6,
            plot_size=(figsize[1], figsize[0]),
        )
        fig = plt.gcf()
        fig.set_size_inches(*figsize)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            ax.set_title(f"SHAP Summary — {target.label}", fontweight="bold", pad=12)
        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Exported SHAP plot: %s", p)

        return fig, ax

    # ── generate_all ───────────────────────────────────────

    def plot_all(
        self,
        output_dir: Union[str, Path] = "reports/figures",
        model_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Generate all prediction figures.

        Parameters
        ----------
        output_dir : str or Path
        model_name : str, optional
            Model to plot. If None, uses the best model.

        Returns
        -------
        dict: {description: Path}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        target = self._resolve_model(model_name)
        best_suffix = f"_{target.name}" if target and target.name != self.best_model_name else ""

        generators = [
            ("model_comparison", self.plot_model_comparison, {}),
            (f"predictions{best_suffix}", self.plot_predictions, {"model_name": model_name}),
            (f"residuals{best_suffix}", self.plot_residuals, {"model_name": model_name}),
        ]

        if target and target.feature_importance is not None:
            generators.append(
                (
                    f"feature_importance{best_suffix}",
                    self.plot_feature_importance,
                    {"model_name": model_name},
                ),
            )
        if target and target.shap_values is not None:
            generators.append(
                (f"shap_summary{best_suffix}", self.plot_shap, {"model_name": model_name}),
            )

        for name, fn, kwargs in generators:
            try:
                result = fn(**kwargs)
                if result is None:
                    continue
                if isinstance(result, tuple):
                    fig = result[0]
                else:
                    fig = result
                path = out / f"{name}.svg"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                paths[name] = path
                plt.close(fig)
            except Exception as exc:
                logger.error("Failed to generate '%s': %s", name, exc)

        logger.info("Generated %d/%d prediction figures in %s", len(paths), len(generators), out)
        return paths

    # ── helpers ────────────────────────────────────────────

    def _resolve_model(self, model_name: Optional[str] = None) -> Optional[ModelResult]:
        if model_name:
            for r in self.results:
                if r.name == model_name:
                    return r
            logger.warning("Model '%s' not found, using best", model_name)
        return self.best()

    def _get_feature_cols(self, target: ModelResult) -> List[str]:
        """Return feature column names from the feature importance if available."""
        if target.feature_importance is not None:
            return target.feature_importance["feature"].tolist()
        return []


# ══════════════════════════════════════════════════════════════
# Module exports
# ══════════════════════════════════════════════════════════════

__all__ = [
    "PredictionPipeline",
    "PredictionResults",
    "ModelResult",
    "EvalMetrics",
    "WRAPPER_CLASSES",
    "MODEL_LABELS",
    "MODEL_GRIDS",
]
