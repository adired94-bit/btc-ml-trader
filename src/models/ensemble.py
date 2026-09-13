"""Model definitions: XGBoost + LightGBM soft-voting direction ensemble and a
Random-Forest price-range regressor. Both support joblib persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

from config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

N_CLASSES = 3


def default_xgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "gamma": 0.1,
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": settings.random_state,
        "verbosity": 0,
    }


def default_lgbm_params() -> dict[str, Any]:
    return {
        "n_estimators": 500,
        "num_leaves": 15,
        "max_depth": -1,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 30,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "n_jobs": -1,
        "random_state": settings.random_state,
        "verbose": -1,
    }


def xgb_search_space() -> dict[str, list[Any]]:
    return {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.3],
    }


def lgbm_search_space() -> dict[str, list[Any]]:
    return {
        "n_estimators": [200, 400, 600],
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [20, 30, 50],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }


class DirectionEnsemble:
    """Soft-voting ensemble of an XGBoost and a LightGBM multi-class classifier."""

    def __init__(
        self,
        xgb_params: dict[str, Any] | None = None,
        lgbm_params: dict[str, Any] | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        balance_classes: bool = True,
    ) -> None:
        self.xgb_params = {**default_xgb_params(), **(xgb_params or {})}
        self.lgbm_params = {**default_lgbm_params(), **(lgbm_params or {})}
        if len(weights) != 2 or any(w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("weights must be two non-negative numbers with a positive sum")
        total = float(sum(weights))
        self.weights = (weights[0] / total, weights[1] / total)
        self.balance_classes = balance_classes
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.feature_names: list[str] = []
        self.classes_ = np.arange(N_CLASSES)
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "DirectionEnsemble":
        """Fit both boosters. ``sample_weight`` (optional) is multiplied with class-balancing weights."""
        y_arr = np.asarray(y, dtype=int)
        if X.empty or len(X) != len(y_arr):
            raise ValueError("X and y must be non-empty and aligned")
        if set(np.unique(y_arr)) - {0, 1, 2}:
            raise ValueError("Labels must be encoded as 0 (DOWN), 1 (FLAT), 2 (UP)")
        self.feature_names = list(X.columns)
        # Boosters require contiguous labels 0..k-1; remember which of the 3 classes are present.
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required to train the ensemble")
        y_enc = np.searchsorted(self.classes_, y_arr)
        k = len(self.classes_)
        self.xgb_model = xgb.XGBClassifier(**{**self.xgb_params, "num_class": k})
        self.lgbm_model = lgb.LGBMClassifier(**{**self.lgbm_params, "num_class": k})
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y_enc) if self.balance_classes else None
        elif self.balance_classes:
            sample_weight = np.asarray(sample_weight, dtype=float) * compute_sample_weight("balanced", y_enc)
        logger.info("Fitting XGBoost on %d rows x %d features", len(X), X.shape[1])
        self.xgb_model.fit(X.to_numpy(dtype=np.float32), y_enc, sample_weight=sample_weight)
        logger.info("Fitting LightGBM on %d rows x %d features", len(X), X.shape[1])
        self.lgbm_model.fit(X.to_numpy(dtype=np.float32), y_enc, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def _check(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        return X[self.feature_names].to_numpy(dtype=np.float32)

    def _full_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Return an (n, 3) probability matrix even if a class was absent in training."""
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 1:  # binary models may return P(class 1) only
            proba = np.column_stack([1 - proba, proba])
        classes = getattr(self, "classes_", np.arange(N_CLASSES))  # artifacts saved before classes_ existed
        if proba.shape[1] == N_CLASSES and np.array_equal(classes, np.arange(N_CLASSES)):
            return proba
        full = np.zeros((proba.shape[0], N_CLASSES))
        full[:, classes] = proba
        return full

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._check(X)
        p_xgb = self._full_proba(self.xgb_model, arr)
        p_lgb = self._full_proba(self.lgbm_model, arr)
        proba = self.weights[0] * p_xgb + self.weights[1] * p_lgb
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_proba_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        arr = self._check(X)
        return {
            "xgboost": self._full_proba(self.xgb_model, arr),
            "lightgbm": self._full_proba(self.lgbm_model, arr),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances_(self) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted")
        xgb_imp = np.asarray(self.xgb_model.feature_importances_, dtype=float)
        lgb_imp = np.asarray(self.lgbm_model.feature_importances_, dtype=float)
        xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
        lgb_imp = lgb_imp / lgb_imp.sum() if lgb_imp.sum() > 0 else lgb_imp
        combined = self.weights[0] * xgb_imp + self.weights[1] * lgb_imp
        return pd.Series(combined, index=self.feature_names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved direction ensemble to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "DirectionEnsemble":
        obj = joblib.load(Path(path))
        # Compare by class name: Streamlit hot-reloads modules, so the class object identity
        # can differ between the unpickled instance and ``cls`` even for the same class.
        if type(obj).__name__ != cls.__name__:
            raise TypeError(f"{path} does not contain a DirectionEnsemble")
        return obj


class RangeRegressor:
    """Random-Forest multi-output regressor predicting the expected max-up and
    max-down excursion (fractions of the entry price) over the horizon."""

    TARGETS = ["future_max_up", "future_max_down"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": 0.5,
            "n_jobs": -1,
            "random_state": settings.random_state,
            **(params or {}),
        }
        self.model = RandomForestRegressor(**self.params)
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RangeRegressor":
        if list(y.columns) != self.TARGETS:
            raise ValueError(f"Range targets must be {self.TARGETS}")
        if X.empty or len(X) != len(y):
            raise ValueError("X and y must be non-empty and aligned")
        self.feature_names = list(X.columns)
        logger.info("Fitting RandomForest range model on %d rows", len(X))
        self.model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Range model is not fitted")
        arr = X[self.feature_names].to_numpy(dtype=np.float32)
        pred = self.model.predict(arr)
        pred = np.atleast_2d(pred)
        out = pd.DataFrame(pred, columns=self.TARGETS, index=X.index)
        # Physical constraints: an up-excursion cannot be negative, a down one cannot be positive.
        out["future_max_up"] = out["future_max_up"].clip(lower=0.0)
        out["future_max_down"] = out["future_max_down"].clip(upper=0.0)
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved range regressor to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RangeRegressor":
        obj = joblib.load(Path(path))
        if type(obj).__name__ != cls.__name__:
            raise TypeError(f"{path} does not contain a RangeRegressor")
        return obj
