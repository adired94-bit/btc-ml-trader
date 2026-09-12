"""Training pipeline: chronological split, optional hyper-parameter search with
time-series cross-validation, evaluation and artifact persistence.

Run as a script::

    python -m src.models.train            # train with default parameters
    python -m src.models.train --tune     # randomized search (slower)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
    mean_absolute_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from config import settings
from src.data import storage
from src.data.processor import CLASS_NAMES, DOWN, UP, build_dataset
from src.logging_config import get_logger
from src.models.ensemble import (
    DirectionEnsemble,
    RangeRegressor,
    default_lgbm_params,
    default_xgb_params,
    lgbm_search_space,
    xgb_search_space,
)

logger = get_logger(__name__)

DIRECTION_MODEL_FILE = "direction_ensemble.joblib"
RANGE_MODEL_FILE = "range_regressor.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class TrainingReport:
    trained_at: str
    symbol: str
    timeframe: str
    horizon: int
    n_train: int
    n_test: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    feature_names: list[str]
    class_distribution_train: dict[str, int]
    class_distribution_test: dict[str, int]
    direction_metrics: dict[str, Any]
    range_metrics: dict[str, Any]
    xgb_params: dict[str, Any]
    lgbm_params: dict[str, Any]
    tuned: bool
    training_seconds: float
    top_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chronological_split(
    X: pd.DataFrame, y: pd.DataFrame, train_fraction: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically and leave a ``horizon`` gap so no test label overlaps training."""
    frac = train_fraction or settings.train_test_split
    if not 0.5 <= frac < 1.0:
        raise ValueError("train fraction must be within [0.5, 1.0)")
    split = int(len(X) * frac)
    gap = settings.prediction_horizon
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split + gap:], y.iloc[split + gap:]
    if len(X_test) < 50:
        raise ValueError("Not enough data for a meaningful test split")
    return X_train, X_test, y_train, y_test


def _search(estimator: Any, space: dict[str, list[Any]], X: pd.DataFrame, y: np.ndarray, n_iter: int, name: str) -> dict[str, Any]:
    cv = TimeSeriesSplit(n_splits=3, gap=settings.prediction_horizon)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=settings.random_state,
        n_jobs=1,  # the estimators already use every core
        refit=False,
        verbose=0,
    )
    sample_weight = compute_sample_weight("balanced", y)
    t0 = time.perf_counter()
    search.fit(X.to_numpy(dtype=np.float32), y, sample_weight=sample_weight)
    logger.info(
        "%s search finished in %.1fs, best CV log-loss %.4f, params %s",
        name, time.perf_counter() - t0, -search.best_score_, search.best_params_,
    )
    return dict(search.best_params_)


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_iter: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    """Randomized search over both boosters using expanding-window CV."""
    y_arr = np.asarray(y, dtype=int)
    xgb_best = _search(xgb.XGBClassifier(**default_xgb_params()), xgb_search_space(), X, y_arr, n_iter, "XGBoost")
    lgb_best = _search(lgb.LGBMClassifier(**default_lgbm_params()), lgbm_search_space(), X, y_arr, n_iter, "LightGBM")
    return xgb_best, lgb_best


def evaluate_direction(model: DirectionEnsemble, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    y_arr = np.asarray(y_test, dtype=int)
    report = classification_report(
        y_arr, pred, labels=[0, 1, 2], target_names=[CLASS_NAMES[i] for i in (0, 1, 2)],
        output_dict=True, zero_division=0,
    )
    # Directional accuracy on confident non-flat calls - the metric that matters for trading.
    p_up, p_down = proba[:, UP], proba[:, DOWN]
    threshold = settings.signal_probability_threshold
    long_mask = (p_up >= threshold) & (p_up > p_down)
    short_mask = (p_down >= threshold) & (p_down > p_up)
    traded = long_mask | short_mask
    hits = ((long_mask & (y_arr == UP)) | (short_mask & (y_arr == DOWN))).sum()
    coverage = float(traded.mean())
    directional_acc = float(hits / traded.sum()) if traded.sum() else 0.0

    components = model.predict_proba_components(X_test)
    return {
        "accuracy": float(accuracy_score(y_arr, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "log_loss": float(log_loss(y_arr, proba, labels=[0, 1, 2])),
        "log_loss_xgboost": float(log_loss(y_arr, components["xgboost"], labels=[0, 1, 2])),
        "log_loss_lightgbm": float(log_loss(y_arr, components["lightgbm"], labels=[0, 1, 2])),
        "signal_threshold": threshold,
        "signal_coverage": coverage,
        "signal_directional_accuracy": directional_acc,
        "n_signals": int(traded.sum()),
        "per_class": {k: v for k, v in report.items() if k in CLASS_NAMES.values()},
    }


def evaluate_range(model: RangeRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict[str, Any]:
    pred = model.predict(X_test)
    truth = y_test[RangeRegressor.TARGETS]
    return {
        "mae_max_up_pct": float(mean_absolute_error(truth["future_max_up"], pred["future_max_up"]) * 100),
        "mae_max_down_pct": float(mean_absolute_error(truth["future_max_down"], pred["future_max_down"]) * 100),
        "mean_pred_up_pct": float(pred["future_max_up"].mean() * 100),
        "mean_pred_down_pct": float(pred["future_max_down"].mean() * 100),
        "mean_true_up_pct": float(truth["future_max_up"].mean() * 100),
        "mean_true_down_pct": float(truth["future_max_down"].mean() * 100),
    }


def train_models(
    df: pd.DataFrame | None = None,
    tune: bool = False,
    n_iter: int = 12,
    models_dir: Path | None = None,
    refit_on_full: bool = True,
) -> TrainingReport:
    """End-to-end training. Returns the report and persists artifacts to ``models_dir``."""
    t0 = time.perf_counter()
    models_dir = Path(models_dir or settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = storage.get_ohlcv()
    X, y, _ = build_dataset(df)
    logger.info("Dataset: %d rows, %d features", len(X), X.shape[1])
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    xgb_params: dict[str, Any] = {}
    lgbm_params: dict[str, Any] = {}
    if tune:
        xgb_params, lgbm_params = tune_hyperparameters(X_train, y_train["direction"], n_iter=n_iter)

    direction = DirectionEnsemble(xgb_params=xgb_params, lgbm_params=lgbm_params)
    direction.fit(X_train, y_train["direction"])
    direction_metrics = evaluate_direction(direction, X_test, y_test["direction"])
    logger.info("Direction metrics: %s", json.dumps({k: v for k, v in direction_metrics.items() if k != "per_class"}, indent=None))

    range_model = RangeRegressor()
    range_model.fit(X_train, y_train[RangeRegressor.TARGETS])
    range_metrics = evaluate_range(range_model, X_test, y_test)
    logger.info("Range metrics: %s", json.dumps(range_metrics))

    if refit_on_full:
        # Production models learn from every available bar (metrics above stay honest/out-of-sample).
        logger.info("Refitting production models on the full dataset")
        direction = DirectionEnsemble(xgb_params=xgb_params, lgbm_params=lgbm_params).fit(X, y["direction"])
        range_model = RangeRegressor().fit(X, y[RangeRegressor.TARGETS])

    direction.save(models_dir / DIRECTION_MODEL_FILE)
    range_model.save(models_dir / RANGE_MODEL_FILE)

    report = TrainingReport(
        trained_at=datetime.now(timezone.utc).isoformat(),
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        horizon=settings.prediction_horizon,
        n_train=len(X_train),
        n_test=len(X_test),
        train_start=X_train.index[0].isoformat(),
        train_end=X_train.index[-1].isoformat(),
        test_start=X_test.index[0].isoformat(),
        test_end=X_test.index[-1].isoformat(),
        feature_names=list(X.columns),
        class_distribution_train={CLASS_NAMES[int(k)]: int(v) for k, v in y_train["direction"].value_counts().items()},
        class_distribution_test={CLASS_NAMES[int(k)]: int(v) for k, v in y_test["direction"].value_counts().items()},
        direction_metrics=direction_metrics,
        range_metrics=range_metrics,
        xgb_params=direction.xgb_params,
        lgbm_params=direction.lgbm_params,
        tuned=tune,
        training_seconds=round(time.perf_counter() - t0, 1),
        top_features={k: float(v) for k, v in direction.feature_importances_.head(15).items()},
    )
    (models_dir / METADATA_FILE).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Training complete in %.1fs; artifacts in %s", report.training_seconds, models_dir)
    return report


def load_metadata(models_dir: Path | None = None) -> dict[str, Any] | None:
    path = Path(models_dir or settings.models_dir) / METADATA_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BTC direction ensemble and range regressor")
    parser.add_argument("--tune", action="store_true", help="run randomized hyper-parameter search")
    parser.add_argument("--n-iter", type=int, default=12, help="search iterations per model")
    parser.add_argument("--refresh", action="store_true", help="force a full re-download of market data")
    args = parser.parse_args()

    df = storage.get_ohlcv(force_refresh=args.refresh)
    report = train_models(df, tune=args.tune, n_iter=args.n_iter)
    dm = report.direction_metrics
    print("\n=== Training report ===")
    print(f"Rows train/test        : {report.n_train} / {report.n_test}")
    print(f"Test period            : {report.test_start} -> {report.test_end}")
    print(f"Accuracy / balanced    : {dm['accuracy']:.3f} / {dm['balanced_accuracy']:.3f}")
    print(f"Log-loss (ens/xgb/lgbm): {dm['log_loss']:.4f} / {dm['log_loss_xgboost']:.4f} / {dm['log_loss_lightgbm']:.4f}")
    print(f"Signal coverage        : {dm['signal_coverage']:.1%} ({dm['n_signals']} signals)")
    print(f"Signal dir. accuracy   : {dm['signal_directional_accuracy']:.1%}")
    print(f"Range MAE up/down (%)  : {report.range_metrics['mae_max_up_pct']:.3f} / {report.range_metrics['mae_max_down_pct']:.3f}")
    print(f"Top features           : {list(report.top_features)[:8]}")
    print(f"Training time          : {report.training_seconds}s")


if __name__ == "__main__":
    main()
