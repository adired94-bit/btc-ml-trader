"""Model tests: ensemble behaviour, persistence, OOS split hygiene, inference & signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.models.ensemble import DirectionEnsemble, RangeRegressor
from src.models.predict import ModelNotTrainedError, Predictor, decide_action, generate_signal
from src.models.train import chronological_split, evaluate_direction, train_models
from src.risk.management import RiskManager
from tests.conftest import FAST_LGBM, FAST_RF, FAST_XGB


def test_chronological_split_has_gap_and_no_overlap(dataset):
    X, y, _ = dataset
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, 0.8)
    assert X_tr.index.max() < X_te.index.min()
    gap_bars = (X_te.index.min() - X_tr.index.max()) / pd.Timedelta(hours=1)
    assert gap_bars >= settings.prediction_horizon + 1
    assert len(X_tr) + len(X_te) <= len(X)
    assert X_tr.index.equals(y_tr.index) and X_te.index.equals(y_te.index)


def test_ensemble_probabilities_are_valid(dataset):
    X, y, _ = dataset
    model = DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X.iloc[:1200], y["direction"].iloc[:1200])
    proba = model.predict_proba(X.iloc[1200:1300])
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    comps = model.predict_proba_components(X.iloc[1200:1210])
    assert set(comps) == {"xgboost", "lightgbm"}
    blended = 0.5 * comps["xgboost"] + 0.5 * comps["lightgbm"]
    assert np.allclose(blended, model.predict_proba(X.iloc[1200:1210]), atol=1e-6)
    imp = model.feature_importances_
    assert len(imp) == X.shape[1] and imp.iloc[0] >= imp.iloc[-1]


def test_ensemble_handles_missing_class_in_training(dataset):
    X, y, _ = dataset
    y2 = y["direction"].iloc[:600].replace(1, 2)  # remove the FLAT class
    model = DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X.iloc[:600], y2)
    proba = model.predict_proba(X.iloc[600:620])
    assert proba.shape == (20, 3) and np.allclose(proba[:, 1], 0.0)


def test_ensemble_persistence_roundtrip(tmp_path, dataset):
    X, y, _ = dataset
    model = DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X.iloc[:800], y["direction"].iloc[:800])
    path = model.save(tmp_path / "ens.joblib")
    loaded = DirectionEnsemble.load(path)
    assert np.allclose(loaded.predict_proba(X.iloc[800:850]), model.predict_proba(X.iloc[800:850]))
    assert loaded.feature_names == list(X.columns)


def test_ensemble_input_validation(dataset):
    X, y, _ = dataset
    with pytest.raises(ValueError):
        DirectionEnsemble(weights=(0, 0))
    with pytest.raises(ValueError):
        DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X.iloc[:100], y["direction"].iloc[:100] + 5)
    model = DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM)
    with pytest.raises(RuntimeError):
        model.predict_proba(X.iloc[:5])
    model.fit(X.iloc[:300], y["direction"].iloc[:300])
    with pytest.raises(ValueError):
        model.predict_proba(X.iloc[:5].drop(columns=["rsi_14"]))


def test_range_regressor_constraints_and_roundtrip(tmp_path, dataset):
    X, y, _ = dataset
    model = RangeRegressor(FAST_RF).fit(X.iloc[:800], y[RangeRegressor.TARGETS].iloc[:800])
    pred = model.predict(X.iloc[800:900])
    assert list(pred.columns) == RangeRegressor.TARGETS
    assert (pred["future_max_up"] >= 0).all() and (pred["future_max_down"] <= 0).all()
    loaded = RangeRegressor.load(model.save(tmp_path / "rng.joblib"))
    pd.testing.assert_frame_equal(loaded.predict(X.iloc[800:810]), pred.iloc[:10])
    with pytest.raises(ValueError):
        RangeRegressor(FAST_RF).fit(X.iloc[:10], y[["future_return", "future_max_up"]].iloc[:10])


def test_evaluate_direction_reports_expected_keys(dataset):
    X, y, _ = dataset
    model = DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X.iloc[:1000], y["direction"].iloc[:1000])
    metrics = evaluate_direction(model, X.iloc[1004:], y["direction"].iloc[1004:])
    for key in ("accuracy", "balanced_accuracy", "log_loss", "signal_coverage", "signal_directional_accuracy", "per_class"):
        assert key in metrics
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["signal_coverage"] <= 1


def test_train_models_end_to_end_writes_artifacts(tmp_path, ohlcv, monkeypatch):
    from src.models import ensemble as ens

    base_xgb, base_lgbm = ens.default_xgb_params, ens.default_lgbm_params
    monkeypatch.setattr(ens, "default_xgb_params", lambda: {**base_xgb(), **FAST_XGB})
    monkeypatch.setattr(ens, "default_lgbm_params", lambda: {**base_lgbm(), **FAST_LGBM})
    report = train_models(ohlcv, tune=False, models_dir=tmp_path, refit_on_full=False)
    assert (tmp_path / "direction_ensemble.joblib").exists()
    assert (tmp_path / "range_regressor.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert report.n_test > 0 and report.test_start > report.train_end
    assert report.direction_metrics["log_loss"] > 0


def test_predictor_requires_artifacts(tmp_path):
    predictor = Predictor(models_dir=tmp_path)
    assert not predictor.artifacts_exist()
    with pytest.raises(ModelNotTrainedError):
        predictor.load()


def test_predictor_latest_prediction_and_signal(trained_models_dir, ohlcv):
    predictor = Predictor(models_dir=trained_models_dir)
    prediction, row = predictor.predict_latest(ohlcv)
    assert prediction.timestamp == ohlcv.index[-1].isoformat()
    assert np.isclose(prediction.prob_down + prediction.prob_flat + prediction.prob_up, 1.0)
    assert prediction.expected_high >= prediction.close >= prediction.expected_low
    assert prediction.predicted_class in {"DOWN", "FLAT", "UP"}
    assert set(prediction.components) == {"xgboost", "lightgbm"}

    signal = generate_signal(ohlcv, predictor, RiskManager(equity=5_000), threshold=0.34)
    assert signal.action in {"LONG", "SHORT", "WAIT"}
    payload = signal.to_dict()
    assert "prediction" in payload and "indicators" in payload
    if signal.action != "WAIT":
        plan = signal.trade_plan
        assert plan is not None and plan["equity"] == 5_000
        if signal.action == "LONG":
            assert plan["stop_loss"] < plan["entry_price"] < plan["take_profits"][0]["price"]
        else:
            assert plan["stop_loss"] > plan["entry_price"] > plan["take_profits"][0]["price"]

    # An impossible threshold always yields WAIT and no plan.
    wait = generate_signal(ohlcv, predictor, threshold=0.99)
    assert wait.action == "WAIT" and wait.trade_plan is None


def test_predictor_rejects_short_history(trained_models_dir, ohlcv):
    predictor = Predictor(models_dir=trained_models_dir)
    with pytest.raises(ValueError):
        predictor.predict_latest(ohlcv.iloc[:150])


@pytest.mark.parametrize("up,down,thr,expected", [
    (0.60, 0.20, 0.55, "LONG"), (0.20, 0.60, 0.55, "SHORT"), (0.50, 0.30, 0.55, "WAIT"), (0.56, 0.56, 0.55, "WAIT"),
])
def test_decide_action(up, down, thr, expected):
    assert decide_action(up, down, thr)[0] == expected
