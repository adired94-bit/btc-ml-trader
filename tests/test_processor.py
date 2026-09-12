"""Data engine tests: indicator correctness, absence of look-ahead bias, labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.data import processor as P


def test_indicator_frame_has_expected_columns(ohlcv):
    ind = P.add_all_indicators(ohlcv)
    expected = {"ema_20", "ema_50", "ema_200", "rsi_14", "atr_14", "atr_pct", "bb_upper", "bb_mid", "bb_lower",
                "bb_width", "bb_pct_b", "vwap", "vwap_24", "macd", "macd_signal", "macd_hist", "vp_poc",
                "vp_va_low", "vp_va_high", "dist_poc_pct"}
    assert expected.issubset(ind.columns)
    assert len(ind) == len(ohlcv)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    assert P.rsi(up).dropna().between(0, 100).all()
    assert np.isclose(P.rsi(up).iloc[-1], 100.0)
    assert np.isclose(P.rsi(down).iloc[-1], 0.0)
    assert P.rsi(up).iloc[:14].isna().all()


def test_atr_positive_and_scales_with_range(ohlcv):
    a = P.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    assert (a.dropna() > 0).all()
    wide = ohlcv.copy()
    wide["high"] *= 1.02
    wide["low"] *= 0.98
    assert P.atr(wide["high"], wide["low"], wide["close"]).iloc[-1] > a.iloc[-1]


def test_bollinger_ordering(ohlcv):
    bb = P.bollinger_bands(ohlcv["close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()
    assert (bb["bb_width"] >= 0).all()


def test_ema_tracks_price_and_smooths(ohlcv):
    e20 = P.ema(ohlcv["close"], 20).dropna()
    assert (e20.pct_change().abs().dropna() < ohlcv["close"].pct_change().abs().reindex(e20.index).dropna().max()).all()
    assert e20.iloc[:19].empty or True  # first 19 values are NaN by construction
    assert P.ema(ohlcv["close"], 20).iloc[:19].isna().all()


def test_vwap_is_within_daily_range(ohlcv):
    v = P.vwap(ohlcv, "D")
    day = ohlcv.index.tz_convert(None).to_period("D")
    lows = ohlcv["low"].groupby(day).cummin()
    highs = ohlcv["high"].groupby(day).cummax()
    assert ((v >= lows - 1e-6) & (v <= highs + 1e-6)).all()


def test_volume_profile_conserves_volume(ohlcv):
    vp = P.volume_profile(ohlcv, lookback=240, bins=30)
    window = ohlcv.tail(240)
    assert np.isclose(vp.volumes.sum(), window["volume"].sum(), rtol=1e-6)
    assert window["low"].min() <= vp.poc <= window["high"].max()
    assert vp.value_area_low <= vp.poc <= vp.value_area_high
    payload = vp.to_dict()
    assert len(payload["levels"]) == 30


def test_features_have_no_look_ahead(ohlcv):
    """Truncating the future must not change any feature computed for the past."""
    full = P.build_features(ohlcv)
    truncated = P.build_features(ohlcv.iloc[:-100])
    common = truncated.index[-300:]
    a = full.loc[common]
    b = truncated.loc[common]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_labels_are_forward_looking_and_consistent(ohlcv):
    horizon = settings.prediction_horizon
    labels = P.build_labels(ohlcv)
    assert (labels["direction"].iloc[-horizon:] == -1).all()
    valid = labels[labels["direction"] >= 0]
    thr = settings.direction_threshold_pct
    assert (valid.loc[valid["future_return"] > thr, "direction"] == P.UP).all()
    assert (valid.loc[valid["future_return"] < -thr, "direction"] == P.DOWN).all()
    # Max excursions can both be negative (gap down) or positive (gap up), but never crossed.
    assert (valid["future_max_up"] >= valid["future_max_down"]).all()
    assert (valid["future_max_up"] >= valid["future_return"] - 1e-12).all()
    assert (valid["future_max_down"] <= valid["future_return"] + 1e-12).all()
    # future_return must equal the realised return ``horizon`` bars later
    i = 500
    expected = ohlcv["close"].iloc[i + horizon] / ohlcv["close"].iloc[i] - 1
    assert np.isclose(labels["future_return"].iloc[i], expected)


def test_build_dataset_is_clean_and_aligned(dataset):
    X, y, ind = dataset
    assert not X.isna().any().any()
    assert not np.isinf(X.to_numpy()).any()
    assert X.index.equals(y.index) and X.index.equals(ind.index)
    assert set(y["direction"].unique()) <= {0, 1, 2}
    assert X.shape[1] >= 30


def test_latest_feature_row_matches_dataset_row(ohlcv):
    feats, ind = P.latest_feature_row(ohlcv)
    assert len(feats) == 1 and len(ind) == 1
    assert feats.index[0] == ohlcv.index[-1]
    assert not feats.isna().any().any()


def test_cross_check_against_pandas_ta(ohlcv):
    ta = pytest.importorskip("pandas_ta")
    close = ohlcv["close"].reset_index(drop=True)
    ref_rsi = ta.rsi(close, length=14)
    ours_rsi = P.rsi(close, 14)
    assert np.allclose(ref_rsi.iloc[100:], ours_rsi.iloc[100:], atol=1e-6)
    ref_atr = ta.atr(ohlcv["high"].reset_index(drop=True), ohlcv["low"].reset_index(drop=True), close, length=14)
    ours_atr = P.atr(ohlcv["high"].reset_index(drop=True), ohlcv["low"].reset_index(drop=True), close, 14)
    assert np.allclose(ref_atr.iloc[100:], ours_atr.iloc[100:], rtol=1e-4)
    ref_ema = ta.ema(close, length=20)
    ours_ema = P.ema(close, 20)
    assert np.allclose(ref_ema.iloc[200:], ours_ema.iloc[200:], rtol=1e-4)
