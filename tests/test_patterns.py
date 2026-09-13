"""Candlestick pattern feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import patterns as P


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def test_engulfing_hammer_doji_detection():
    rows = [(100, 101, 99, 100.5)] * 20  # warm-up
    rows += [(100, 100.5, 99, 99.2)]  # bearish candle
    rows += [(99, 101.5, 98.9, 101.2)]  # bullish engulfing
    rows += [(101, 101.1, 97, 100.5)]  # hammer (long lower wick, small body, no upper wick)
    rows += [(100, 101, 99, 100.02)]  # doji
    df = _bars(rows)
    p = P.candlestick_patterns(df)
    assert bool(p["pat_bull_engulfing"].iloc[21])
    assert bool(p["pat_hammer"].iloc[22])
    assert bool(p["pat_doji"].iloc[23])
    assert not bool(p["pat_bear_engulfing"].iloc[21])


def test_three_soldiers_and_inside_bar():
    rows = [(100, 101, 99, 100.5)] * 20
    rows += [(100, 102, 99.5, 101.8), (101.8, 104, 101.5, 103.7), (103.7, 106, 103.4, 105.6)]
    rows += [(105, 105.5, 104.5, 105.2)]  # inside previous bar
    df = _bars(rows)
    p = P.candlestick_patterns(df)
    assert bool(p["pat_three_soldiers"].iloc[22])
    assert bool(p["pat_inside_bar"].iloc[23])
    assert set(p.columns) == set(P.PATTERN_COLUMNS)
    assert p.dtypes.eq(bool).all()


def test_pattern_features_are_causal(ohlcv):
    full = P.build_pattern_features(ohlcv)
    short = P.build_pattern_features(ohlcv.iloc[:-100])
    common = short.index[-200:]
    pd.testing.assert_frame_equal(full.loc[common], short.loc[common], check_exact=False, rtol=1e-9, atol=1e-12)
    assert full.iloc[-1].notna().all()
    assert any(c.startswith("d_pat_") for c in full.columns)


def test_daily_patterns_visible_only_after_day_close(ohlcv):
    daily = P.daily_patterns_on_hourly(ohlcv)
    # Every hour of a given day carries the same value, and it equals the previous day's pattern.
    day = ohlcv.index[-30].floor("D")
    same_day = daily[daily.index.floor("D") == day]
    assert (same_day.nunique() <= 1).all()
    assert daily.shape[0] == len(ohlcv)
    assert daily.dtypes.eq(bool).all()
    assert daily.iloc[0].eq(False).all()  # nothing known before the first full day closes
