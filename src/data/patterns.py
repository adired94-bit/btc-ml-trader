"""Candlestick pattern features (hourly and daily-aggregated), strictly causal.

Each pattern is a signed / boolean indicator computed from completed candles up
to and including bar *t*. Daily patterns are computed on UTC daily candles and
forward-filled to the hourly index, using only days that have fully closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.processor import build_extended_features

PATTERN_COLUMNS = [
    "pat_doji", "pat_hammer", "pat_shooting_star", "pat_bull_engulfing", "pat_bear_engulfing",
    "pat_inside_bar", "pat_outside_bar", "pat_three_soldiers", "pat_three_crows",
    "pat_morning_star", "pat_evening_star", "pat_bull_marubozu", "pat_bear_marubozu",
]


def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean pattern flags for every candle of ``df`` (open/high/low/close)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = body.rolling(20, min_periods=10).mean()
    bull = c > o
    bear = c < o

    o1, c1, h1, l1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    body1, body2 = (c1 - o1).abs(), (c2 - o2).abs()

    out = pd.DataFrame(index=df.index)
    out["pat_doji"] = body <= 0.1 * rng
    # Hammer / shooting star: real body (not a doji), long wick on one side, almost none on the other.
    real_body = body >= 0.1 * rng
    out["pat_hammer"] = real_body & (lower >= 2 * body) & (upper <= 0.1 * rng)
    out["pat_shooting_star"] = real_body & (upper >= 2 * body) & (lower <= 0.1 * rng)
    out["pat_bull_engulfing"] = bull & (c1 < o1) & (c >= o1) & (o <= c1) & (body > body1)
    out["pat_bear_engulfing"] = bear & (c1 > o1) & (c <= o1) & (o >= c1) & (body > body1)
    out["pat_inside_bar"] = (h <= h1) & (l >= l1)
    out["pat_outside_bar"] = (h > h1) & (l < l1)
    out["pat_three_soldiers"] = bull & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2) & (body > 0.5 * avg_body)
    out["pat_three_crows"] = bear & (c1 < o1) & (c2 < o2) & (c < c1) & (c1 < c2) & (body > 0.5 * avg_body)
    small_mid = body1 <= 0.3 * avg_body
    out["pat_morning_star"] = (c2 < o2) & (body2 > avg_body) & small_mid & bull & (c > (o2 + c2) / 2)
    out["pat_evening_star"] = (c2 > o2) & (body2 > avg_body) & small_mid & bear & (c < (o2 + c2) / 2)
    out["pat_bull_marubozu"] = bull & (body >= 0.9 * rng)
    out["pat_bear_marubozu"] = bear & (body >= 0.9 * rng)
    return out.fillna(False).astype(bool)


def daily_patterns_on_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-candle patterns aligned to the hourly index; a day's pattern becomes
    visible only from the first hour of the *next* day (no look-ahead)."""
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    pats = candlestick_patterns(daily)
    pats.index = pats.index + pd.Timedelta(days=1)  # known once the day has closed
    aligned = pats.reindex(df.index, method="ffill").fillna(False).astype(bool)
    aligned.columns = [f"d_{c}" for c in aligned.columns]
    return aligned


def build_pattern_features(df: pd.DataFrame, indicators: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extended features + hourly pattern flags, recent pattern counts and daily patterns."""
    feats = build_extended_features(df, indicators)
    hourly = candlestick_patterns(df)
    for col in PATTERN_COLUMNS:
        feats[col] = hourly[col].astype(float)
    bull_cols = ["pat_hammer", "pat_bull_engulfing", "pat_three_soldiers", "pat_morning_star", "pat_bull_marubozu"]
    bear_cols = ["pat_shooting_star", "pat_bear_engulfing", "pat_three_crows", "pat_evening_star", "pat_bear_marubozu"]
    bull_score = hourly[bull_cols].sum(axis=1).astype(float)
    bear_score = hourly[bear_cols].sum(axis=1).astype(float)
    feats["pat_bull_count_24"] = bull_score.rolling(24, min_periods=1).sum()
    feats["pat_bear_count_24"] = bear_score.rolling(24, min_periods=1).sum()
    feats["pat_net_score_24"] = feats["pat_bull_count_24"] - feats["pat_bear_count_24"]
    feats["pat_doji_count_24"] = hourly["pat_doji"].astype(float).rolling(24, min_periods=1).sum()
    daily = daily_patterns_on_hourly(df)
    for col in daily.columns:
        feats[col] = daily[col].astype(float)
    return feats.replace([np.inf, -np.inf], np.nan)
