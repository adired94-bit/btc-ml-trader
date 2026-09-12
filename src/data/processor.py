"""Technical indicators, feature engineering and supervised labels.

Indicators are implemented natively in pandas/numpy so the numerical
definitions are explicit, deterministic and independent of third-party
versioning. ``pandas_ta`` is used as an *independent cross-check* in the unit
tests (see ``tests/test_processor.py``) when it is installed.

Every indicator/feature at bar *t* only uses information up to and including
bar *t* - there is no look-ahead bias. Labels are the only forward-looking
quantities and are kept in a separate frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings

# ======================================================================
# Indicators
# ======================================================================


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (exponentially smoothed gains / losses)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)  # no losses at all -> RSI 100
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b}
    )


def vwap(df: pd.DataFrame, anchor: str = "D") -> pd.Series:
    """Volume-weighted average price, reset at each ``anchor`` period (daily by default)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) is not None else df.index
    groups = idx.to_period(anchor)
    cum_pv = pv.groupby(groups).cumsum()
    cum_vol = df["volume"].groupby(groups).cumsum()
    return (cum_pv / cum_vol.replace(0.0, np.nan)).rename("vwap")


def rolling_vwap(df: pd.DataFrame, period: int = 24) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(period, min_periods=period).sum()
    vol = df["volume"].rolling(period, min_periods=period).sum()
    return (pv / vol.replace(0.0, np.nan)).rename(f"vwap_{period}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


@dataclass
class VolumeProfile:
    """Volume distribution across price bins for a look-back window."""

    bin_edges: np.ndarray
    volumes: np.ndarray
    poc: float  # point of control - price level with the most traded volume
    value_area_low: float
    value_area_high: float

    def to_dict(self) -> dict:
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        return {
            "levels": [
                {"price": float(p), "volume": float(v)} for p, v in zip(centers, self.volumes)
            ],
            "poc": float(self.poc),
            "value_area_low": float(self.value_area_low),
            "value_area_high": float(self.value_area_high),
        }


def volume_profile(
    df: pd.DataFrame, lookback: int = 240, bins: int = 30, value_area: float = 0.70
) -> VolumeProfile:
    """Compute a volume profile over the last ``lookback`` bars.

    Each candle's volume is spread uniformly over the price range it covered so
    that wide candles contribute to several bins.
    """
    window = df.tail(lookback)
    lo, hi = float(window["low"].min()), float(window["high"].max())
    if hi <= lo:
        hi = lo * 1.0001 + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    volumes = np.zeros(bins)
    lows = window["low"].to_numpy()
    highs = window["high"].to_numpy()
    vols = window["volume"].to_numpy()
    for low_px, high_px, vol in zip(lows, highs, vols):
        if vol <= 0:
            continue
        span = max(high_px - low_px, 1e-9)
        first = int(np.clip(np.searchsorted(edges, low_px, side="right") - 1, 0, bins - 1))
        last = int(np.clip(np.searchsorted(edges, high_px, side="right") - 1, 0, bins - 1))
        for b in range(first, last + 1):
            overlap = min(high_px, edges[b + 1]) - max(low_px, edges[b])
            if overlap > 0:
                volumes[b] += vol * overlap / span
    poc_idx = int(volumes.argmax())
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)

    # Value area: expand outward from the POC until ``value_area`` of volume is covered.
    total = volumes.sum()
    covered = volumes[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while total > 0 and covered / total < value_area and (lo_i > 0 or hi_i < bins - 1):
        down = volumes[lo_i - 1] if lo_i > 0 else -1.0
        up = volumes[hi_i + 1] if hi_i < bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            covered += up
        else:
            lo_i -= 1
            covered += down
    return VolumeProfile(edges, volumes, poc, float(edges[lo_i]), float(edges[hi_i + 1]))


def volume_profile_features(df: pd.DataFrame, lookback: int = 240, bins: int = 30) -> pd.DataFrame:
    """Rolling POC / value-area distance features computed causally for every bar.

    The profile is recomputed every ``step`` bars and forward-filled; the POC
    moves slowly so the approximation is negligible and keeps the pipeline fast.
    """
    step = max(1, lookback // 20)
    idx = df.index
    poc = np.full(len(df), np.nan)
    va_lo = np.full(len(df), np.nan)
    va_hi = np.full(len(df), np.nan)
    for end in range(lookback, len(df) + 1, step):
        vp = volume_profile(df.iloc[end - lookback:end], lookback, bins)
        poc[end - 1] = vp.poc
        va_lo[end - 1] = vp.value_area_low
        va_hi[end - 1] = vp.value_area_high
    poc_s = pd.Series(poc, index=idx).ffill()
    va_lo_s = pd.Series(va_lo, index=idx).ffill()
    va_hi_s = pd.Series(va_hi, index=idx).ffill()
    close = df["close"]
    return pd.DataFrame(
        {
            "vp_poc": poc_s,
            "vp_va_low": va_lo_s,
            "vp_va_high": va_hi_s,
            "dist_poc_pct": (close - poc_s) / poc_s,
            "dist_va_low_pct": (close - va_lo_s) / va_lo_s,
            "dist_va_high_pct": (close - va_hi_s) / va_hi_s,
        }
    )


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with every indicator used by the platform."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)
    out["rsi_14"] = rsi(close, 14)
    out["atr_14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close
    out = out.join(bollinger_bands(close, 20, 2.0))
    out["vwap"] = vwap(out, "D")
    out["vwap_24"] = rolling_vwap(out, 24)
    out = out.join(macd(close))
    out = out.join(volume_profile_features(out))
    return out


# ======================================================================
# Feature engineering & labels
# ======================================================================

# Class encoding for the direction classifier.
DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}


def build_features(df: pd.DataFrame, indicators: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute scale-free model features for every bar (all strictly causal)."""
    ind = indicators if indicators is not None else add_all_indicators(df)
    close = ind["close"]
    feats = pd.DataFrame(index=ind.index)

    # Momentum / returns
    for lag in (1, 2, 4, 8, 24):
        feats[f"ret_{lag}"] = close.pct_change(lag)
    feats["log_ret_1"] = np.log(close / close.shift(1))
    feats["volatility_24"] = feats["log_ret_1"].rolling(24).std()
    feats["volatility_72"] = feats["log_ret_1"].rolling(72).std()

    # Trend structure (distance to moving averages, normalised by price)
    for p in (20, 50, 200):
        feats[f"dist_ema_{p}"] = (close - ind[f"ema_{p}"]) / close
    feats["ema_20_50_spread"] = (ind["ema_20"] - ind["ema_50"]) / close
    feats["ema_50_200_spread"] = (ind["ema_50"] - ind["ema_200"]) / close
    feats["ema_20_slope"] = ind["ema_20"].pct_change(4)

    # Oscillators
    feats["rsi_14"] = ind["rsi_14"] / 100.0
    feats["rsi_14_change"] = ind["rsi_14"].diff(3) / 100.0
    feats["macd_hist_norm"] = ind["macd_hist"] / close
    feats["macd_norm"] = ind["macd"] / close

    # Volatility
    feats["atr_pct"] = ind["atr_pct"]
    feats["atr_pct_change"] = ind["atr_pct"].pct_change(12)
    feats["bb_width"] = ind["bb_width"]
    feats["bb_pct_b"] = ind["bb_pct_b"]

    # Volume-based
    feats["dist_vwap"] = (close - ind["vwap"]) / close
    feats["dist_vwap_24"] = (close - ind["vwap_24"]) / close
    vol_ma = ind["volume"].rolling(24).mean()
    feats["volume_ratio_24"] = ind["volume"] / vol_ma.replace(0.0, np.nan)
    vol_std_72 = ind["volume"].rolling(72).std().replace(0.0, np.nan)
    feats["volume_z_72"] = (ind["volume"] - ind["volume"].rolling(72).mean()) / vol_std_72
    feats["dist_poc_pct"] = ind["dist_poc_pct"]
    feats["dist_va_low_pct"] = ind["dist_va_low_pct"]
    feats["dist_va_high_pct"] = ind["dist_va_high_pct"]

    # Candle anatomy
    rng = (ind["high"] - ind["low"]).replace(0.0, np.nan)
    feats["body_ratio"] = (ind["close"] - ind["open"]) / rng
    feats["upper_wick_ratio"] = (ind["high"] - ind[["open", "close"]].max(axis=1)) / rng
    feats["lower_wick_ratio"] = (ind[["open", "close"]].min(axis=1) - ind["low"]) / rng

    # Calendar seasonality
    hours = ind.index.hour.to_numpy()
    dows = ind.index.dayofweek.to_numpy()
    feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dows / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dows / 7)

    return feats.replace([np.inf, -np.inf], np.nan)


def build_extended_features(df: pd.DataFrame, indicators: pd.DataFrame | None = None) -> pd.DataFrame:
    """Base features plus multi-day context (all causal). Used by the daily-horizon experiments."""
    ind = indicators if indicators is not None else add_all_indicators(df)
    feats = build_features(df, ind)
    close = ind["close"]
    for lag in (48, 72, 168, 336, 720):
        feats[f"ret_{lag}"] = close.pct_change(lag)
    log_ret = np.log(close / close.shift(1))
    feats["volatility_168"] = log_ret.rolling(168).std()
    feats["volatility_720"] = log_ret.rolling(720).std()
    feats["vol_ratio_24_168"] = feats["volatility_24"] / feats["volatility_168"].replace(0.0, np.nan)
    feats["vol_ratio_72_720"] = feats["volatility_72"] / feats["volatility_720"].replace(0.0, np.nan)
    high_30d = ind["high"].rolling(720, min_periods=240).max()
    low_30d = ind["low"].rolling(720, min_periods=240).min()
    feats["dist_high_30d"] = close / high_30d - 1.0
    feats["dist_low_30d"] = close / low_30d - 1.0
    feats["range_pos_30d"] = (close - low_30d) / (high_30d - low_30d).replace(0.0, np.nan)
    high_7d = ind["high"].rolling(168, min_periods=100).max()
    low_7d = ind["low"].rolling(168, min_periods=100).min()
    feats["range_pos_7d"] = (close - low_7d) / (high_7d - low_7d).replace(0.0, np.nan)
    feats["volume_ratio_24_168"] = ind["volume"].rolling(24).mean() / ind["volume"].rolling(168).mean().replace(0.0, np.nan)
    feats["rsi_daily_proxy"] = rsi(close, 14 * 24) / 100.0
    feats["ema_200_slope_24"] = ind["ema_200"].pct_change(24)
    feats["ema_50_slope_24"] = ind["ema_50"].pct_change(24)
    feats["up_days_7"] = (close.pct_change(24) > 0).astype(float).rolling(7 * 24, min_periods=24).mean()
    dom = ind.index.day.to_numpy()
    feats["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    feats["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    return feats.replace([np.inf, -np.inf], np.nan)


def build_labels(
    df: pd.DataFrame, horizon: int | None = None, threshold: float | None = None
) -> pd.DataFrame:
    """Forward-looking targets aligned to each bar's close.

    * ``direction``: 0=DOWN, 1=FLAT, 2=UP from the return ``horizon`` bars ahead
      (-1 marks the trailing rows whose future is not yet known).
    * ``future_return``: fractional close-to-close return over the horizon.
    * ``future_max_up`` / ``future_max_down``: maximum favourable / adverse
      excursion (fraction of the entry close) within the horizon.
    """
    horizon = horizon or settings.prediction_horizon
    threshold = threshold if threshold is not None else settings.direction_threshold_pct
    close = df["close"]
    future_return = close.shift(-horizon) / close - 1.0

    # Forward-looking max/min of the next ``horizon`` bars (excluding the current one).
    fut_high = df["high"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    fut_low = df["low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    future_max_up = fut_high / close - 1.0
    future_max_down = fut_low / close - 1.0

    direction = pd.Series(FLAT, index=df.index, dtype=int)
    direction[future_return > threshold] = UP
    direction[future_return < -threshold] = DOWN
    direction[future_return.isna() | future_max_up.isna()] = -1

    return pd.DataFrame(
        {
            "direction": direction,
            "future_return": future_return,
            "future_max_up": future_max_up,
            "future_max_down": future_max_down,
        }
    )


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(features, labels, indicators)`` restricted to fully-defined rows."""
    indicators = add_all_indicators(df)
    features = build_features(df, indicators)
    labels = build_labels(df)
    mask = features.notna().all(axis=1) & (labels["direction"] >= 0) & labels.notna().all(axis=1)
    return features[mask], labels[mask], indicators[mask]


def latest_feature_row(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features + indicators for the most recent closed bar (no labels required)."""
    indicators = add_all_indicators(df)
    features = build_features(df, indicators)
    return features.iloc[[-1]], indicators.iloc[[-1]]
