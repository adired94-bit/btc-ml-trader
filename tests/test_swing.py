"""Swing snapshot tests on synthetic daily data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services import swing as S


def _daily(n: int = 800, drift: float = 0.001, seed: int = 1, start: float = 100.0, noise: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, noise, n)
    close = start * np.exp(np.cumsum(r))
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(start)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    df["volume"] = rng.lognormal(10, 0.3, n)
    return df


def _onchain(idx: pd.DatetimeIndex, mvrv: float, fg: float, stable_chg: float) -> pd.DataFrame:
    n = len(idx)
    cap = np.full(n, 1e11)
    cap[-31:] = 1e11 * np.linspace(1, 1 + stable_chg, 31)  # the change happens inside the last 30 days
    return pd.DataFrame({"cm_mvrv": np.linspace(1.5, mvrv, n), "fear_greed": np.full(n, fg), "cm_stablecoin_cap": cap}, index=idx)


def test_trend_states():
    assert S.trend_block(_daily(drift=0.006))["state"] == "UP"
    assert S.trend_block(_daily(drift=-0.006))["state"] == "DOWN"


def test_valuation_sentiment_liquidity_scores():
    px = _daily()
    hot = _onchain(px.index, 3.5, 85, 0.30)
    assert S.valuation_block(hot)["zone"] == "HOT" and S.valuation_block(hot)["score"] == -1
    assert S.sentiment_block(hot)["label"] == "EXTREME_GREED" and S.sentiment_block(hot)["score"] == -1
    assert S.liquidity_block(hot)["score"] == 1
    cheap = _onchain(px.index, 0.8, 15, -0.10)
    assert S.valuation_block(cheap)["zone"] == "CHEAP"
    assert S.sentiment_block(cheap)["score"] == 1
    assert S.liquidity_block(cheap)["score"] == -1
    assert S.valuation_block(None) == {"available": False, "score": 0}


def test_regime_composition():
    px = _daily(drift=0.006)
    oc = _onchain(px.index, 1.2, 20, 0.05)
    regime = S.regime_block("BTC", S.trend_block(px), S.valuation_block(oc), S.sentiment_block(oc), S.liquidity_block(oc), {"available": False, "score": 0})
    assert regime["label"] == "BULL" and regime["score"] >= 2 and len(regime["reasons"]) == 4
    bear = S.regime_block("BTC", S.trend_block(_daily(drift=-0.006)), S.valuation_block(_onchain(px.index, 3.5, 90, -0.05)), S.sentiment_block(_onchain(px.index, 3.5, 90, -0.05)), S.liquidity_block(_onchain(px.index, 3.5, 90, -0.05)), {"available": False, "score": 0})
    assert bear["label"] == "BEAR"


def test_risk_levels_and_sizing():
    px = _daily()
    risk = S.risk_block(px, "BTC", None)
    assert risk["expected_move_1m_pct"] > risk["expected_move_1w_pct"] > 0
    assert risk["range_1m"][0] < px["close"].iloc[-1] < risk["range_1m"][1]
    assert -100 <= risk["max_drawdown_1y_pct"] <= 0
    levels = S.levels_block(px, "BTC")
    assert levels["low_52w"] <= levels["poc_120d"] <= levels["high_52w"]
    sizing = S.position_sizing(10_000, 10.0, risk["expected_move_1m_pct"])
    assert 0 < sizing.max_position_pct <= 100 and sizing.max_position_value == pytest.approx(10_000 * sizing.max_position_pct / 100)
    assert S.position_sizing(10_000, 50.0, 5.0).max_position_pct == 100


def test_premium_and_mstr_beta():
    btc = _daily(drift=0.002, seed=3)
    mstr = btc.copy()
    mstr["close"] = btc["close"] ** 1.5 / 10  # amplified proxy
    mstr["high"], mstr["low"], mstr["open"] = mstr["close"] * 1.02, mstr["close"] * 0.98, mstr["close"].shift(1).bfill()
    prem = S.premium_block(mstr, btc)
    assert prem["available"] and "ratio_z_1y" in prem
    risk = S.risk_block(mstr, "MSTR", btc)
    assert risk["beta_to_btc_63d"] > 1.0 and risk["corr_to_btc_63d"] > 0.9


def test_build_snapshot_with_stubbed_data(monkeypatch):
    px = _daily(drift=0.002)
    monkeypatch.setattr(S, "yahoo_daily", lambda symbol, **kw: px)
    monkeypatch.setattr(S, "get_onchain", lambda **kw: _onchain(px.index, 1.4, 60, 0.03))
    snap = S.build_snapshot("BTC")
    d = snap.to_dict()
    assert d["price"] == pytest.approx(float(px["close"].iloc[-1]))
    assert set(d["changes_pct"]) == {"1d", "1w", "1m", "3m", "1y"}
    assert d["regime"]["label"] in {"BULL", "BEAR", "NEUTRAL"}
    assert len(d["history"]) == 365 and "ema200" in d["history"][-1]
    with pytest.raises(ValueError):
        S.build_snapshot("DOGE")
