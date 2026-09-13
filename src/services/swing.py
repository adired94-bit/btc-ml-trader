"""Swing-horizon situational snapshot for BTC and MSTR.

Not a buy/sell predictor (direction is not predictable, see SYSTEM_LEARNINGS.md). Instead it
answers the questions that matter for a multi-week holder: where are we in the cycle, how
risky is the next week / month, which price levels matter, and how large a position fits a
given loss tolerance. Every number is computed from data available at the time of the call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.daily import yahoo_daily
from src.data.onchain import download_onchain, load_onchain
from src.data.processor import volume_profile
from src.logging_config import get_logger

logger = get_logger(__name__)

ASSETS = {"BTC": "BTC-USD", "MSTR": "MSTR"}
TRADING_DAYS = {"BTC": 365, "MSTR": 252}
_onchain_cache: dict[str, Any] = {"df": None, "loaded_at": 0.0}


@dataclass
class Sizing:
    portfolio: float
    max_loss_pct: float
    expected_move_1m_2sigma_pct: float
    max_position_pct: float
    max_position_value: float
    note: str


@dataclass
class SwingSnapshot:
    asset: str
    symbol: str
    as_of: str
    price: float
    changes_pct: dict[str, float]
    trend: dict[str, Any]
    valuation: dict[str, Any]
    sentiment: dict[str, Any]
    liquidity: dict[str, Any]
    premium: dict[str, Any]
    regime: dict[str, Any]
    risk: dict[str, Any]
    levels: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_onchain(max_age_hours: float = 12.0) -> pd.DataFrame | None:
    """Daily on-chain / sentiment frame, refreshed at most every ``max_age_hours``."""
    now = time.time()
    if _onchain_cache["df"] is not None and now - _onchain_cache["loaded_at"] < max_age_hours * 3600:
        return _onchain_cache["df"]
    df = load_onchain()
    stale = df is None or (pd.Timestamp.utcnow().normalize() - df.index[-1]).days > 2
    if stale:
        try:
            df = download_onchain(years=6)
        except Exception as exc:  # noqa: BLE001 - offline: keep whatever we have
            logger.warning("On-chain refresh failed: %s", exc)
    _onchain_cache["df"], _onchain_cache["loaded_at"] = df, now
    return df


def _pct(a: float, b: float) -> float:
    return float((a / b - 1) * 100) if b else 0.0


def _percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s < value).mean() * 100) if len(s) else float("nan")


def trend_block(px: pd.DataFrame) -> dict[str, Any]:
    c = px["close"]
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    slope50 = _pct(ema50.iloc[-1], ema50.iloc[-22])
    above200 = c.iloc[-1] > ema200.iloc[-1]
    golden = ema50.iloc[-1] > ema200.iloc[-1]
    if above200 and slope50 > 0 and golden:
        state, score = "UP", 1
    elif not above200 and slope50 < 0 and not golden:
        state, score = "DOWN", -1
    else:
        state, score = "MIXED", 0
    return {
        "ema50": float(ema50.iloc[-1]), "ema200": float(ema200.iloc[-1]),
        "price_vs_ema200_pct": _pct(c.iloc[-1], ema200.iloc[-1]), "ema50_slope_21d_pct": slope50,
        "golden_cross": bool(golden), "state": state, "score": score,
    }


def valuation_block(onchain: pd.DataFrame | None) -> dict[str, Any]:
    if onchain is None or "cm_mvrv" not in onchain or onchain["cm_mvrv"].dropna().empty:
        return {"available": False, "score": 0}
    mvrv = onchain["cm_mvrv"].dropna()
    val = float(mvrv.iloc[-1])
    pct = _percentile(mvrv, val)
    if val < 1.0 or pct < 10:
        zone, score = "CHEAP", 1
    elif val > 3.0 or pct > 90:
        zone, score = "HOT", -1
    elif val > 2.4 or pct > 75:
        zone, score = "ELEVATED", 0
    else:
        zone, score = "NORMAL", 0
    return {"available": True, "mvrv": val, "percentile_6y": pct, "zone": zone, "score": score, "as_of": mvrv.index[-1].date().isoformat()}


def sentiment_block(onchain: pd.DataFrame | None) -> dict[str, Any]:
    if onchain is None or "fear_greed" not in onchain or onchain["fear_greed"].dropna().empty:
        return {"available": False, "score": 0}
    fg = onchain["fear_greed"].dropna()
    val = float(fg.iloc[-1])
    label = "EXTREME_FEAR" if val <= 25 else "FEAR" if val <= 45 else "NEUTRAL" if val < 55 else "GREED" if val < 75 else "EXTREME_GREED"
    score = 1 if val <= 25 else -1 if val >= 75 else 0  # contrarian
    return {"available": True, "fear_greed": val, "label": label, "avg_30d": float(fg.tail(30).mean()), "score": score}


def liquidity_block(onchain: pd.DataFrame | None) -> dict[str, Any]:
    if onchain is None or "cm_stablecoin_cap" not in onchain or onchain["cm_stablecoin_cap"].dropna().empty:
        return {"available": False, "score": 0}
    cap = onchain["cm_stablecoin_cap"].dropna()
    chg30 = _pct(cap.iloc[-1], cap.iloc[max(0, len(cap) - 31)])
    chg90 = _pct(cap.iloc[-1], cap.iloc[max(0, len(cap) - 91)])
    score = 1 if chg30 > 2 else -1 if chg30 < -2 else 0
    return {"available": True, "stablecoin_cap_usd": float(cap.iloc[-1]), "chg_30d_pct": chg30, "chg_90d_pct": chg90, "score": score}


def premium_block(px: pd.DataFrame, btc: pd.DataFrame) -> dict[str, Any]:
    """MSTR-only: how stretched MSTR is relative to BTC (ratio z-score over 1 year)."""
    ratio = (px["close"] / btc["close"].reindex(px.index, method="ffill")).dropna()
    if len(ratio) < 120:
        return {"available": False, "score": 0}
    window = ratio.tail(252)
    z = float((ratio.iloc[-1] - window.mean()) / window.std()) if window.std() > 0 else 0.0
    rel_21 = _pct(ratio.iloc[-1], ratio.iloc[-22])
    score = -1 if z > 1.5 else 1 if z < -1.5 else 0
    return {"available": True, "mstr_btc_ratio": float(ratio.iloc[-1]), "ratio_z_1y": z, "ratio_chg_21d_pct": rel_21, "score": score}


def risk_block(px: pd.DataFrame, asset: str, btc: pd.DataFrame | None) -> dict[str, Any]:
    c = px["close"]
    lr = np.log(c / c.shift(1)).dropna()
    vol_d = float(lr.tail(21).std())
    ann = np.sqrt(TRADING_DAYS[asset])
    tr = pd.concat([px["high"] - px["low"], (px["high"] - c.shift(1)).abs(), (px["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    move_1w, move_1m = vol_d * np.sqrt(5) * 100, vol_d * np.sqrt(21) * 100
    price = float(c.iloc[-1])
    year = c.tail(TRADING_DAYS[asset])
    dd_1y = float((year / year.cummax() - 1).min() * 100)
    out = {
        "vol_21d_annualised_pct": vol_d * ann * 100, "vol_daily_pct": vol_d * 100, "atr_daily": atr, "atr_pct": atr / price * 100,
        "expected_move_1w_pct": move_1w, "expected_move_1m_pct": move_1m,
        "range_1w": [price * (1 - move_1w / 100), price * (1 + move_1w / 100)],
        "range_1m": [price * (1 - move_1m / 100), price * (1 + move_1m / 100)],
        "range_1m_2sigma": [price * (1 - 2 * move_1m / 100), price * (1 + 2 * move_1m / 100)],
        "max_drawdown_1y_pct": dd_1y, "drawdown_from_ath_pct": _pct(price, float(c.max())),
        "vol_percentile_1y": _percentile(lr.rolling(21).std().tail(TRADING_DAYS[asset]), vol_d),
    }
    if btc is not None and asset != "BTC":
        b = np.log(btc["close"] / btc["close"].shift(1)).reindex(px.index).dropna()
        a = lr.reindex(b.index).dropna()
        b = b.reindex(a.index)
        if len(a) > 63:
            cov = a.tail(63).cov(b.tail(63))
            out["beta_to_btc_63d"] = float(cov / b.tail(63).var()) if b.tail(63).var() > 0 else float("nan")
            out["corr_to_btc_63d"] = float(a.tail(63).corr(b.tail(63)))
    return out


def levels_block(px: pd.DataFrame, asset: str) -> dict[str, float]:
    c, h, l = px["close"], px["high"], px["low"]
    n = TRADING_DAYS[asset]
    vp = volume_profile(px.tail(120), lookback=120, bins=30)
    return {
        "high_52w": float(h.tail(n).max()), "low_52w": float(l.tail(n).min()),
        "high_20d": float(h.tail(20).max()), "low_20d": float(l.tail(20).min()),
        "ema50": float(c.ewm(span=50, adjust=False).mean().iloc[-1]), "ema200": float(c.ewm(span=200, adjust=False).mean().iloc[-1]),
        "poc_120d": float(vp.poc), "value_area_low_120d": float(vp.value_area_low), "value_area_high_120d": float(vp.value_area_high),
        "all_time_high": float(c.max()),
    }


def regime_block(asset: str, trend: dict, valuation: dict, sentiment: dict, liquidity: dict, premium: dict) -> dict[str, Any]:
    parts = {"trend": trend["score"], "valuation": valuation.get("score", 0), "sentiment": sentiment.get("score", 0), "liquidity": liquidity.get("score", 0)}
    if asset == "MSTR":
        parts["premium"] = premium.get("score", 0)
    score = int(sum(parts.values()))
    if score >= 2:
        label = "BULL"
    elif score <= -2:
        label = "BEAR"
    else:
        label = "NEUTRAL"
    reasons = []
    reasons.append({"UP": "מגמה עולה: המחיר מעל ממוצע 200 יום והממוצע ל-50 יום עולה", "DOWN": "מגמה יורדת: המחיר מתחת לממוצע 200 יום והממוצע ל-50 יום יורד", "MIXED": "מגמה מעורבת: אין הסכמה בין המחיר לממוצעים"}[trend["state"]])
    if valuation.get("available"):
        reasons.append({"CHEAP": f"MVRV {valuation['mvrv']:.2f}: ביטקוין מתחת לשווי הממומש, אזור שהיסטורית סימן תחתיות",
                        "HOT": f"MVRV {valuation['mvrv']:.2f}: אזור שהיסטורית סימן שיאים (אחוזון {valuation['percentile_6y']:.0f})",
                        "ELEVATED": f"MVRV {valuation['mvrv']:.2f}: מעל הממוצע אבל לא קיצוני",
                        "NORMAL": f"MVRV {valuation['mvrv']:.2f}: טווח נורמלי"}[valuation["zone"]])
    if sentiment.get("available"):
        reasons.append(f"פחד/חמדנות {sentiment['fear_greed']:.0f} ({sentiment['label'].replace('_', ' ').lower()})" + (": פחד קיצוני, היסטורית הזדמנות" if sentiment["score"] == 1 else ": חמדנות קיצונית, היסטורית סיכון" if sentiment["score"] == -1 else ""))
    if liquidity.get("available"):
        reasons.append(f"היצע סטייבלקוינים {liquidity['chg_30d_pct']:+.1f}% ב-30 יום" + (": כסף נכנס לשוק" if liquidity["score"] == 1 else ": כסף יוצא מהשוק" if liquidity["score"] == -1 else ""))
    if asset == "MSTR" and premium.get("available"):
        reasons.append(f"יחס MSTR/BTC בסטיית תקן {premium['ratio_z_1y']:+.1f} מהממוצע השנתי" + (": הפרמיה מתוחה" if premium["score"] == -1 else ": הפרמיה נמוכה" if premium["score"] == 1 else ""))
    return {"score": score, "min_score": -len(parts), "max_score": len(parts), "label": label, "components": parts, "reasons": reasons}


def position_sizing(portfolio: float, max_loss_pct: float, expected_move_1m_pct: float) -> Sizing:
    """Largest position whose 2-sigma one-month move loses at most ``max_loss_pct`` of the portfolio."""
    two_sigma = 2 * expected_move_1m_pct
    frac = min(1.0, (max_loss_pct / two_sigma)) if two_sigma > 0 else 1.0
    return Sizing(portfolio=portfolio, max_loss_pct=max_loss_pct, expected_move_1m_2sigma_pct=two_sigma,
                  max_position_pct=frac * 100, max_position_value=portfolio * frac,
                  note="גודל הפוזיציה שבה ירידה חודשית של 2 סטיות תקן מפסידה לכל היותר את אחוז ההפסד שהגדרת")


def build_snapshot(asset: str, history_days: int = 365) -> SwingSnapshot:
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset {asset}; choose from {list(ASSETS)}")
    px = yahoo_daily(ASSETS[asset])
    btc = px if asset == "BTC" else yahoo_daily(ASSETS["BTC"])
    onchain = get_onchain()
    c = px["close"]
    trend = trend_block(px)
    valuation = valuation_block(onchain)
    sentiment = sentiment_block(onchain)
    liquidity = liquidity_block(onchain)
    premium = premium_block(px, btc) if asset == "MSTR" else {"available": False, "score": 0}
    regime = regime_block(asset, trend, valuation, sentiment, liquidity, premium)
    risk = risk_block(px, asset, btc if asset != "BTC" else None)
    levels = levels_block(px, asset)
    hist = px.tail(history_days).copy()
    hist["ema50"] = c.ewm(span=50, adjust=False).mean().tail(history_days)
    hist["ema200"] = c.ewm(span=200, adjust=False).mean().tail(history_days)
    if onchain is not None:
        for col in ("cm_mvrv", "fear_greed"):
            if col in onchain:
                hist[col] = onchain[col].reindex(hist.index, method="ffill")
    history = [{"date": d.date().isoformat(), **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}} for d, row in hist.iterrows()]
    return SwingSnapshot(
        asset=asset, symbol=ASSETS[asset], as_of=px.index[-1].date().isoformat(), price=float(c.iloc[-1]),
        changes_pct={"1d": _pct(c.iloc[-1], c.iloc[-2]), "1w": _pct(c.iloc[-1], c.iloc[-6]), "1m": _pct(c.iloc[-1], c.iloc[-22]), "3m": _pct(c.iloc[-1], c.iloc[-64]), "1y": _pct(c.iloc[-1], c.iloc[-min(len(c), TRADING_DAYS[asset] + 1)])},
        trend=trend, valuation=valuation, sentiment=sentiment, liquidity=liquidity, premium=premium, regime=regime, risk=risk, levels=levels, history=history,
    )
