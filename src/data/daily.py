"""Daily OHLCV for any Yahoo Finance symbol (stocks, ETFs, indices, BTC-USD) using the
public chart endpoint - no API key and no extra dependency. Cached as CSV for 6 hours."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (research; btc-ml-trader)"}


def _cache_path(symbol: str) -> Path:
    slug = symbol.replace("^", "").replace("=", "").replace(".", "_").replace("-", "_")
    return settings.data_dir / f"yahoo_{slug}_1d.csv"


def fetch_yahoo_daily(symbol: str, years: int = 6) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"range": f"{years}y", "interval": "1d", "events": "div,splits"}, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"]},
                      index=pd.to_datetime(ts, unit="s", utc=True).normalize())
    if adj:
        factor = pd.Series(adj, index=df.index) / df["close"]
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df.astype(float)


def yahoo_daily(symbol: str, years: int = 6, max_age_hours: float = 6.0) -> pd.DataFrame:
    """Cached daily bars; falls back to the cache when Yahoo is unreachable."""
    path = _cache_path(symbol)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    if not fresh:
        try:
            df = fetch_yahoo_daily(symbol, years)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            logger.info("Yahoo %s: %d daily bars (%s -> %s)", symbol, len(df), df.index[0].date(), df.index[-1].date())
            return df
        except Exception as exc:  # noqa: BLE001 - serve the cache if the network is down
            logger.warning("Yahoo %s fetch failed (%s); using cache if present", symbol, exc)
    if not path.exists():
        raise RuntimeError(f"No daily data for {symbol} and Yahoo is unreachable")
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.astype(float)
