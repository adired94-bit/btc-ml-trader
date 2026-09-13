"""Alternative market data from Binance (free, multi-year): order-flow fields of spot
klines (taker buy volume, trade count), USDT-perpetual volume, premium index (basis)
and funding rates. Everything is aligned to the hourly spot index and used causally.

    python -m src.data.altdata --years 6      # download / refresh data/BTC_USDT_1h_alt.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from config import settings
from src.data.processor import build_extended_features
from src.logging_config import get_logger

logger = get_logger(__name__)

HOUR_MS = 3_600_000
ALT_COLUMNS = ["quote_volume", "trades", "taker_buy_base", "perp_close", "perp_volume", "premium_close", "funding_rate"]


def alt_cache_path() -> Path:
    return settings.data_dir / f"{settings.symbol_slug}_{settings.timeframe}_alt.csv"


def _binance_symbol() -> str:
    return settings.symbol.replace("/", "")


def _paginate(fetch, since_ms: int, until_ms: int, step_ms: int, label: str) -> list[list]:
    rows: list[list] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = fetch(cursor)
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1][0])
        if last <= cursor:
            break
        cursor = last + step_ms
        if len(rows) % 10_000 < len(batch):
            logger.info("%s: %d rows so far", label, len(rows))
    return rows


def fetch_spot_flow(years: float) -> pd.DataFrame:
    """Spot klines with taker-buy volume, trade count and quote volume."""
    client = ccxt.binance({"enableRateLimit": True, "timeout": settings.request_timeout_ms})
    now = client.milliseconds()
    since = now - int(years * 365 * 24) * HOUR_MS
    sym = _binance_symbol()
    raw = _paginate(
        lambda c: client.publicGetKlines({"symbol": sym, "interval": "1h", "startTime": c, "limit": 1000}),
        since, now, HOUR_MS, "spot flow",
    )
    df = pd.DataFrame(raw).iloc[:, [0, 7, 8, 9]]
    df.columns = ["timestamp", "quote_volume", "trades", "taker_buy_base"]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    return df.drop_duplicates("timestamp").set_index("timestamp").astype(float).sort_index()


def fetch_perp(years: float) -> pd.DataFrame:
    """USDT-margined perpetual hourly close/volume and premium-index close (basis)."""
    client = ccxt.binanceusdm({"enableRateLimit": True, "timeout": settings.request_timeout_ms})
    now = client.milliseconds()
    since = now - int(years * 365 * 24) * HOUR_MS
    sym = _binance_symbol()
    ohlcv = _paginate(
        lambda c: client.fapiPublicGetKlines({"symbol": sym, "interval": "1h", "startTime": c, "limit": 1000}),
        since, now, HOUR_MS, "perp klines",
    )
    perp = pd.DataFrame(ohlcv).iloc[:, [0, 4, 5]]
    perp.columns = ["timestamp", "perp_close", "perp_volume"]
    premium = _paginate(
        lambda c: client.fapiPublicGetPremiumIndexKlines({"symbol": sym, "interval": "1h", "startTime": c, "limit": 1000}),
        since, now, HOUR_MS, "premium index",
    )
    prem = pd.DataFrame(premium).iloc[:, [0, 4]]
    prem.columns = ["timestamp", "premium_close"]
    out = perp.merge(prem, on="timestamp", how="outer")
    out["timestamp"] = pd.to_datetime(out["timestamp"].astype("int64"), unit="ms", utc=True)
    return out.drop_duplicates("timestamp").set_index("timestamp").astype(float).sort_index()


def fetch_funding(years: float) -> pd.DataFrame:
    client = ccxt.binanceusdm({"enableRateLimit": True, "timeout": settings.request_timeout_ms})
    now = client.milliseconds()
    since = now - int(years * 365 * 24) * HOUR_MS
    market = f"{settings.symbol}:USDT"
    rows: list[tuple[int, float]] = []
    cursor = since
    while cursor < now:
        batch = client.fetch_funding_rate_history(market, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend((int(b["timestamp"]), float(b["fundingRate"])) for b in batch)
        last = int(batch[-1]["timestamp"])
        if last <= cursor:
            break
        cursor = last + 1
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("h")
    return df.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def download_alt_data(years: float = 6.0, path: Path | None = None) -> pd.DataFrame:
    path = path or alt_cache_path()
    t0 = time.perf_counter()
    spot = fetch_spot_flow(years)
    perp = fetch_perp(years)
    funding = fetch_funding(years)
    alt = spot.join(perp, how="outer").join(funding, how="outer").sort_index()
    alt = alt[~alt.index.duplicated(keep="last")]
    alt["funding_rate"] = alt["funding_rate"].ffill()  # settled rate carried forward - causal
    alt = alt[ALT_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    alt.to_csv(path, index_label="timestamp")
    logger.info("Saved %d alt rows (%s -> %s) to %s in %.0fs", len(alt), alt.index[0], alt.index[-1], path.name, time.perf_counter() - t0)
    return alt


def load_alt_data(path: Path | None = None) -> pd.DataFrame | None:
    path = path or alt_cache_path()
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df[ALT_COLUMNS].astype(float).sort_index()


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def _z(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0.0, np.nan)
    return (s - m) / sd


def build_flow_features(ohlcv: pd.DataFrame, indicators: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    """Extended features + order-flow / derivatives context. Every value at bar *t* only
    uses alt rows with timestamp <= t (reindex + ffill), so the set stays causal."""
    feats = build_extended_features(ohlcv, indicators)
    a = alt.reindex(ohlcv.index, method="ffill")
    vol = ohlcv["volume"].replace(0.0, np.nan)

    taker_ratio = (a["taker_buy_base"] / vol).clip(0, 1)
    feats["taker_buy_ratio"] = taker_ratio
    feats["taker_buy_ratio_24"] = taker_ratio.rolling(24, min_periods=12).mean()
    feats["taker_buy_ratio_z168"] = _z(taker_ratio.rolling(24, min_periods=12).mean(), 168)
    taker_imbalance = (2 * a["taker_buy_base"] - ohlcv["volume"])  # buy - sell volume
    feats["taker_imbalance_24"] = taker_imbalance.rolling(24, min_periods=12).sum() / vol.rolling(24, min_periods=12).sum()

    trades = a["trades"].replace(0.0, np.nan)
    feats["trades_ratio_168"] = trades / trades.rolling(168, min_periods=48).mean()
    avg_trade = vol / trades
    feats["avg_trade_size_ratio_168"] = avg_trade / avg_trade.rolling(168, min_periods=48).mean()

    feats["funding_rate"] = a["funding_rate"] * 1e3
    feats["funding_ma_3d"] = a["funding_rate"].rolling(72, min_periods=24).mean() * 1e3
    feats["funding_z_30d"] = _z(a["funding_rate"], 720)
    feats["funding_cum_7d"] = a["funding_rate"].rolling(168, min_periods=48).sum() * 1e3

    feats["premium"] = a["premium_close"] * 1e3
    feats["premium_ma_24"] = a["premium_close"].rolling(24, min_periods=12).mean() * 1e3
    feats["premium_z_168"] = _z(a["premium_close"], 168)

    basis = a["perp_close"] / ohlcv["close"] - 1.0
    feats["perp_basis"] = basis * 1e3
    feats["perp_basis_z_168"] = _z(basis, 168)
    perp_vol_ratio = a["perp_volume"] / vol
    feats["perp_spot_volume_ratio_24"] = perp_vol_ratio.rolling(24, min_periods=12).mean()
    feats["perp_spot_volume_ratio_z168"] = _z(perp_vol_ratio.rolling(24, min_periods=12).mean(), 168)
    return feats.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download alternative Binance data")
    parser.add_argument("--years", type=float, default=6.0)
    args = parser.parse_args()
    alt = download_alt_data(args.years)
    print(f"{len(alt)} rows, {alt.index[0]} -> {alt.index[-1]}")
    print(alt.tail(3).to_string())
    print("missing per column:", alt.isna().sum().to_dict())


if __name__ == "__main__":
    main()
