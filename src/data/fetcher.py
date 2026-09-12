"""Historical + live OHLCV acquisition through CCXT with retry and exchange fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import ccxt
import pandas as pd

from config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Some exchanges quote BTC against USD instead of USDT.
_SYMBOL_ALIASES: dict[str, dict[str, str]] = {
    "kraken": {"BTC/USDT": "BTC/USD", "ETH/USDT": "ETH/USD"},
}


class MarketDataError(RuntimeError):
    """Raised when no exchange could deliver the requested market data."""


@dataclass
class ExchangeHandle:
    exchange_id: str
    client: Any
    symbol: str


def timeframe_to_ms(timeframe: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return int(timeframe[:-1]) * units[timeframe[-1]]


def _build_client(exchange_id: str) -> Any:
    if not hasattr(ccxt, exchange_id):
        raise MarketDataError(f"Unknown CCXT exchange id: {exchange_id}")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls(
        {
            "enableRateLimit": True,
            "timeout": settings.request_timeout_ms,
            "options": {"defaultType": "spot"},
        }
    )


def _resolve_symbol(exchange_id: str, symbol: str) -> str:
    return _SYMBOL_ALIASES.get(exchange_id, {}).get(symbol, symbol)


def connect(symbol: str | None = None) -> ExchangeHandle:
    """Return the first exchange (primary, then fallbacks) that answers."""
    symbol = symbol or settings.symbol
    candidates = [settings.exchange_id, *settings.fallback_exchanges]
    last_error: Exception | None = None
    for exchange_id in candidates:
        try:
            client = _build_client(exchange_id)
            client.load_markets()
            resolved = _resolve_symbol(exchange_id, symbol)
            if resolved not in client.markets:
                raise MarketDataError(f"{exchange_id} does not list {resolved}")
            logger.info("Connected to %s (%s)", exchange_id, resolved)
            return ExchangeHandle(exchange_id, client, resolved)
        except Exception as exc:  # noqa: BLE001 - intentionally try the next exchange
            last_error = exc
            logger.warning("Exchange %s unavailable: %s", exchange_id, exc)
    raise MarketDataError(f"All exchanges failed. Last error: {last_error}")


def _with_retries(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    delay = 1.0
    for attempt in range(1, settings.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            if attempt == settings.max_retries:
                raise
            logger.warning(
                "Transient error (%s), retry %d/%d in %.1fs", exc, attempt, settings.max_retries, delay
            )
            time.sleep(delay)
            delay *= 2
    raise MarketDataError("Retry loop exhausted")  # pragma: no cover - unreachable


def to_dataframe(raw: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df.astype(float)


def fetch_ohlcv_history(
    limit: int | None = None,
    timeframe: str | None = None,
    since_ms: int | None = None,
    handle: ExchangeHandle | None = None,
) -> pd.DataFrame:
    """Download up to ``limit`` closed candles, paginating forward from ``since_ms``."""
    timeframe = timeframe or settings.timeframe
    limit = limit or settings.history_candles
    handle = handle or connect()
    tf_ms = timeframe_to_ms(timeframe)
    page_size = 1000 if handle.exchange_id != "kraken" else 720

    now_ms = handle.client.milliseconds()
    if since_ms is None:
        since_ms = now_ms - (limit + 1) * tf_ms

    chunks: list[list[list[float]]] = []
    cursor = since_ms
    fetched = 0
    while cursor < now_ms and fetched < limit + 1:
        batch = _with_retries(
            handle.client.fetch_ohlcv, handle.symbol, timeframe, since=cursor, limit=page_size
        )
        if not batch:
            break
        chunks.append(batch)
        fetched += len(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:  # exchange returned nothing new; avoid an infinite loop
            break
        cursor = last_ts + tf_ms
        logger.debug("Fetched %d candles (total %d)", len(batch), fetched)
        if len(batch) < page_size:
            break

    if not chunks:
        raise MarketDataError(f"{handle.exchange_id} returned no OHLCV data for {handle.symbol}")

    df = to_dataframe([row for chunk in chunks for row in chunk])
    # Drop the still-forming candle so that features are computed on closed bars only.
    last_open_ms = int(df.index[-1].timestamp() * 1000)
    if last_open_ms + tf_ms > now_ms:
        df = df.iloc[:-1]
    if df.empty:
        raise MarketDataError("Only the forming candle was returned; nothing to store")
    logger.info(
        "Downloaded %d %s candles from %s (%s -> %s)",
        len(df), timeframe, handle.exchange_id, df.index[0], df.index[-1],
    )
    return df.tail(limit)


def fetch_latest_candles(
    n: int = 200, timeframe: str | None = None, handle: ExchangeHandle | None = None
) -> pd.DataFrame:
    """Fetch the most recent ``n`` candles including the currently forming one."""
    timeframe = timeframe or settings.timeframe
    handle = handle or connect()
    raw = _with_retries(handle.client.fetch_ohlcv, handle.symbol, timeframe, limit=n)
    if not raw:
        raise MarketDataError("Empty live candle response")
    return to_dataframe(raw)


def fetch_ticker(handle: ExchangeHandle | None = None) -> dict[str, Any]:
    """Return a compact live ticker snapshot."""
    handle = handle or connect()
    ticker = _with_retries(handle.client.fetch_ticker, handle.symbol)
    ts_ms = ticker.get("timestamp") or handle.client.milliseconds()
    return {
        "exchange": handle.exchange_id,
        "symbol": handle.symbol,
        "last": float(ticker.get("last") or ticker.get("close") or 0.0),
        "bid": float(ticker.get("bid") or 0.0),
        "ask": float(ticker.get("ask") or 0.0),
        "high_24h": float(ticker.get("high") or 0.0),
        "low_24h": float(ticker.get("low") or 0.0),
        "volume_24h": float(ticker.get("baseVolume") or 0.0),
        "change_24h_pct": float(ticker.get("percentage") or 0.0),
        "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC").isoformat(),
    }
