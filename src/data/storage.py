"""CSV-backed OHLCV cache with incremental refresh."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings
from src.data import fetcher
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_cached(path: Path | None = None) -> pd.DataFrame | None:
    path = path or settings.ohlcv_cache_path
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[fetcher.OHLCV_COLUMNS[1:]].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info("Loaded %d cached candles from %s", len(df), path.name)
        return df
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must never break the app
        logger.warning("Cache unreadable (%s); it will be rebuilt", exc)
        return None


def save_cache(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or settings.ohlcv_cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp")
    logger.info("Saved %d candles to %s", len(df), path.name)
    return path


def get_ohlcv(force_refresh: bool = False, min_candles: int | None = None) -> pd.DataFrame:
    """Return an up-to-date OHLCV frame, downloading only the missing tail."""
    min_candles = min_candles or settings.history_candles
    cached = None if force_refresh else load_cached()

    if cached is None or len(cached) < min(min_candles, 500):
        df = fetcher.fetch_ohlcv_history(limit=min_candles)
        save_cache(df)
        return df

    tf_ms = fetcher.timeframe_to_ms(settings.timeframe)
    last_ts_ms = int(cached.index[-1].timestamp() * 1000)
    try:
        handle = fetcher.connect()
        now_ms = handle.client.milliseconds()
        missing = (now_ms - last_ts_ms) // tf_ms
        if missing >= 1:
            fresh = fetcher.fetch_ohlcv_history(
                limit=int(missing) + 2, since_ms=last_ts_ms, handle=handle
            )
            df = pd.concat([cached, fresh])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = cached
    except fetcher.MarketDataError as exc:
        logger.error("Incremental refresh failed, serving cached data: %s", exc)
        return cached

    df = df.tail(max(min_candles, len(cached)))
    save_cache(df)
    return df
