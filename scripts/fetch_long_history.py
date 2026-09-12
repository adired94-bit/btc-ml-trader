"""Download a long hourly history (default 6 years) into a separate research cache.

The production cache (``data/BTC_USDT_1h.csv``, ~2 years) stays small so the API
remains fast; research scripts read ``data/BTC_USDT_1h_long.csv`` instead.

    venv\\Scripts\\python.exe scripts\\fetch_long_history.py --years 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.data import fetcher, storage  # noqa: E402

LONG_CACHE = settings.data_dir / f"{settings.symbol_slug}_{settings.timeframe}_long.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=6.0)
    args = parser.parse_args()
    candles = int(args.years * 365 * 24)
    df = fetcher.fetch_ohlcv_history(limit=candles)
    storage.save_cache(df, LONG_CACHE)
    print(f"{len(df)} candles {df.index[0]} -> {df.index[-1]} saved to {LONG_CACHE}")


if __name__ == "__main__":
    main()
