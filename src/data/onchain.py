"""Free on-chain, sentiment and macro daily data (no API keys):

* Blockchain.com charts  - active addresses, transactions, hash rate, fees, mempool, tx volume
* Coin Metrics community - MVRV, realised cap, active addresses (whatever the free tier allows),
                           USDT + USDC market cap (stable-coin liquidity)
* Alternative.me         - Fear & Greed index
* FRED                   - S&P 500, broad dollar index, VIX, 10-year yield

All series are daily. When aligned to the hourly BTC index every value is shifted by one full
day, so the model at the close of day *T* only sees values published for day *T-1* - the
safest assumption about publication lags.

    python -m src.data.onchain --years 6
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import settings
from src.data.processor import build_extended_features
from src.logging_config import get_logger

logger = get_logger(__name__)

UA = {"User-Agent": "btc-ml-trader/1.0 (research)"}
BLOCKCHAIN_CHARTS = {
    "n-unique-addresses": "bc_active_addresses",
    "n-transactions": "bc_tx_count",
    "hash-rate": "bc_hash_rate",
    "transaction-fees-usd": "bc_fees_usd",
    "estimated-transaction-volume-usd": "bc_tx_volume_usd",
    "mempool-size": "bc_mempool_bytes",
}
COINMETRICS_BTC = {"CapMVRVCur": "cm_mvrv", "CapRealUSD": "cm_realized_cap", "AdrActCnt": "cm_active_addr", "TxTfrValAdjUSD": "cm_tx_value_usd", "SplyAct1yr": "cm_supply_active_1y"}
FRED_SERIES = {"SP500": "spx", "DTWEXBGS": "dxy", "VIXCLS": "vix", "DGS10": "us10y"}


def onchain_cache_path() -> Path:
    return settings.data_dir / "BTC_onchain_daily.csv"


def _get(url: str, timeout: int = 40) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r


def fetch_blockchain_com(years: float) -> pd.DataFrame:
    frames = []
    for chart, col in BLOCKCHAIN_CHARTS.items():
        try:
            j = _get(f"https://api.blockchain.info/charts/{chart}?timespan={int(years)}years&format=json&sampled=false").json()
            s = pd.Series({pd.Timestamp(v["x"], unit="s", tz="UTC").floor("D"): float(v["y"]) for v in j["values"]}, name=col)
            frames.append(s.groupby(level=0).mean())
            logger.info("blockchain.com %s: %d points", chart, len(s))
        except Exception as exc:  # noqa: BLE001 - one missing chart must not sink the whole set
            logger.warning("blockchain.com %s failed: %s", chart, exc)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_coinmetrics(years: float) -> pd.DataFrame:
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=int(years * 365))).strftime("%Y-%m-%d")
    frames = []
    for metric, col in COINMETRICS_BTC.items():
        try:
            url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics={metric}&frequency=1d&start_time={start}&page_size=10000"
            data = _get(url).json().get("data", [])
            if not data:
                continue
            s = pd.Series({pd.Timestamp(d["time"]).floor("D"): float(d[metric]) for d in data if d.get(metric) not in (None, "")}, name=col)
            frames.append(s)
            logger.info("coinmetrics %s: %d points", metric, len(s))
        except Exception as exc:  # noqa: BLE001
            logger.warning("coinmetrics %s unavailable on the community tier: %s", metric, exc)
    try:
        url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=usdt,usdc&metrics=CapMrktCurUSD&frequency=1d&start_time={start}&page_size=10000"
        data = _get(url).json().get("data", [])
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.floor("D")
            df["CapMrktCurUSD"] = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
            stable = df.pivot_table(index="time", columns="asset", values="CapMrktCurUSD", aggfunc="last")
            frames.append(stable.sum(axis=1).rename("cm_stablecoin_cap"))
            logger.info("coinmetrics stablecoin cap: %d points", len(stable))
    except Exception as exc:  # noqa: BLE001
        logger.warning("coinmetrics stablecoin cap failed: %s", exc)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_fear_greed() -> pd.DataFrame:
    j = _get("https://api.alternative.me/fng/?limit=0&format=json").json()
    s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC").floor("D"): float(d["value"]) for d in j["data"]}, name="fear_greed")
    logger.info("fear & greed: %d points", len(s))
    return s.sort_index().to_frame()


def fetch_fred() -> pd.DataFrame:
    frames = []
    for series, col in FRED_SERIES.items():
        try:
            txt = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}").text
            df = pd.read_csv(io.StringIO(txt))
            df.columns = ["date", col]
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df.set_index("date")[col])
            logger.info("FRED %s: %d points", series, len(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED %s failed: %s", series, exc)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def download_onchain(years: float = 6.0, path: Path | None = None) -> pd.DataFrame:
    path = path or onchain_cache_path()
    t0 = time.perf_counter()
    parts = [fetch_blockchain_com(years), fetch_coinmetrics(years), fetch_fear_greed(), fetch_fred()]
    daily = pd.concat([p for p in parts if not p.empty], axis=1).sort_index()
    daily = daily[daily.index >= pd.Timestamp.utcnow().floor("D") - pd.Timedelta(days=int(years * 365) + 60)]
    daily.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path)
    logger.info("Saved %d daily rows x %d columns to %s in %.0fs", len(daily), daily.shape[1], path.name, time.perf_counter() - t0)
    return daily


def load_onchain(path: Path | None = None) -> pd.DataFrame | None:
    path = path or onchain_cache_path()
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def _z(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 3)).std().replace(0.0, np.nan)
    return (s - m) / sd


def daily_onchain_features(daily: pd.DataFrame, btc_daily_close: pd.Series | None = None) -> pd.DataFrame:
    """Daily-frequency derived features; every value uses only rows up to the same date."""
    d = daily.ffill()
    f = pd.DataFrame(index=d.index)
    for col in ("bc_active_addresses", "bc_tx_count", "bc_fees_usd", "bc_tx_volume_usd", "bc_mempool_bytes", "cm_active_addr", "cm_tx_value_usd"):
        if col in d:
            f[f"{col}_z30"] = _z(d[col], 30)
            f[f"{col}_chg7"] = d[col].pct_change(7)
    if "bc_hash_rate" in d:
        f["hash_rate_chg30"] = d["bc_hash_rate"].pct_change(30)
        f["hash_rate_z90"] = _z(d["bc_hash_rate"], 90)
    if "cm_mvrv" in d:
        f["mvrv"] = d["cm_mvrv"]
        f["mvrv_z90"] = _z(d["cm_mvrv"], 90)
        f["mvrv_chg30"] = d["cm_mvrv"].pct_change(30)
    if "cm_realized_cap" in d:
        f["realized_cap_chg30"] = d["cm_realized_cap"].pct_change(30)
    if "cm_supply_active_1y" in d:
        f["supply_active_1y_chg30"] = d["cm_supply_active_1y"].pct_change(30)
    if "cm_stablecoin_cap" in d:
        f["stablecoin_chg7"] = d["cm_stablecoin_cap"].pct_change(7)
        f["stablecoin_chg30"] = d["cm_stablecoin_cap"].pct_change(30)
    if "fear_greed" in d:
        f["fear_greed"] = d["fear_greed"] / 100.0
        f["fear_greed_chg7"] = d["fear_greed"].diff(7) / 100.0
        f["fear_greed_z30"] = _z(d["fear_greed"], 30)
    if "spx" in d:
        f["spx_ret1"] = d["spx"].pct_change(1)
        f["spx_ret5"] = d["spx"].pct_change(5)
        f["spx_ret20"] = d["spx"].pct_change(20)
        if btc_daily_close is not None:
            b = btc_daily_close.reindex(d.index).ffill().pct_change()
            f["btc_spx_corr30"] = b.rolling(30, min_periods=15).corr(d["spx"].pct_change())
    if "dxy" in d:
        f["dxy_ret5"] = d["dxy"].pct_change(5)
        f["dxy_ret20"] = d["dxy"].pct_change(20)
    if "vix" in d:
        f["vix"] = d["vix"] / 100.0
        f["vix_chg5"] = d["vix"].pct_change(5)
    if "us10y" in d:
        f["us10y_chg20"] = d["us10y"].diff(20)
    return f.replace([np.inf, -np.inf], np.nan)


def align_daily_to_hourly(daily_feats: pd.DataFrame, hourly_index: pd.DatetimeIndex, lag_days: int = 1) -> pd.DataFrame:
    """Value for date D becomes visible from (D + lag_days) 00:00 UTC onward."""
    shifted = daily_feats.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    return shifted.reindex(hourly_index, method="ffill")


def build_onchain_features(ohlcv: pd.DataFrame, indicators: pd.DataFrame | None, daily: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    feats = build_extended_features(ohlcv, indicators)
    btc_daily_close = ohlcv["close"].resample("1D").last()
    df = daily_onchain_features(daily, btc_daily_close)
    aligned = align_daily_to_hourly(df, ohlcv.index, lag_days)
    for col in aligned.columns:
        feats[f"oc_{col}"] = aligned[col]
    # the newest rows can lack a few slow-updating series; forward-fill within the hourly frame
    oc_cols = [c for c in feats.columns if c.startswith("oc_")]
    feats[oc_cols] = feats[oc_cols].ffill()
    return feats.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free on-chain / sentiment / macro daily data")
    parser.add_argument("--years", type=float, default=6.0)
    args = parser.parse_args()
    daily = download_onchain(args.years)
    print(f"{len(daily)} rows, {daily.index[0].date()} -> {daily.index[-1].date()}")
    print("columns:", list(daily.columns))
    print("missing share per column:", {c: round(float(daily[c].isna().mean()), 3) for c in daily.columns})


if __name__ == "__main__":
    main()
