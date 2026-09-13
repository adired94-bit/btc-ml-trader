"""On-chain / macro feature tests (synthetic daily data, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import onchain as O


@pytest.fixture(scope="module")
def daily(ohlcv) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    days = pd.date_range(ohlcv.index[0].floor("D") - pd.Timedelta(days=120), ohlcv.index[-1].floor("D"), freq="D", tz="UTC")
    n = len(days)
    df = pd.DataFrame(index=days)
    df["bc_active_addresses"] = 700_000 + rng.normal(0, 30_000, n).cumsum() / 10
    df["bc_tx_count"] = 300_000 + rng.normal(0, 5_000, n)
    df["bc_hash_rate"] = np.linspace(400, 600, n) + rng.normal(0, 5, n)
    df["bc_fees_usd"] = rng.lognormal(14, 0.3, n)
    df["cm_mvrv"] = 1.5 + rng.normal(0, 0.05, n).cumsum() / 5
    df["cm_stablecoin_cap"] = 1.2e11 * (1 + np.linspace(0, 0.3, n))
    df["fear_greed"] = np.clip(50 + rng.normal(0, 15, n), 0, 100)
    df["spx"] = 4500 * np.exp(rng.normal(0.0002, 0.01, n).cumsum())
    df["dxy"] = 100 + rng.normal(0, 0.3, n).cumsum()
    df["vix"] = np.clip(18 + rng.normal(0, 2, n), 9, 80)
    df["us10y"] = 4 + rng.normal(0, 0.03, n).cumsum()
    df.loc[df.index[::7], "spx"] = np.nan  # weekends / holidays in macro data
    df.index.name = "date"
    return df


def test_daily_features_and_alignment(ohlcv, daily):
    f = O.daily_onchain_features(daily, ohlcv["close"].resample("1D").last())
    for col in ("bc_active_addresses_z30", "mvrv_z90", "stablecoin_chg7", "fear_greed", "spx_ret5", "btc_spx_corr30", "dxy_ret20", "vix"):
        assert col in f.columns
    feats = O.build_onchain_features(ohlcv, None, daily)
    assert len(feats) == len(ohlcv)
    oc = [c for c in feats.columns if c.startswith("oc_")]
    assert len(oc) >= 15
    assert feats.iloc[-1][oc].notna().all()


def test_one_day_publication_lag(ohlcv, daily):
    f = O.daily_onchain_features(daily)
    aligned = O.align_daily_to_hourly(f[["fear_greed"]], ohlcv.index, lag_days=1)
    some_day = ohlcv.index[500].floor("D")
    # During day D the hourly frame must carry the value published for D-1, not D.
    expected = f.loc[some_day - pd.Timedelta(days=1), "fear_greed"]
    assert aligned.loc[some_day + pd.Timedelta(hours=5), "fear_greed"] == pytest.approx(expected)
    assert aligned.loc[some_day + pd.Timedelta(hours=23), "fear_greed"] == pytest.approx(expected)


def test_features_are_causal(ohlcv, daily):
    full = O.build_onchain_features(ohlcv, None, daily)
    cut_ohlcv = ohlcv.iloc[:-120]
    cut_daily = daily[daily.index <= cut_ohlcv.index[-1].floor("D")]
    short = O.build_onchain_features(cut_ohlcv, None, cut_daily)
    common = short.index[-150:]
    pd.testing.assert_frame_equal(full.loc[common], short.loc[common], check_exact=False, rtol=1e-9, atol=1e-12)


def test_cache_roundtrip(tmp_path, daily):
    path = tmp_path / "onchain.csv"
    daily.to_csv(path)
    loaded = O.load_onchain(path)
    assert loaded is not None and list(loaded.columns) == list(daily.columns)
    assert O.load_onchain(tmp_path / "missing.csv") is None
