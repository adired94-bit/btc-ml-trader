"""Order-flow / derivatives feature tests (synthetic alt data, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import altdata as A
from src.data.processor import add_all_indicators


@pytest.fixture(scope="module")
def alt(ohlcv) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = len(ohlcv)
    df = pd.DataFrame(index=ohlcv.index)
    df["quote_volume"] = ohlcv["volume"] * ohlcv["close"]
    df["trades"] = rng.integers(500, 5000, n).astype(float)
    df["taker_buy_base"] = ohlcv["volume"] * rng.uniform(0.3, 0.7, n)
    df["perp_close"] = ohlcv["close"] * (1 + rng.normal(0, 0.0005, n))
    df["perp_volume"] = ohlcv["volume"] * rng.uniform(2, 6, n)
    df["premium_close"] = rng.normal(0, 0.0002, n)
    funding = pd.Series(np.nan, index=ohlcv.index)
    funding[ohlcv.index.hour % 8 == 0] = rng.normal(0.0001, 0.0002, (ohlcv.index.hour % 8 == 0).sum())
    df["funding_rate"] = funding.ffill()
    return df[A.ALT_COLUMNS]


def test_flow_features_shape_and_ranges(ohlcv, alt):
    ind = add_all_indicators(ohlcv)
    feats = A.build_flow_features(ohlcv, ind, alt)
    assert len(feats) == len(ohlcv)
    for col in ("taker_buy_ratio", "funding_rate", "premium", "perp_basis", "trades_ratio_168", "perp_spot_volume_ratio_24"):
        assert col in feats.columns
    assert feats["taker_buy_ratio"].dropna().between(0, 1).all()
    assert feats.iloc[-1].notna().all()


def test_flow_features_are_causal(ohlcv, alt):
    ind = add_all_indicators(ohlcv)
    full = A.build_flow_features(ohlcv, ind, alt)
    cut = ohlcv.iloc[:-100]
    short = A.build_flow_features(cut, add_all_indicators(cut), alt.iloc[:-100])
    common = short.index[-200:]
    pd.testing.assert_frame_equal(full.loc[common], short.loc[common], check_exact=False, rtol=1e-9, atol=1e-12)


def test_flow_features_survive_alt_gaps(ohlcv, alt):
    ind = add_all_indicators(ohlcv)
    gappy = alt.drop(alt.index[500:520])  # missing hours are forward-filled, not dropped
    feats = A.build_flow_features(ohlcv, ind, gappy)
    assert len(feats) == len(ohlcv)
    assert feats["funding_rate"].iloc[510] == pytest.approx(feats["funding_rate"].iloc[499])


def test_alt_cache_roundtrip(tmp_path, alt):
    path = tmp_path / "alt.csv"
    alt.to_csv(path, index_label="timestamp")
    loaded = A.load_alt_data(path)
    assert loaded is not None and list(loaded.columns) == A.ALT_COLUMNS
    assert loaded.index.equals(alt.index)  # names may differ after CSV roundtrip
    assert A.load_alt_data(tmp_path / "missing.csv") is None
