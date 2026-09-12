"""Backtest engine tests: trade mechanics, metrics and the walk-forward loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import engine as E
from src.risk.management import RiskManager
from tests.conftest import FAST_LGBM, FAST_XGB


def _path(prices: list[float], start="2024-01-01") -> pd.DataFrame:
    """Build an OHLCV frame where each bar opens/closes at the given price with a ±0.5% wick."""
    idx = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    p = np.array(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.005, "low": p * 0.995, "close": p, "volume": 10.0}, index=idx)


def _probs(idx, longs: dict[int, float] | None = None, shorts: dict[int, float] | None = None) -> pd.DataFrame:
    probs = pd.DataFrame({"p_down": 0.2, "p_flat": 0.6, "p_up": 0.2}, index=idx)
    for i, p in (longs or {}).items():
        probs.iloc[i] = [1 - p - 0.05, 0.05, p]
    for i, p in (shorts or {}).items():
        probs.iloc[i] = [p, 0.05, 1 - p - 0.05]
    return probs


RISK = RiskManager(equity=10_000, risk_per_trade_pct=0.01, atr_multiplier=1.0, reward_risk_ratios=[2.0], max_leverage=100)


def test_long_take_profit_hit():
    # Signal at bar 0 -> enter at bar 1 open (100). ATR=1 -> stop 99, TP 102. Price reaches 102 at bar 3.
    df = _path([100, 100, 101, 102.5, 103])
    atr = pd.Series(1.0, index=df.index)
    trades, curve = E.simulate_trades(df, _probs(df.index, longs={0: 0.8}), atr, RISK, 0.55, 2.0, 10, 0.0, 0.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "LONG" and t.exit_reason == "TAKE_PROFIT"
    assert t.entry_price == 100 and t.take_profit == pytest.approx(102) and t.stop_loss == pytest.approx(99)
    assert t.pnl == pytest.approx(200.0)  # risk 100 * R:R 2
    assert curve.iloc[-1] == pytest.approx(10_200.0)


def test_long_stop_hit_and_fees():
    df = _path([100, 100, 98, 97])
    atr = pd.Series(1.0, index=df.index)
    trades, curve = E.simulate_trades(df, _probs(df.index, longs={0: 0.8}), atr, RISK, 0.55, 2.0, 10, 0.001, 0.0)
    assert len(trades) == 1 and trades[0].exit_reason == "STOP"
    expected_fees = 0.001 * trades[0].size * (100 + 99)
    assert trades[0].pnl == pytest.approx(-100.0 - expected_fees)
    assert curve.iloc[-1] == pytest.approx(10_000 - 100 - expected_fees)


def test_short_trade_mechanics_and_slippage():
    df = _path([100, 100, 99, 97.5, 97])
    atr = pd.Series(1.0, index=df.index)
    trades, _ = E.simulate_trades(df, _probs(df.index, shorts={0: 0.9}), atr, RISK, 0.55, 2.0, 10, 0.0, 0.001)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "SHORT" and t.entry_price == pytest.approx(100 * (1 - 0.001))
    assert t.stop_loss > t.entry_price > t.take_profit
    assert t.exit_reason == "TAKE_PROFIT" and t.pnl > 0


def test_stop_takes_priority_when_both_touched_in_same_bar():
    df = _path([100, 100, 100])
    df.loc[df.index[1], ["high", "low"]] = [105, 95]  # wide bar touches both TP and SL
    atr = pd.Series(1.0, index=df.index)
    trades, _ = E.simulate_trades(df, _probs(df.index, longs={0: 0.8}), atr, RISK, 0.55, 2.0, 10, 0.0, 0.0)
    assert trades[0].exit_reason == "STOP"


def test_time_exit_after_max_holding():
    df = _path([100] * 8)
    atr = pd.Series(1.0, index=df.index)
    trades, _ = E.simulate_trades(df, _probs(df.index, longs={0: 0.8}), atr, RISK, 0.55, 2.0, 3, 0.0, 0.0)
    assert len(trades) == 1 and trades[0].exit_reason == "TIME" and trades[0].bars_held == 3


def test_no_signal_below_threshold_and_no_overlapping_positions():
    df = _path([100] * 10)
    atr = pd.Series(1.0, index=df.index)
    probs = _probs(df.index, longs={0: 0.5, 1: 0.9, 2: 0.9, 3: 0.9})
    trades, curve = E.simulate_trades(df, probs, atr, RISK, 0.55, 2.0, 4, 0.0, 0.0)
    assert len(trades) == 1  # bar 0 is below threshold; bars 2-3 fall inside the open trade
    assert trades[0].entry_time == df.index[2].isoformat()
    assert len(curve) == len(df)


def test_metrics_computation():
    df = _path([100, 100, 101, 102.5, 103, 103, 101, 100, 99])
    atr = pd.Series(1.0, index=df.index)
    probs = _probs(df.index, longs={0: 0.8, 4: 0.8})
    trades, curve = E.simulate_trades(df, probs, atr, RISK, 0.55, 2.0, 10, 0.0, 0.0)
    m = E.compute_metrics(trades, curve, df, "1h", 10_000)
    assert m.total_trades == 2 and m.win_rate == 0.5
    assert m.exit_reasons == {"TAKE_PROFIT": 1, "STOP": 1}
    # First trade wins 200 (1% of 10 000 x R:R 2); second risks 1% of the compounded 10 200 = 102.
    assert m.profit_factor == pytest.approx(200 / 102)
    assert m.final_equity == pytest.approx(10_098)
    assert m.total_return_pct == pytest.approx(0.98)
    assert m.max_drawdown_pct == pytest.approx(102 / 10_200 * 100)
    assert m.buy_and_hold_return_pct == pytest.approx(-1.0)


def test_metrics_with_no_trades():
    df = _path([100] * 5)
    curve = pd.Series(10_000.0, index=df.index)
    m = E.compute_metrics([], curve, df, "1h", 10_000)
    assert m.total_trades == 0 and m.win_rate == 0 and m.sharpe_ratio == 0 and m.max_drawdown_pct == 0


def test_walk_forward_only_predicts_unseen_bars(dataset):
    X, y, _ = dataset
    probs, folds = E.walk_forward_probabilities(X, y, train_window=600, test_window=300, xgb_params=FAST_XGB, lgbm_params=FAST_LGBM)
    assert len(folds) >= 2
    assert not probs.index.duplicated().any()
    for fold in folds:
        assert pd.Timestamp(fold["test_start"]) > pd.Timestamp(fold["train_end"])
    # Every predicted bar lies strictly after its fold's training window.
    assert probs.index.min() > pd.Timestamp(folds[0]["train_end"])


def test_run_backtest_holdout_and_walk_forward(ohlcv):
    hold = E.run_backtest(ohlcv, mode="holdout", threshold=0.4, xgb_params=FAST_XGB, lgbm_params=FAST_LGBM)
    assert hold.bars > 0 and hold.metrics.initial_equity > 0
    payload = hold.to_dict(max_curve_points=100)
    assert len(payload["equity_curve"]) <= 101 and "metrics" in payload
    wf = E.run_backtest(ohlcv, mode="walk_forward", threshold=0.4, train_window=700, test_window=300, xgb_params=FAST_XGB, lgbm_params=FAST_LGBM)
    assert len(wf.folds) >= 2
    assert E.format_report(wf).startswith("=== Backtest (walk_forward)")
    with pytest.raises(ValueError):
        E.run_backtest(ohlcv, mode="bogus")
