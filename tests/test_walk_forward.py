"""Tests for the day-by-day walk-forward evaluation engine and its feedback loop."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import walk_forward as WF
from tests.conftest import make_ohlcv

TINY = {
    "xgb_params": {"n_estimators": 25, "max_depth": 3},
    "lgbm_params": {"n_estimators": 25, "num_leaves": 7},
    "reg_params": {**WF.WF_REG_PARAMS, "n_estimators": 25},
}


@pytest.fixture(scope="module")
def long_ohlcv() -> pd.DataFrame:
    # ~200 days of hourly data: enough for a 60-day evaluation window after feature warm-up.
    return make_ohlcv(n=24 * 200, seed=3)


@pytest.fixture(scope="module")
def cfg() -> WF.WalkForwardConfig:
    return WF.WalkForwardConfig(eval_start_days_ago=70, eval_end_days_ago=10, retrain_every_days=20, min_train_rows=500, threshold=0.5, label="test", **TINY)


@pytest.fixture(scope="module")
def result(long_ohlcv, cfg) -> WF.WalkForwardResult:
    return WF.run_walk_forward(long_ohlcv, cfg, verbose=False)


def test_prepare_data_is_causal_and_aligned(long_ohlcv):
    data = WF.prepare_data(long_ohlcv)
    assert data.features.index.equals(long_ohlcv.index)
    assert data.direction.iloc[-24:].eq(-1).all()
    assert set(data.direction.unique()) <= {-1, 0, 2}
    assert set(WF.REGIME_FLAGS) == set(data.regimes.columns)
    assert data.daily_atr.dropna().gt(0).all()
    # Regime flags must not change when the future is removed.
    shorter = WF.compute_regimes(WF.add_all_indicators(long_ohlcv.iloc[:-200]))
    pd.testing.assert_frame_equal(shorter.iloc[-300:], data.regimes.loc[shorter.index[-300:]])


def test_walk_forward_records_have_no_look_ahead(result, cfg):
    assert len(result.records) >= 50
    for r in result.records:
        day = pd.Timestamp(r.day)
        assert pd.Timestamp(r.train_end) <= day + pd.Timedelta(hours=23) - pd.Timedelta(hours=cfg.horizon_bars)
        assert pd.Timestamp(r.target_day) == day + pd.Timedelta(days=1)
        assert r.predicted_direction in {"UP", "DOWN"} and r.actual_direction in {"UP", "DOWN"}
        assert r.hit == (r.predicted_direction == r.actual_direction)
        assert r.abs_error_pct == pytest.approx(abs(r.error_pct))
        assert r.risk_level in {"LOW", "MEDIUM", "HIGH"}
        assert (r.trade_side == "NONE") == (not r.traded)
    days = [r.day for r in result.records]
    assert days == sorted(days) and len(set(days)) == len(days)
    assert result.retrains >= 3


def test_metrics_are_consistent(result, cfg):
    m = result.metrics
    hits = sum(r.hit for r in result.records)
    assert m["directional_accuracy"] == pytest.approx(hits / len(result.records))
    assert m["traded_days"] == sum(r.traded for r in result.records)
    assert 0 <= m["coverage"] <= 1
    assert m["mae_pct"] >= 0 and m["rmse_pct"] >= m["mae_pct"] - 1e-9
    assert m["rmse_usd"] >= m["mae_usd"] - 1e-9
    assert m["final_equity"] == pytest.approx(cfg.initial_equity + m["total_pnl"])
    assert m["max_drawdown_pct"] >= 0
    for key in ("sharpe_ratio", "win_rate", "profit_factor", "buy_and_hold_return_pct", "naive_mae_pct"):
        assert key in m


def test_failure_analysis_structure(result):
    a = result.failure_analysis
    assert a["hits"] + a["misses"] == len(result.records)
    assert a["overall_miss_rate"] == pytest.approx(1 - result.metrics["directional_accuracy"])
    names = {p["pattern"] for p in a["patterns"]}
    assert names & set(WF.REGIME_FLAGS)
    for p in a["patterns"]:
        assert 0 <= p["miss_rate"] <= 1 and p["days"] >= 1
    assert isinstance(a["worst_patterns"], list)


def test_simulate_day_trade_mechanics(cfg):
    idx = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1.0}, index=idx)
    bars.loc[idx[5], "high"] = 110.0  # TP (100 + 2 * 1.5 * 2 = 106) hit on bar 5
    flat_cfg = cfg.copy(fee_pct=0.0, slippage_pct=0.0)
    pnl, ret, reason, entry, stop, tp = WF.simulate_day_trade("LONG", bars, atr=2.0, equity=10_000, cfg=flat_cfg)
    assert reason == "TAKE_PROFIT" and entry == 100 and stop == pytest.approx(97) and tp == pytest.approx(106)
    assert pnl == pytest.approx(200) and ret == pytest.approx(2.0)
    bars.loc[idx[5], "high"] = 100.2
    bars.loc[idx[3], "low"] = 90.0
    pnl, _, reason, *_ = WF.simulate_day_trade("LONG", bars, atr=2.0, equity=10_000, cfg=flat_cfg)
    assert reason == "STOP" and pnl == pytest.approx(-100)
    pnl, _, reason, *_ = WF.simulate_day_trade("SHORT", bars.assign(low=99.8, high=100.2), atr=2.0, equity=10_000, cfg=flat_cfg)
    assert reason == "CLOSE" and pnl == pytest.approx(0.0)


def test_skip_regimes_and_threshold_reduce_trades(long_ohlcv, cfg):
    data = WF.prepare_data(long_ohlcv)
    base = WF.run_walk_forward(long_ohlcv, cfg, data, verbose=False)
    strict = WF.run_walk_forward(long_ohlcv, cfg.copy(threshold=0.99), data, verbose=False)
    assert strict.metrics["traded_days"] == 0 and strict.metrics["total_pnl"] == 0
    skipping = WF.run_walk_forward(long_ohlcv, cfg.copy(skip_regimes=list(WF.REGIME_FLAGS)), data, verbose=False)
    assert skipping.metrics["traded_days"] <= base.metrics["traded_days"]
    for r in skipping.records:
        if r.traded:
            assert not any(f in WF.REGIME_FLAGS for f in r.regimes)


def test_self_improve_returns_validated_choice(long_ohlcv, cfg):
    data = WF.prepare_data(long_ohlcv)
    candidates = [cfg.copy(label="cand_threshold", threshold=0.6), cfg.copy(label="cand_weights", regime_weights={"high_volatility": 1.0})]
    fb = WF.self_improve(long_ohlcv, cfg, data, candidates=candidates, verbose=False)
    assert len(fb.candidates) == 3 and sum(row["chosen"] for row in fb.candidates) == 1
    assert fb.chosen.label in {"test", "cand_threshold", "cand_weights"}
    if not fb.improved:
        assert fb.chosen.label == "test"
    for key in ("traded_directional_accuracy", "total_return_pct", "mae_pct"):
        assert key in fb.chosen_validation and key in fb.baseline_validation


def test_build_candidates_uses_failure_patterns(cfg):
    analysis = {"patterns": [{"pattern": "high_volatility", "days": 30, "lift_vs_overall": 0.1, "ex_ante": True},
                             {"pattern": "trend_reversal", "days": 30, "lift_vs_overall": 0.2, "ex_ante": False}]}
    cands = WF.build_candidates(cfg, analysis)
    labels = [c.label for c in cands]
    assert "regularised" in labels and "higher_threshold" in labels
    assert any(l.startswith("upweight_high_volatility") for l in labels)
    assert any(c.skip_regimes == ["high_volatility"] for c in cands)
    assert all(c.eval_start_days_ago == cfg.eval_start_days_ago for c in cands)


def test_report_and_learnings_entry(tmp_path, result):
    text = WF.format_report(result)
    assert text.startswith("=== Walk-forward daily evaluation [test]")
    entry = WF.learnings_entry(result, None, None)
    path = WF.append_learnings(entry, tmp_path / "SYSTEM_LEARNINGS.md")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# SYSTEM_LEARNINGS") and "Directional accuracy" in content
    payload = result.to_dict()
    assert payload["metrics"]["days"] == len(payload["records"])
    assert not result.to_frame().empty


def test_invalid_window_raises(long_ohlcv, cfg):
    with pytest.raises(ValueError):
        WF.run_walk_forward(long_ohlcv, cfg.copy(eval_start_days_ago=10, eval_end_days_ago=20), verbose=False)
