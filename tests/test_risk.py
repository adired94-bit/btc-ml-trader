"""Risk management tests: ATR stops, reward/risk take-profits and position sizing."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.management import RiskError, RiskManager


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(equity=10_000, risk_per_trade_pct=0.01, atr_multiplier=1.5, reward_risk_ratios=[2.0, 3.0], max_leverage=3.0)


def test_long_stop_below_entry_and_short_above(rm):
    assert rm.stop_loss("LONG", 50_000, 400) == 50_000 - 600
    assert rm.stop_loss("SHORT", 50_000, 400) == 50_000 + 600


def test_take_profits_respect_reward_risk_ratios(rm):
    entry, stop = 50_000.0, 49_400.0
    tps = rm.take_profits("LONG", entry, stop)
    assert [tp.reward_risk for tp in tps] == [2.0, 3.0]
    assert np.isclose(tps[0].price, entry + 2 * 600)
    assert np.isclose(tps[1].price, entry + 3 * 600)
    short_tps = rm.take_profits("SHORT", entry, entry + 600)
    assert np.isclose(short_tps[0].price, entry - 1_200)


def test_minimum_reward_risk_is_two():
    manager = RiskManager()
    assert min(manager.reward_risk_ratios) >= 2.0


def test_position_size_risks_exactly_the_budget(rm):
    plan = rm.build_plan("LONG", 50_000, 400)
    assert np.isclose(plan.risk_amount, 100.0)  # 1% of 10 000
    assert np.isclose(plan.position_size * plan.stop_distance, 100.0)
    assert plan.notional == pytest.approx(plan.position_size * 50_000)
    assert not plan.capped_by_leverage


def test_leverage_cap_limits_size(rm):
    # Tiny ATR -> huge theoretical size -> capped at 3x equity notional.
    plan = rm.build_plan("SHORT", 50_000, 1.0)
    assert plan.capped_by_leverage
    assert plan.notional == pytest.approx(3.0 * 10_000)
    assert plan.risk_amount < 100.0


def test_plan_serialisation_is_json_friendly(rm):
    data = rm.build_plan("LONG", 50_000, 400).to_dict()
    assert data["side"] == "LONG"
    assert isinstance(data["take_profits"], list) and data["take_profits"][0]["reward_risk"] == 2.0


@pytest.mark.parametrize("kwargs", [
    {"equity": 0}, {"risk_per_trade_pct": 0}, {"risk_per_trade_pct": 0.5}, {"atr_multiplier": -1},
    {"reward_risk_ratios": []}, {"reward_risk_ratios": [-2]}, {"max_leverage": 0},
])
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(RiskError):
        RiskManager(**kwargs)


def test_invalid_trade_inputs_raise(rm):
    with pytest.raises(RiskError):
        rm.stop_loss("LONG", 0, 100)
    with pytest.raises(RiskError):
        rm.stop_loss("LONG", 100, 0)
    with pytest.raises(RiskError):
        rm.build_plan("SIDEWAYS", 100, 1)  # type: ignore[arg-type]
    with pytest.raises(RiskError):
        rm.stop_loss("LONG", 100, 100)  # stop would be <= 0


def test_kelly_and_expectancy():
    assert RiskManager.kelly_fraction(0.5, 2.0, 1.0) == pytest.approx(0.25)
    assert RiskManager.kelly_fraction(0.3, 1.0, 1.0) == 0.0
    assert RiskManager.kelly_fraction(0.5, 1.0, 0.0) == 0.0
    assert RiskManager.expectancy(0.5, 2.0, 1.0) == pytest.approx(0.5)
