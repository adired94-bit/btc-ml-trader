"""Risk management: ATR-based stops, reward/risk take-profits and position sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from config import settings

Side = Literal["LONG", "SHORT"]


class RiskError(ValueError):
    """Raised for invalid risk inputs (non-positive prices, ATR, equity...)."""


@dataclass
class TakeProfit:
    reward_risk: float
    price: float
    distance_pct: float


@dataclass
class TradePlan:
    side: Side
    entry_price: float
    stop_loss: float
    stop_distance: float
    stop_distance_pct: float
    take_profits: list[TakeProfit] = field(default_factory=list)
    atr: float = 0.0
    equity: float = 0.0
    risk_amount: float = 0.0
    position_size: float = 0.0  # units of the base asset (BTC)
    notional: float = 0.0  # position value in quote currency (USDT)
    leverage_required: float = 0.0
    capped_by_leverage: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["take_profits"] = [asdict(tp) for tp in self.take_profits]
        return data


class RiskManager:
    """Stateless calculator that turns (side, entry, ATR) into a fully sized trade plan."""

    def __init__(
        self,
        equity: float | None = None,
        risk_per_trade_pct: float | None = None,
        atr_multiplier: float | None = None,
        reward_risk_ratios: list[float] | None = None,
        max_leverage: float | None = None,
    ) -> None:
        self.equity = float(equity if equity is not None else settings.account_equity)
        self.risk_per_trade_pct = float(
            risk_per_trade_pct if risk_per_trade_pct is not None else settings.risk_per_trade_pct
        )
        self.atr_multiplier = float(atr_multiplier if atr_multiplier is not None else settings.atr_stop_multiplier)
        ratios = reward_risk_ratios if reward_risk_ratios is not None else settings.reward_risk_ratios
        self.reward_risk_ratios = sorted(float(r) for r in ratios)
        self.max_leverage = float(max_leverage if max_leverage is not None else settings.max_leverage)
        self._validate()

    def _validate(self) -> None:
        if self.equity <= 0:
            raise RiskError("Equity must be positive")
        if not 0 < self.risk_per_trade_pct <= 0.2:
            raise RiskError("risk_per_trade_pct must be in (0, 0.2]")
        if self.atr_multiplier <= 0:
            raise RiskError("ATR multiplier must be positive")
        if not self.reward_risk_ratios or any(r <= 0 for r in self.reward_risk_ratios):
            raise RiskError("Reward/risk ratios must be positive")
        if self.max_leverage <= 0:
            raise RiskError("max_leverage must be positive")

    # --- Individual components --------------------------------------------

    def stop_loss(self, side: Side, entry_price: float, atr: float) -> float:
        if entry_price <= 0 or atr <= 0:
            raise RiskError("Entry price and ATR must be positive")
        distance = self.atr_multiplier * atr
        stop = entry_price - distance if side == "LONG" else entry_price + distance
        if stop <= 0:
            raise RiskError("Computed stop-loss is not positive; ATR too large relative to price")
        return stop

    def take_profits(self, side: Side, entry_price: float, stop_loss: float) -> list[TakeProfit]:
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            raise RiskError("Stop-loss must differ from entry price")
        out: list[TakeProfit] = []
        for rr in self.reward_risk_ratios:
            distance = rr * risk
            price = entry_price + distance if side == "LONG" else entry_price - distance
            if price <= 0:
                continue
            out.append(TakeProfit(reward_risk=rr, price=price, distance_pct=distance / entry_price))
        if not out:
            raise RiskError("No valid take-profit level could be computed")
        return out

    def risk_amount(self) -> float:
        return self.equity * self.risk_per_trade_pct

    def position_size(self, entry_price: float, stop_loss: float) -> tuple[float, bool]:
        """Units to trade so that hitting the stop loses exactly ``risk_amount``.

        The size is capped so the notional never exceeds ``max_leverage * equity``.
        Returns ``(units, was_capped)``.
        """
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0 or entry_price <= 0:
            raise RiskError("Invalid entry / stop for sizing")
        units = self.risk_amount() / risk_per_unit
        max_units = self.max_leverage * self.equity / entry_price
        if units > max_units:
            return max_units, True
        return units, False

    # --- Full plan ----------------------------------------------------------

    def build_plan(self, side: Side, entry_price: float, atr: float) -> TradePlan:
        if side not in ("LONG", "SHORT"):
            raise RiskError(f"Unsupported side: {side}")
        stop = self.stop_loss(side, entry_price, atr)
        tps = self.take_profits(side, entry_price, stop)
        units, capped = self.position_size(entry_price, stop)
        notional = units * entry_price
        stop_distance = abs(entry_price - stop)
        return TradePlan(
            side=side,
            entry_price=float(entry_price),
            stop_loss=float(stop),
            stop_distance=float(stop_distance),
            stop_distance_pct=float(stop_distance / entry_price),
            take_profits=tps,
            atr=float(atr),
            equity=self.equity,
            risk_amount=float(units * stop_distance),
            position_size=float(units),
            notional=float(notional),
            leverage_required=float(notional / self.equity),
            capped_by_leverage=capped,
        )

    # --- Portfolio-level helpers -------------------------------------------

    @staticmethod
    def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion optimal fraction (clipped to [0, 1]); 0 when the edge is negative."""
        if avg_loss <= 0 or not 0 <= win_rate <= 1:
            return 0.0
        payoff = avg_win / avg_loss
        if payoff <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / payoff
        return float(min(max(kelly, 0.0), 1.0))

    @staticmethod
    def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Expected return per unit risked."""
        return float(win_rate * avg_win - (1 - win_rate) * avg_loss)
