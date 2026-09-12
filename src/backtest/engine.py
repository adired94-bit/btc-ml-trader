"""Backtesting engine.

Two evaluation modes share one trade simulator:

* ``walk_forward`` - the ensemble is re-trained on a rolling window and only
  ever predicts bars it has never seen (the honest, production-like estimate).
* ``holdout`` - fast: fit once on the first ``train_fraction`` of the data and
  simulate on the remainder. Used by the API for quick refreshes.

Trade simulation rules (all deliberately conservative):

* A signal computed on the close of bar *t* is executed at the **open of bar
  t+1** (no look-ahead), with slippage applied against the trader.
* Stop-loss and take-profit are evaluated on every subsequent bar's high/low.
  If both are touched in the same bar the stop is assumed to fill first.
* Positions are force-closed at the close after ``max_holding_bars``.
* Fees are charged on both entry and exit notional. Only one position at a time.
* Position size comes from :class:`RiskManager` with the current (compounding) equity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.data import storage
from src.data.processor import DOWN, UP, build_dataset
from src.logging_config import get_logger
from src.models.ensemble import DirectionEnsemble
from src.risk.management import RiskManager

logger = get_logger(__name__)

BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520, "1h": 8_760, "4h": 2_190, "1d": 365}


@dataclass
class Trade:
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    size: float
    pnl: float
    return_pct: float  # pnl / equity at entry
    bars_held: int
    exit_reason: str  # STOP / TAKE_PROFIT / TIME
    probability: float


@dataclass
class BacktestMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    buy_and_hold_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    avg_bars_held: float
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    exit_reasons: dict[str, int]
    final_equity: float
    initial_equity: float
    kelly_fraction: float


@dataclass
class BacktestResult:
    mode: str
    symbol: str
    timeframe: str
    start: str
    end: str
    bars: int
    parameters: dict[str, Any]
    metrics: BacktestMetrics
    equity_curve: pd.Series = field(repr=False)
    trades: list[Trade] = field(default_factory=list, repr=False)
    folds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, max_curve_points: int = 2000) -> dict[str, Any]:
        curve = self.equity_curve
        if len(curve) > max_curve_points:
            step = math.ceil(len(curve) / max_curve_points)
            curve = pd.concat([curve.iloc[::step], curve.iloc[[-1]]])
            curve = curve[~curve.index.duplicated()]
        return {
            "mode": self.mode,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "bars": self.bars,
            "parameters": self.parameters,
            "metrics": asdict(self.metrics),
            "equity_curve": [{"timestamp": ts.isoformat(), "equity": float(v)} for ts, v in curve.items()],
            "trades": [asdict(t) for t in self.trades],
            "folds": self.folds,
        }


# ----------------------------------------------------------------------
# Out-of-sample probability generation
# ----------------------------------------------------------------------


def holdout_probabilities(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_fraction: float,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    split = int(len(X) * train_fraction)
    gap = settings.prediction_horizon
    model = DirectionEnsemble(xgb_params=xgb_params, lgbm_params=lgbm_params)
    model.fit(X.iloc[:split], y["direction"].iloc[:split])
    test_X = X.iloc[split + gap:]
    proba = model.predict_proba(test_X)
    probs = pd.DataFrame(proba, index=test_X.index, columns=["p_down", "p_flat", "p_up"])
    fold = {
        "fold": 1,
        "train_start": X.index[0].isoformat(),
        "train_end": X.index[split - 1].isoformat(),
        "test_start": test_X.index[0].isoformat(),
        "test_end": test_X.index[-1].isoformat(),
        "n_test": len(test_X),
    }
    return probs, [fold]


def walk_forward_probabilities(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_window: int,
    test_window: int,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rolling re-training: train on ``train_window`` bars, predict the next ``test_window``."""
    if len(X) < train_window + test_window + settings.prediction_horizon:
        raise ValueError("Not enough rows for a single walk-forward fold")
    gap = settings.prediction_horizon
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    start = 0
    fold_no = 0
    while start + train_window + gap < len(X):
        train_end = start + train_window
        test_start = train_end + gap
        test_end = min(test_start + test_window, len(X))
        if test_end - test_start < 10:
            break
        fold_no += 1
        model = DirectionEnsemble(xgb_params=xgb_params, lgbm_params=lgbm_params)
        model.fit(X.iloc[start:train_end], y["direction"].iloc[start:train_end])
        test_X = X.iloc[test_start:test_end]
        proba = model.predict_proba(test_X)
        parts.append(pd.DataFrame(proba, index=test_X.index, columns=["p_down", "p_flat", "p_up"]))
        y_true = y["direction"].iloc[test_start:test_end].to_numpy()
        folds.append(
            {
                "fold": fold_no,
                "train_start": X.index[start].isoformat(),
                "train_end": X.index[train_end - 1].isoformat(),
                "test_start": test_X.index[0].isoformat(),
                "test_end": test_X.index[-1].isoformat(),
                "n_test": len(test_X),
                "accuracy": float((proba.argmax(axis=1) == y_true).mean()),
            }
        )
        logger.info("Fold %d: test %s -> %s, accuracy %.3f", fold_no, folds[-1]["test_start"], folds[-1]["test_end"], folds[-1]["accuracy"])
        start += test_window
    return pd.concat(parts), folds


# ----------------------------------------------------------------------
# Trade simulation
# ----------------------------------------------------------------------


def simulate_trades(
    ohlcv: pd.DataFrame,
    probs: pd.DataFrame,
    atr_series: pd.Series,
    risk: RiskManager,
    threshold: float,
    take_profit_rr: float,
    max_holding_bars: int,
    fee_pct: float,
    slippage_pct: float,
    allow_short: bool = True,
) -> tuple[list[Trade], pd.Series]:
    """Bar-by-bar simulation returning the trade list and an equity curve."""
    idx = probs.index
    o = ohlcv["open"].reindex(idx).to_numpy()
    h = ohlcv["high"].reindex(idx).to_numpy()
    lo = ohlcv["low"].reindex(idx).to_numpy()
    c = ohlcv["close"].reindex(idx).to_numpy()
    atr = atr_series.reindex(idx).to_numpy()
    p_up = probs["p_up"].to_numpy()
    p_down = probs["p_down"].to_numpy()
    n = len(idx)

    equity = risk.equity
    equity_curve = np.full(n, equity)
    trades: list[Trade] = []

    i = 0
    while i < n - 1:
        side = None
        if p_up[i] >= threshold and p_up[i] > p_down[i]:
            side = "LONG"
        elif allow_short and p_down[i] >= threshold and p_down[i] > p_up[i]:
            side = "SHORT"
        if side is None or not np.isfinite(atr[i]) or atr[i] <= 0:
            equity_curve[i] = equity
            i += 1
            continue

        # Enter at next bar open with adverse slippage.
        entry_idx = i + 1
        raw_entry = o[entry_idx]
        entry = raw_entry * (1 + slippage_pct) if side == "LONG" else raw_entry * (1 - slippage_pct)
        sizing_rm = RiskManager(
            equity=equity,
            risk_per_trade_pct=risk.risk_per_trade_pct,
            atr_multiplier=risk.atr_multiplier,
            reward_risk_ratios=[take_profit_rr],
            max_leverage=risk.max_leverage,
        )
        plan = sizing_rm.build_plan(side, entry, float(atr[i]))
        stop, tp, size = plan.stop_loss, plan.take_profits[0].price, plan.position_size
        equity_curve[i] = equity

        exit_price, exit_reason, exit_idx = None, "TIME", entry_idx
        for j in range(entry_idx, min(entry_idx + max_holding_bars, n)):
            exit_idx = j
            if side == "LONG":
                if lo[j] <= stop:
                    exit_price, exit_reason = stop, "STOP"
                    break
                if h[j] >= tp:
                    exit_price, exit_reason = tp, "TAKE_PROFIT"
                    break
            else:
                if h[j] >= stop:
                    exit_price, exit_reason = stop, "STOP"
                    break
                if lo[j] <= tp:
                    exit_price, exit_reason = tp, "TAKE_PROFIT"
                    break
            # mark-to-market while the trade is open
            mtm = (c[j] - entry) * size if side == "LONG" else (entry - c[j]) * size
            equity_curve[j] = equity + mtm
        if exit_price is None:
            exit_price = c[exit_idx] * (1 - slippage_pct) if side == "LONG" else c[exit_idx] * (1 + slippage_pct)

        gross = (exit_price - entry) * size if side == "LONG" else (entry - exit_price) * size
        fees = fee_pct * size * (entry + exit_price)
        pnl = gross - fees
        trades.append(
            Trade(
                side=side,
                entry_time=idx[entry_idx].isoformat(),
                exit_time=idx[exit_idx].isoformat(),
                entry_price=float(entry),
                exit_price=float(exit_price),
                stop_loss=float(stop),
                take_profit=float(tp),
                size=float(size),
                pnl=float(pnl),
                return_pct=float(pnl / equity * 100),
                bars_held=int(exit_idx - entry_idx + 1),
                exit_reason=exit_reason,
                probability=float(p_up[i] if side == "LONG" else p_down[i]),
            )
        )
        equity += pnl
        equity_curve[exit_idx] = equity
        if equity <= 0:
            logger.warning("Equity wiped out at %s; stopping simulation", idx[exit_idx])
            equity_curve[exit_idx:] = max(equity, 0.0)
            break
        i = exit_idx + 1
    equity_curve[i:] = equity if equity > 0 else 0.0
    return trades, pd.Series(equity_curve, index=idx, name="equity")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def compute_metrics(
    trades: list[Trade], equity_curve: pd.Series, ohlcv: pd.DataFrame, timeframe: str, initial_equity: float
) -> BacktestMetrics:
    bars_per_year = BARS_PER_YEAR.get(timeframe, 8_760)
    rets = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(bars_per_year)) if len(rets) > 1 and rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * math.sqrt(bars_per_year)) if len(downside) > 1 and downside.std() > 0 else 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max - 1.0).fillna(0.0)
    max_dd = float(-drawdown.min() * 100)

    pnls = np.array([t.pnl for t in trades])
    rets_pct = np.array([t.return_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = float(rets_pct[pnls > 0].mean()) if len(wins) else 0.0
    avg_loss = float(-rets_pct[pnls <= 0].mean()) if len(losses) else 0.0

    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]
    long_wr = float(np.mean([t.pnl > 0 for t in longs])) if longs else 0.0
    short_wr = float(np.mean([t.pnl > 0 for t in shorts])) if shorts else 0.0
    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    first_close = float(ohlcv["close"].reindex(equity_curve.index).iloc[0])
    last_close = float(ohlcv["close"].reindex(equity_curve.index).iloc[-1])
    final_equity = float(equity_curve.iloc[-1])
    return BacktestMetrics(
        total_trades=len(trades),
        win_rate=win_rate,
        profit_factor=float(profit_factor) if math.isfinite(profit_factor) else 999.0,
        total_return_pct=float((final_equity / initial_equity - 1) * 100),
        buy_and_hold_return_pct=float((last_close / first_close - 1) * 100),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd,
        avg_trade_return_pct=float(rets_pct.mean()) if len(rets_pct) else 0.0,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy_pct=float(RiskManager.expectancy(win_rate, avg_win, avg_loss)),
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        long_trades=len(longs),
        short_trades=len(shorts),
        long_win_rate=long_wr,
        short_win_rate=short_wr,
        exit_reasons=exit_reasons,
        final_equity=final_equity,
        initial_equity=float(initial_equity),
        kelly_fraction=float(RiskManager.kelly_fraction(win_rate, avg_win, avg_loss)),
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame | None = None,
    mode: str = "holdout",
    threshold: float | None = None,
    take_profit_rr: float | None = None,
    atr_multiplier: float | None = None,
    risk_per_trade_pct: float | None = None,
    initial_equity: float | None = None,
    max_holding_bars: int | None = None,
    train_fraction: float | None = None,
    train_window: int = 6_000,
    test_window: int = 1_000,
    allow_short: bool = True,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> BacktestResult:
    if mode not in ("holdout", "walk_forward"):
        raise ValueError("mode must be 'holdout' or 'walk_forward'")
    threshold = threshold if threshold is not None else settings.signal_probability_threshold
    take_profit_rr = take_profit_rr if take_profit_rr is not None else settings.reward_risk_ratios[0]
    max_holding_bars = max_holding_bars or settings.backtest_max_holding_bars
    train_fraction = train_fraction or settings.train_test_split

    if df is None:
        df = storage.get_ohlcv()
    X, y, ind = build_dataset(df)

    if mode == "holdout":
        probs, folds = holdout_probabilities(X, y, train_fraction, xgb_params, lgbm_params)
    else:
        probs, folds = walk_forward_probabilities(X, y, train_window, test_window, xgb_params, lgbm_params)

    risk = RiskManager(
        equity=initial_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        atr_multiplier=atr_multiplier,
        reward_risk_ratios=[take_profit_rr],
    )
    trades, curve = simulate_trades(
        df, probs, ind["atr_14"], risk, threshold, take_profit_rr, max_holding_bars,
        settings.backtest_fee_pct, settings.backtest_slippage_pct, allow_short,
    )
    metrics = compute_metrics(trades, curve, df, settings.timeframe, risk.equity)
    logger.info(
        "Backtest %s: %d trades, win rate %.1f%%, return %.2f%%, Sharpe %.2f, max DD %.2f%%",
        mode, metrics.total_trades, metrics.win_rate * 100, metrics.total_return_pct, metrics.sharpe_ratio, metrics.max_drawdown_pct,
    )
    return BacktestResult(
        mode=mode,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        start=probs.index[0].isoformat(),
        end=probs.index[-1].isoformat(),
        bars=len(probs),
        parameters={
            "threshold": threshold,
            "take_profit_rr": take_profit_rr,
            "atr_multiplier": risk.atr_multiplier,
            "risk_per_trade_pct": risk.risk_per_trade_pct,
            "max_holding_bars": max_holding_bars,
            "fee_pct": settings.backtest_fee_pct,
            "slippage_pct": settings.backtest_slippage_pct,
            "allow_short": allow_short,
            "train_fraction": train_fraction if mode == "holdout" else None,
            "train_window": train_window if mode == "walk_forward" else None,
            "test_window": test_window if mode == "walk_forward" else None,
        },
        metrics=metrics,
        equity_curve=curve,
        trades=trades,
        folds=folds,
    )


def format_report(result: BacktestResult) -> str:
    m = result.metrics
    lines = [
        f"=== Backtest ({result.mode}) {result.symbol} {result.timeframe} ===",
        f"Period            : {result.start} -> {result.end} ({result.bars} bars, {len(result.folds)} fold(s))",
        f"Trades            : {m.total_trades} (long {m.long_trades} / short {m.short_trades})",
        f"Win rate          : {m.win_rate:.1%} (long {m.long_win_rate:.1%}, short {m.short_win_rate:.1%})",
        f"Profit factor     : {m.profit_factor:.2f}",
        f"Total return      : {m.total_return_pct:+.2f}%  (buy & hold {m.buy_and_hold_return_pct:+.2f}%)",
        f"Sharpe / Sortino  : {m.sharpe_ratio:.2f} / {m.sortino_ratio:.2f}",
        f"Max drawdown      : {m.max_drawdown_pct:.2f}%",
        f"Avg trade         : {m.avg_trade_return_pct:+.3f}% (win {m.avg_win_pct:.3f}%, loss {m.avg_loss_pct:.3f}%)",
        f"Expectancy        : {m.expectancy_pct:+.3f}% per trade, Kelly {m.kelly_fraction:.2f}",
        f"Avg bars held     : {m.avg_bars_held:.1f}   exits {m.exit_reasons}",
        f"Equity            : {m.initial_equity:,.0f} -> {m.final_equity:,.0f}",
    ]
    return "\n".join(lines)


def save_result(result: BacktestResult, name: str) -> None:
    path = settings.models_dir / f"backtest_{name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved backtest result to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BTC strategy backtest")
    parser.add_argument("--mode", choices=["holdout", "walk_forward"], default="walk_forward")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--rr", type=float, default=None, help="take-profit reward/risk ratio")
    parser.add_argument("--atr-mult", type=float, default=None)
    parser.add_argument("--train-window", type=int, default=6_000)
    parser.add_argument("--test-window", type=int, default=1_000)
    parser.add_argument("--no-short", action="store_true")
    args = parser.parse_args()

    result = run_backtest(
        mode=args.mode,
        threshold=args.threshold,
        take_profit_rr=args.rr,
        atr_multiplier=args.atr_mult,
        train_window=args.train_window,
        test_window=args.test_window,
        allow_short=not args.no_short,
    )
    print("\n" + format_report(result))
    save_result(result, args.mode)


if __name__ == "__main__":
    main()
