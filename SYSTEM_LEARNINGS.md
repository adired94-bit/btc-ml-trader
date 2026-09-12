# SYSTEM_LEARNINGS

Memory bank for the BTC/USDT ML trading platform. Every iteration appends a dated entry:
what was tried, the **out-of-sample** numbers it produced, and the decision. Autonomous
agents must read this file before changing anything and must never report in-sample metrics
here.

Conventions:

* Data: Binance `BTC/USDT`, 1h candles, ~2 years (17,520 bars) cached in `data/`.
* Production model: 3-class (DOWN / FLAT / UP, ±0.15 % over 4 bars) XGBoost + LightGBM
  soft-voting ensemble + RandomForest range regressor (`src/models`).
* Backtests: signals on the close of bar *t* execute at the open of *t+1*; ATR(14) × 1.5
  stop, 1:2 take-profit, 1 % equity risk per trade, 0.04 % fee + 0.02 % slippage per side.
* Daily walk-forward evaluation (`src/backtest/walk_forward.py`): binary next-day direction,
  target price and risk level, retrained weekly on data ≤ T only.

## Current best (update when beaten)

| Evaluation | Metric | Value | Iteration |
|---|---|---|---|
| Holdout 4h-horizon strategy (last 20 % of data) | Sharpe | -1.27 | 2026-09-12 baseline |
| Holdout 4h-horizon strategy | Max drawdown | 10.76 % | 2026-09-12 baseline |
| Daily walk-forward | Directional accuracy | see entries below | |

## 2026-09-12 — Baseline build

**Setup.** First end-to-end build. Chronological split 80/20 with a 4-bar gap.
Test window 2026-04-21 → 2026-09-12 (3,452 hourly bars). Default booster parameters, no
hyper-parameter search.

**Direction model (OOS test window).**

| Metric | Value |
|---|---|
| Accuracy / balanced accuracy | 0.381 / 0.394 (chance = 0.333) |
| Log-loss ensemble / XGB / LGBM | 1.0887 / 1.0867 / 1.0942 (uniform = 1.0986) |
| Signal coverage at p ≥ 0.55 | 4.3 % of bars (150 signals) |
| Signal directional accuracy | 40.7 % |
| Class balance (train) | UP 40 % / DOWN 38 % / FLAT 22 % |
| Top features | atr_pct, dow_sin, volatility_72, volatility_24, atr_pct_change, ema_50_200_spread, dist_poc_pct |

**Range model.** MAE of max-up / max-down excursion 0.398 % / 0.400 % over 4 bars.

**Holdout strategy backtest (same window).**

| Metric | Value |
|---|---|
| Trades | 44 (29 long / 15 short) |
| Win rate | 38.6 % |
| Profit factor | 0.74 |
| Total return | -6.96 % (buy & hold +2.22 %) |
| Sharpe / Sortino | -1.27 / -0.65 |
| Max drawdown | 10.76 % |
| Exits | 26 stop / 9 take-profit / 9 time |

**Learnings.**

1. The ensemble is only marginally better than chance on 1h bars (log-loss 1.089 vs
   1.099 uniform). Volatility features dominate importance: the model mostly learns *when*
   moves are large, not *which way* they go.
2. 59 % of trades end at the stop. With a 1.5 × ATR stop and 1:2 target the win rate needed
   to break even is ~35 %, so the edge is thin and fees/slippage eat it.
3. `atr_pct` being the top feature suggests regime-aware filtering (skip high-vol regimes or
   widen stops) is worth testing before adding more features.

**Open hypotheses (one per iteration).**

* Longer horizon (12–24 bars) with a volatility-scaled label threshold (±0.5 × ATR %).
* Trade only when both boosters agree (XGB and LGBM argmax equal).
* Stop at 2.0 × ATR with 1:2 target to cut stop-outs on noise.
* Skip high-volatility regime (ATR% above 80th trailing percentile).
* Add funding/liquidation proxies or BTC dominance (needs new data source).
