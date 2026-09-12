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
| Holdout 4h-horizon strategy (last 20 % of data) | Sharpe / max DD | -1.27 / 10.76 % | 2026-09-12 baseline |
| Rolling walk-forward 4h-horizon strategy (12 folds, 470 trades) | Sharpe / max DD | -1.66 / 49.73 % | 2026-09-12 baseline |
| Daily walk-forward (276 days) | Traded directional accuracy | 50.5 % (baseline 47.7 %) | 2026-09-12 regime upweighting |
| Daily walk-forward (276 days) | Target-price MAE | 1.85 % (naive 1.71 %) | 2026-09-12 |
| Daily walk-forward (276 days) | Simulated P&L | -0.94 % (baseline -4.70 %, buy & hold -42.8 %) | 2026-09-12 regime upweighting |

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

**Rolling walk-forward strategy backtest (12 folds × 1,000 bars, train 6,000 bars).**

| Metric | Value |
|---|---|
| Window | 2025-05-30 → 2026-09-12 (11,273 OOS bars) |
| Trades | 470 (275 long / 195 short) |
| Win rate | 36.0 % (long 39.6 %, short 30.8 %) |
| Profit factor | 0.83 |
| Total return | -43.69 % (buy & hold -25.59 %) |
| Sharpe / Sortino | -1.66 / -1.47 |
| Max drawdown | 49.73 % |
| Exits | 281 stop / 122 take-profit / 67 time |
| Fold accuracy | 0.37 – 0.45 (3-class, chance 0.33) |

The 4h-horizon strategy has **no exploitable edge** in its current form: per-fold accuracy is
only slightly above chance and 60 % of trades stop out. Shorts are markedly worse than longs.

**Open hypotheses (one per iteration).**

* Longer horizon (12–24 bars) with a volatility-scaled label threshold (±0.5 × ATR %).
* Trade only when both boosters agree (XGB and LGBM argmax equal).
* Stop at 2.0 × ATR with 1:2 target to cut stop-outs on noise.
* Skip high-volatility regime (ATR% above 80th trailing percentile).
* Add funding/liquidation proxies or BTC dominance (needs new data source).

## 2026-09-12 20:40 UTC — Walk-forward daily evaluation (2025-09-12 → 2026-06-14)

### Baseline

| Metric | Value |
|---|---|
| Days evaluated | 276 (retrains: 40, every 7d) |
| Directional accuracy (all / traded) | 48.2% / 47.7% (195 trades) |
| Target price MAE / RMSE | 1.86% / 2.67% (1,577 / 2,201 USD) |
| Naive (no-change) MAE | 1.71% |
| Simulated P&L | -470 USDT (-4.70%), buy&hold -42.83% |
| Win rate / profit factor | 43.1% / 0.85 |
| Sharpe / max drawdown | -0.95 / 5.68% |

### Error analysis (misses)

* 143 misses out of 276 days (51.8%).
* Mean ATR% on misses 0.68 vs hits 0.68; mean confidence on misses 0.61 vs hits 0.60.
* Miss rate by pattern (lift vs overall):
  * `overbought`: 13 days, miss 84.6% (+32.8%), abs err 1.29%
  * `oversold`: 13 days, miss 61.5% (+9.7%), abs err 2.57%
  * `low_volatility`: 60 days, miss 55.0% (+3.2%), abs err 1.44%
  * `large_move`: 40 days, miss 55.0% (+3.2%), abs err 5.08%
  * `trend_up`: 82 days, miss 53.7% (+1.8%), abs err 1.55%
  * `low_volume`: 170 days, miss 52.9% (+1.1%), abs err 1.70%
  * `trend_down`: 115 days, miss 52.2% (+0.4%), abs err 2.36%
  * `high_volatility`: 72 days, miss 51.4% (-0.4%), abs err 3.00%
  * `trend_reversal`: 97 days, miss 46.4% (-5.4%), abs err 1.81%
  * `volume_spike`: 18 days, miss 44.4% (-7.4%), abs err 1.31%
* Worst patterns: overbought, oversold, low_volatility, large_move

### Self-learning feedback loop

Tuned on the first half of the window, validated on the second half (score = traded accuracy + 0.001 × return%).

| Candidate | Tune traded acc. | Tune return | Tune MAE | Score | Chosen |
|---|---|---|---|---|---|
| baseline | 51.5% | -1.12% | 1.77% | 0.5137 |  |
| regularised | 48.0% | -5.20% | 1.77% | 0.4748 |  |
| upweight_trend_up+high_volatility | 53.8% | +0.64% | 1.69% | 0.5383 | ✅ |
| skip_trend_up | 53.3% | +0.01% | 1.77% | 0.5333 |  |
| higher_threshold | 45.1% | -3.43% | 1.77% | 0.4473 |  |

Validation half — baseline: traded acc 40.4%, return -3.98%, MAE 2.01% | chosen (upweight_trend_up+high_volatility): traded acc 45.0%, return -2.24%, MAE 2.01%

**Decision:** adopted `upweight_trend_up+high_volatility`.
Adjustments: xgb={"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05}, lgbm={"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.05}, regime_weights={'trend_up': 1.5, 'high_volatility': 1.5}, skip_regimes=[], threshold=0.55

### Full-window re-run with the chosen configuration

| Metric | Baseline | Chosen |
|---|---|---|
| Directional accuracy (all) | 48.2% | 46.4% |
| Directional accuracy (traded) | 47.7% | 50.5% |
| Trades | 195 | 184 |
| MAE / RMSE | 1.86% / 2.67% | 1.85% / 2.68% |
| P&L | -4.70% | -0.94% |
| Sharpe / max DD | -0.95 / 5.68% | -0.14 / 5.98% |

## 2026-09-12 21:00 UTC — Strategy-layer parameter sweep (out-of-sample)

`scripts/param_sweep.py`: the ensemble was fitted once per rolling fold (12 folds, same as
the walk-forward backtest) and 96 risk/threshold combinations were simulated on the fixed OOS
probabilities (threshold 0.50–0.65 × ATR stop 1.5/2.0/3.0 × R:R 2/3 × holding 24/48 bars ×
booster-agreement filter). Window 2025-05-30 → 2026-09-12, 11,273 bars.

| Rank | Threshold | ATR × | R:R | Hold | Trades | Win rate | PF | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| best | 0.60 | 3.0 | 2 | 48 | 183 | 41.5 % | 0.86 | -13.45 % | -0.75 | 23.4 % |
| 2 | 0.60 | 3.0 | 3 | 48 | 179 | 39.7 % | 0.84 | -15.41 % | -0.84 | 25.8 % |
| current defaults | 0.55 | 1.5 | 2 | 24 | 470 | 36.0 % | 0.83 | -43.69 % | -1.66 | 49.7 % |
| worst | 0.65 | 2.0 | 3 | 24 | 153 | 29.4 % | 0.46 | -43.02 % | -3.25 | 47.0 % |

**Learnings.**

1. **No risk-layer setting turns the 1h model profitable** — all 96 combinations have a
   profit factor below 1. The edge has to come from the model / labels, not from stops or
   thresholds.
2. Wider stops (3 × ATR) and longer holding (48 bars) roughly halve the drawdown because the
   signal is too noisy for tight intraday stops; this is worth adopting once the model has an
   edge, but it is not adopted now (selecting the best of 96 OOS rows would be mild overfitting,
   and `tests/test_api.py` pins the 1.5 × ATR default).
3. Requiring XGBoost and LightGBM to agree changes almost nothing: the boosters already agree
   whenever either is confident.
4. Raising the threshold to 0.65 *hurts* (fewer, worse trades) — high ensemble confidence on 1h
   bars is not associated with better outcomes, another sign of mis-calibration.

**Next hypotheses (priority order).**

1. Daily-horizon production model (the daily walk-forward reaches 50.5 % traded accuracy with
   regime upweighting vs ~40 % 3-class accuracy at 4h) — promote the `walk_forward.py`
   DailyModels approach to `src/models/train.py` with a 24-bar horizon.
2. Volatility-scaled label threshold (±0.5 × ATR%) to stop the FLAT class from absorbing most
   of the signal.
3. Probability calibration (isotonic on a rolling OOS window) before thresholding.
4. Keep `regime_weights={'trend_up': 1.5, 'high_volatility': 1.5}` (adopted for the daily model).
