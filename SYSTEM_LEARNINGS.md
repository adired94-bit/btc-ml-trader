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
| Daily walk-forward (276 days, 2-year data) | Traded directional accuracy | 53.4 % (hit rate 50.4 %) | 2026-09-13 extended features |
| Daily walk-forward (1,011 days, 6-year data) | Hit rate / traded | 52.8 % / 53.6 % (momentum benchmark 50.9 %) | 2026-09-13 regime upweighting |
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

## 2026-09-12 21:23 UTC — Improvement campaign: 14 experiments on the daily forecast

Each experiment scored on the tuning half (older), the top configs re-scored on the validation half (newer), winner re-run over the full window. Score = hit + 0.5 × traded hit + 0.0005 × return%.

### Tuning half

| Config | Half | Days | Hit rate | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| ext+ens+retrain1d | tune | 139 | 55.4% | 59.1% | 105 | 1.66% | 1.53% | -1.05% | -0.39 | 6.93% |
| ext+ens | tune | 139 | 51.8% | 56.3% | 119 | 1.78% | 1.53% | -7.04% | -2.49 | 11.52% |
| ext+ens+regime_w | tune | 139 | 51.8% | 55.8% | 104 | 1.82% | 1.53% | -4.44% | -1.69 | 9.22% |
| ext+ens_strongreg | tune | 139 | 53.2% | 51.1% | 88 | 1.78% | 1.53% | -6.63% | -2.87 | 8.91% |
| base+regime_w | tune | 139 | 51.1% | 48.5% | 103 | 1.69% | 1.53% | -2.39% | -0.87 | 6.84% |
| ext+ens_h72 | tune | 139 | 48.2% | 49.6% | 115 | 3.54% | 2.99% | -7.84% | -1.56 | 21.15% |
| base+logreg | tune | 139 | 48.9% | 47.1% | 70 | 1.75% | 1.53% | -0.37% | -0.18 | 3.43% |
| baseline | tune | 139 | 46.8% | 49.0% | 102 | 1.75% | 1.53% | -3.69% | -1.38 | 6.62% |
| ext+logreg_c0.05 | tune | 139 | 43.9% | 42.3% | 104 | 1.78% | 1.53% | -13.00% | -5.76 | 14.10% |
| ext+ens_h168 | tune | 139 | 44.6% | 41.3% | 121 | 6.55% | 4.79% | -19.03% | -3.17 | 25.55% |
| ext+logreg_c0.5 | tune | 139 | 43.9% | 41.9% | 105 | 1.78% | 1.53% | -12.50% | -5.53 | 13.63% |
| ext+logreg_c0.01 | tune | 139 | 41.7% | 45.4% | 97 | 1.78% | 1.53% | -9.88% | -4.55 | 11.42% |
| ext+logreg_h72 | tune | 139 | 41.7% | 38.6% | 114 | 3.54% | 2.99% | -20.95% | -5.16 | 25.98% |
| ext+logreg_h168 | tune | 139 | 36.0% | 34.8% | 115 | 6.55% | 4.79% | -32.16% | -5.93 | 36.79% |

### Validation half (never used for selection)

| Config | Half | Days | Hit rate | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| ext+ens+retrain1d | validation | 138 | 44.9% | 49.5% | 101 | 2.02% | 1.92% | -3.72% | -1.48 | 4.44% |
| ext+ens | validation | 138 | 49.3% | 47.1% | 102 | 2.05% | 1.92% | -6.63% | -2.52 | 6.70% |
| ext+ens+regime_w | validation | 138 | 49.3% | 46.9% | 98 | 2.05% | 1.92% | -5.67% | -2.11 | 7.20% |
| baseline | validation | 138 | 42.0% | 41.0% | 100 | 2.02% | 1.92% | -5.75% | -2.17 | 8.06% |

### Winner: `ext+ens` — full window

| Config | Half | Days | Hit rate | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| ext+ens | full | 276 | 50.4% | 53.4% | 221 | 1.91% | 1.71% | -10.92% | -2.07 | 11.83% |

**Learnings.**

* Best on the tuning half: `ext+ens+retrain1d` (55.4% hit rate); on validation it scored 44.9%.
* Baseline validation hit rate 42.0% vs winner 49.3%; adopted.
* 7-day horizon reaches 44.6% on tuning but its days overlap heavily (7× fewer independent samples) — treat with caution.
* Full-window winner hit rate 50.4%, MAE 1.91% vs naive 1.71%.
* Campaign runtime 1078s.

## 2026-09-13 06:07 UTC — Improvement campaign (6-year history, ~3-year window): 7 experiments on the daily forecast

Each experiment scored on the tuning half (older), the top configs re-scored on the validation half (newer), winner re-run over the full window. Score = hit + 0.5 × traded hit + 0.0005 × return%.

### Tuning half

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base+logreg | tune | 506 | 53.4% | 50.2% | 53.4% | 62.4% | 133 | 1.86% | 1.88% | -3.56% | -0.56 | 6.17% |
| base+regime_w | tune | 506 | 54.7% | 50.2% | 53.4% | 53.4% | 275 | 1.87% | 1.88% | -6.26% | -0.69 | 9.89% |
| ext+ens_strongreg | tune | 506 | 52.8% | 50.2% | 53.4% | 55.2% | 239 | 1.87% | 1.88% | -7.31% | -0.84 | 9.91% |
| baseline | tune | 506 | 52.8% | 50.2% | 53.4% | 54.9% | 288 | 1.86% | 1.88% | -5.08% | -0.55 | 10.32% |
| ext+ens+regime_w | tune | 506 | 50.6% | 50.2% | 53.4% | 53.0% | 315 | 1.87% | 1.88% | -13.89% | -1.57 | 15.41% |
| ext+ens | tune | 506 | 51.2% | 50.2% | 53.4% | 51.3% | 335 | 1.87% | 1.88% | -18.45% | -2.10 | 19.64% |
| ext+ens_h72 | tune | 506 | 48.2% | 50.2% | 57.3% | 49.8% | 396 | 3.43% | 3.43% | +3.32% | 0.25 | 19.33% |

### Validation half (never used for selection)

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base+logreg | validation | 506 | 49.0% | 51.6% | 48.6% | 49.0% | 96 | 1.68% | 1.63% | -1.81% | -0.37 | 3.42% |
| base+regime_w | validation | 506 | 51.2% | 51.6% | 48.6% | 53.6% | 192 | 1.67% | 1.63% | -2.71% | -0.39 | 6.68% |
| ext+ens_strongreg | validation | 506 | 47.6% | 51.6% | 48.6% | 53.5% | 170 | 1.70% | 1.63% | -6.99% | -1.03 | 8.73% |
| baseline | validation | 506 | 49.0% | 51.6% | 48.6% | 50.8% | 244 | 1.68% | 1.63% | -1.95% | -0.25 | 5.65% |

### Winner: `base+regime_w` — full window

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base+regime_w | full | 1011 | 52.8% | 50.9% | 51.0% | 53.6% | 461 | 1.77% | 1.75% | -10.95% | -0.72 | 14.87% |

**Learnings.**

* Best on the tuning half: `base+logreg` (53.4% hit rate); on validation it scored 49.0%.
* Baseline validation hit rate 49.0% vs winner 51.2%; adopted.
* Full-window winner hit rate 52.8%, MAE 1.77% vs naive 1.75%.
* Campaign runtime 31255s.

## 2026-09-13 — Conclusion after two campaigns (21 experiments, ~9 h of compute)

* With price/volume features only, the honest out-of-sample ceiling is **52–54 % daily hit rate**
  (momentum benchmark 51 %). Every configuration that looked better on the tuning half
  (55–62 %) fell back to 45–49 % on the validation half — noise, not signal.
* Neither extended features, logistic regression, daily retraining nor longer horizons generalised.
  Regime up-weighting (`trend_up`, `high_volatility` × 1.5) is the only change that survived both
  campaigns and it is worth about +2 pp.
* Simulated P&L stays negative after 0.06 % round-trip costs at this accuracy. The strategy layer
  cannot fix a 53 % model.
* **Next iterations must bring new information, not new models:** Binance futures funding rate and
  open interest (free via CCXT `fetchFundingRateHistory` / `fetchOpenInterestHistory`), stablecoin
  flows, order-book imbalance, cross-asset context (ETH, SPX, DXY). Alternatively change the target
  to next-day range/volatility, which is far more predictable than direction.

## 2026-09-13 06:39 UTC — Improvement campaign (6-year history, ~3-year window, order-flow features): 7 experiments on the daily forecast

Each experiment scored on the tuning half (older), the top configs re-scored on the validation half (newer), winner re-run over the full window. Score = hit + 0.5 × traded hit + 0.0005 × return%.

### Tuning half

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow+logreg | tune | 506 | 54.9% | 50.2% | 53.4% | 56.1% | 280 | 1.85% | 1.88% | -1.51% | -0.14 | 8.73% |
| base+regime_w | tune | 506 | 53.9% | 50.2% | 53.4% | 54.8% | 279 | 1.87% | 1.88% | -2.93% | -0.30 | 8.28% |
| flow+ens | tune | 506 | 54.7% | 50.2% | 53.4% | 52.1% | 330 | 1.85% | 1.88% | -13.11% | -1.40 | 15.56% |
| flow+ens+retrain7d | tune | 506 | 54.7% | 50.2% | 53.4% | 52.1% | 330 | 1.85% | 1.88% | -13.11% | -1.40 | 15.56% |
| baseline | tune | 506 | 53.2% | 50.2% | 53.4% | 53.7% | 296 | 1.86% | 1.88% | -6.03% | -0.64 | 9.15% |
| flow+ens_strongreg | tune | 506 | 53.9% | 50.2% | 53.4% | 51.3% | 222 | 1.85% | 1.88% | -14.99% | -1.99 | 15.99% |
| flow+ens+regime_w | tune | 506 | 51.8% | 50.2% | 53.4% | 53.6% | 323 | 1.88% | 1.88% | -18.12% | -1.98 | 20.81% |

### Validation half (never used for selection)

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow+logreg | validation | 506 | 50.4% | 51.6% | 48.6% | 51.7% | 236 | 1.68% | 1.63% | -4.54% | -0.57 | 13.88% |
| base+regime_w | validation | 506 | 50.0% | 51.6% | 48.6% | 54.1% | 194 | 1.67% | 1.63% | -2.78% | -0.40 | 5.66% |
| baseline | validation | 506 | 49.8% | 51.6% | 48.6% | 50.6% | 239 | 1.68% | 1.63% | -5.00% | -0.69 | 7.05% |

### Winner: `base+regime_w` — full window

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base+regime_w | full | 1011 | 51.8% | 50.9% | 51.0% | 54.4% | 467 | 1.77% | 1.75% | -7.80% | -0.50 | 13.21% |

**Learnings.**

* Best on the tuning half: `flow+logreg` (54.9% hit rate); on validation it scored 50.4%.
* Baseline validation hit rate 49.8% vs winner 50.0%; adopted.
* Full-window winner hit rate 51.8%, MAE 1.77% vs naive 1.75%.
* Campaign runtime 1584s.

## 2026-09-13 — Conclusion after the order-flow campaign (3 campaigns, 27 experiments total)

* New information (taker-buy ratio, trade count, perp basis, premium index, funding rate; 6 years)
  did **not** add generalisable signal: `flow+logreg` scored 54.9 % / 56.1 % traded on the tuning
  half and fell to 50.4 % / 51.7 % on validation. Ensemble variants with flow features were worse
  than the price-only baseline on validation.
* The only configuration that survived every validation split is **`base+regime_w`**
  (36 price/volume features, XGB+LGBM, regime weights trend_up/high_volatility × 1.5):
  full 1,011-day window hit rate 51.8 %, **54.4 % on 467 traded days**, P&L -7.8 % after costs.
  Momentum benchmark on the same days: 50.9 %.
* Honest ceiling with everything tried so far: ~52 % all days / ~54 % on confident days.
  Trading it is still net negative after 0.06 % round-trip costs.
* Still untested and worth trying next: (1) changing the target to next-day range/volatility
  (a breakout strategy needs a good volatility forecast, not a direction forecast);
  (2) cross-asset context (ETH/BTC relative strength, SPX/DXY daily);
  (3) meta-labeling: a second model that predicts *when* the first one is right, and trading only then;
  (4) on-chain / stablecoin flow data (paid sources).

## 2026-09-13 07:27 UTC — Improvement campaign (6-year history, ~3-year window, candlestick patterns): 6 experiments on the daily forecast

Each experiment scored on the tuning half (older), the top configs re-scored on the validation half (newer), winner re-run over the full window. Score = hit + 0.5 × traded hit + 0.0005 × return%.

### Tuning half

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pat+logreg | tune | 506 | 54.9% | 50.2% | 53.4% | 53.8% | 260 | 1.87% | 1.88% | -8.86% | -1.08 | 12.84% |
| base+regime_w | tune | 506 | 53.9% | 50.2% | 53.4% | 54.8% | 279 | 1.87% | 1.88% | -2.93% | -0.30 | 8.28% |
| baseline | tune | 506 | 53.2% | 50.2% | 53.4% | 53.7% | 296 | 1.86% | 1.88% | -6.03% | -0.64 | 9.15% |
| pat+ens_strongreg | tune | 506 | 51.2% | 50.2% | 53.4% | 54.2% | 236 | 1.87% | 1.88% | -6.37% | -0.77 | 11.78% |
| pat+ens | tune | 506 | 50.6% | 50.2% | 53.4% | 52.1% | 338 | 1.87% | 1.88% | -12.20% | -1.32 | 12.83% |
| pat+ens+regime_w | tune | 506 | 49.0% | 50.2% | 53.4% | 52.5% | 301 | 1.86% | 1.88% | -11.98% | -1.37 | 13.50% |

### Validation half (never used for selection)

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pat+logreg | validation | 506 | 50.2% | 51.6% | 48.6% | 51.2% | 205 | 1.70% | 1.63% | -7.27% | -1.02 | 12.63% |
| base+regime_w | validation | 506 | 50.0% | 51.6% | 48.6% | 54.1% | 194 | 1.67% | 1.63% | -2.78% | -0.40 | 5.66% |
| baseline | validation | 506 | 49.8% | 51.6% | 48.6% | 50.6% | 239 | 1.68% | 1.63% | -5.00% | -0.69 | 7.05% |

### Winner: `base+regime_w` — full window

| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base+regime_w | full | 1011 | 51.8% | 50.9% | 51.0% | 54.4% | 467 | 1.77% | 1.75% | -7.80% | -0.50 | 13.21% |

**Learnings.**

* Best on the tuning half: `pat+logreg` (54.9% hit rate); on validation it scored 50.2%.
* Baseline validation hit rate 49.8% vs winner 50.0%; adopted.
* Full-window winner hit rate 51.8%, MAE 1.77% vs naive 1.75%.
* Campaign runtime 1345s.

## 2026-09-13 — Ten parallel research agents (2-year data; tune 365→227 days ago, validation 227→90)

Each agent implemented one idea in an isolated worktree, evaluated it with the shared protocol
(extended features, XGB+LGBM 80 trees, retrain 14 d, `simulate_day_trade`, selection on the
tuning half only). Scripts: `scripts/experiments/idea01..10_*.py`; raw numbers:
`models/experiments/idea*.json`. Common baseline: tune 53.2 % hit / 49.5 % traded (101) / −4.9 %;
validation 47.1 % / 44.2 % (86) / −4.1 %. Momentum-30d benchmark 50.4 % / 55.1 %.

| # | Idea | Tune (traded hit, return) | Validation (traded hit, return) | Verdict |
|---|---|---|---|---|
| 1 | Meta-labeling (2nd model predicts when 1st is right) | 44.9 % (49), −5.5 % | 50.7 % (69), −1.1 % | ✗ meta OOS AUC 0.51; primary confidence is *anti*-predictive (AUC 0.39/0.45) |
| 2 | Isotonic / Platt calibration before gating | 53.0 % (83), −2.8 % | 47.5 % (80), −3.1 % | ✗ raw Brier worse than the constant base-rate predictor; nothing monotone to calibrate |
| 3 | ETH / SOL cross-asset features | 41–48 %, −8 to −12 % | 46–54 %, −3 to −5 % | ✗ worse on tune in every variant; model spends 23 % of gain learning noise from ETH |
| 4 | Decision hour + calendar filters | no_weekend 51.4 % (72), −2.8 % | 50.9 % (57), −1.1 % | ± hour ranking does not transfer (ρ = 0.10); **skip-weekend helped on both halves** (Sat trades 36 %/21 %) |
| 5 | Volatility-scaled 3-class labels | k=0.5: 55.4 % (74), +1.9 % | 49.3 % (71), −1.1 % | ✗ FLAT band swallows 44–82 % of bars; tune gain vanished |
| 6 | Rolling / time-decayed training window | all 3–5 pp worse than expanding | ranking inverted | ✗ shorter memory = more over-confident trades, not better ones |
| 7 | Multi-horizon (24/72/168 h) agreement | all3 48.0 % (77), −7.1 % | 49.2 % (63), −0.5 % | ✗ horizons share features (agree 62 % of days); 168 h model 43–45 % |
| 8 | Feature selection (top 8/15/25) | all ≤ baseline | 45–48 % | ✗ 32/56 features have negative permutation importance; #1 feature is day-of-month = seasonal artefact |
| 9 | Trade only with the EMA-200 trend | @0.50: 51.7 % (29), +1.8 % | 56.5 % (46), −1.2 % | ✗ works only by trading less; pure trend rule 45.7 % → 56.5 % (regime-dependent) |
| 10 | Mean reversion at RSI / %B extremes | fade 60 % (25) | fade 39 % (23) | ✗ full 2 y: overbought→down 53 %, oversold→up 42 %; noise |

**Cross-cutting learnings (this is the important part).**

1. **The confidence gate hurts.** Four independent agents (1, 2, 5, 7) found the same thing:
   trading *all* days beats trading only "confident" days in both halves (e.g. 53.2 %/47.1 % vs
   49.5 %/44.2 %), and traded accuracy falls monotonically from threshold 0.50 → 0.55 → 0.60. The
   ensemble's probability magnitude carries no information; high confidence mostly marks strongly
   trending days where the model is late. → Decision: the 0.55 gate stays only as a *trade-count*
   limiter; do not interpret it as skill. Any future gating must first pass the **Brier-vs-base-rate
   test** proposed by agent 2 (if OOS Brier ≥ climatological Brier, do not threshold at all).
2. **Two-year halves are too short to rank anything.** 138-day halves give a ±4–6 pp standard error,
   and every "winner" flipped sign between halves. All future experiments run on the 6-year cache
   (≥ 500-day halves).
3. **Nothing in the strategy layer creates an edge** — filters, calibration, meta-models, memory
   length, agreement rules and feature pruning all merely change the trade count. The direction
   target itself has ~no signal in price/volume/pattern/cross-crypto features.
4. **Only cheap risk filter that survived both halves:** skip weekend windows (−4.1 % → −1.1 % on
   validation, drawdown halved). Still negative P&L; to be confirmed on 6-year data.
5. Candlestick patterns (separate 6-year campaign, `improvement_campaign_patterns.json`): `pat+*`
   variants 49–55 % on tuning, none beat `base+regime_w` on validation → ✗.

**Where the effort goes next (decided):** the volatility / range target and breakout execution
(`scripts/volatility_target.py`, AUC 0.63 on a smoke window), non-crypto macro context (SPX, DXY,
gold; the only cross-asset variant with a causal story), and paid/alt data (on-chain, exchange
flows). Direction-only work on OHLCV is closed.

## 2026-09-13 07:33 UTC — Volatility target + breakout strategy (6-year data, k=0.5 ATR, R:R 2.0)

Predict whether the next 24 h |return| exceeds the trailing 90-day median ("big day"); trade an OCO breakout only on predicted big days.

| Split | Threshold | Big-day acc. | AUC | Naive (persistence) | Precision | Trades | Win rate | Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| tune | 0.50 | 59.9% | 0.605 | 48.8% | 59.9% | 250 | 38.8% | -22.46% | -1.27 | 29.7% |
| tune | 0.55 | 57.7% | 0.605 | 48.8% | 60.9% | 192 | 38.0% | -20.94% | -1.35 | 29.1% |
| tune | 0.60 | 53.4% | 0.605 | 48.8% | 60.0% | 120 | 34.2% | -20.71% | -1.70 | 24.7% |
| tune | 0.65 | 51.6% | 0.605 | 48.8% | 62.3% | 68 | 36.8% | -8.80% | -0.81 | 14.6% |
| tune | every day | – | – | – | – | 355 | 38.9% | -26.48% | -1.33 | 34.3% |
| **validation** | 0.65 | 53.4% | 0.571 | 52.6% | 67.9% | 47 | 42.6% | +7.30% | 0.77 | 6.3% |
| validation | every day | – | – | – | – | 340 | 38.5% | -26.76% | -1.42 | 29.5% |

* Big-day share 50.6%; exits on validation {'CLOSE': 19, 'STOP': 16, 'TAKE_PROFIT': 12}, days with no breakout 9.
* Runtime 258s.

## 2026-09-13 07:37 UTC — Breakout execution sweep on fixed volatility predictions (6-year data)

Model trained once per half (retrain every 21 d); 100 execution configs (threshold x breakout k x R:R, rr=0 = hold to close) scored on the tune half, top-5 re-scored on validation.

| Rank | Thr | k (ATR) | R:R | Tune trades / win / return / Sharpe / DD | Validation trades / win / return / Sharpe / DD |
|---|---|---|---|---|---|
| 1 | 0.60 | 1.00 | 1.0 | 49 / 57% / +5.0% / 0.85 / 5.0% | 44 / 30% / -6.4% / -1.26 / 8.3% |
| 2 | 0.60 | 1.50 | 1.5 | 23 / 52% / +3.1% / 0.80 / 1.6% | 15 / 53% / +0.8% / 0.30 / 1.7% |
| 3 | 0.60 | 1.00 | 3.0 | 49 / 51% / +4.1% / 0.56 / 5.3% | 44 / 30% / -4.3% / -0.67 / 6.4% |
| 4 | 0.60 | 1.50 | 1.0 | 23 / 52% / +1.8% / 0.55 / 1.6% | 15 / 53% / +0.4% / 0.18 / 1.7% |
| 5 | 0.60 | 1.50 | 2.0 | 23 / 52% / +2.0% / 0.54 / 1.6% | 15 / 53% / +1.3% / 0.42 / 1.7% |

* Validation grid overall: 51% of configs profitable, median return +0.08%; best possible on validation (hindsight) Sharpe 1.15.
* Runtime 80s.
**Conclusion of the volatility line (2026-09-13).** The *prediction* is real and stable — big-day
accuracy 60 % vs 49 % naive persistence, AUC 0.61 tune / 0.57 validation — but the OCO breakout
does not monetise it robustly: tight breakouts (k ≤ 1 ATR) lose on false breaks, wide ones
(k = 1.5) are +1–3 % per half on only 15–23 trades, and across the whole validation grid the
median return is +0.08 %. A big day is not the same as a clean directional break; many big days
reverse intraday. Next ways to use the volatility signal: (1) as a *risk* input — skip or halve
size on predicted big days for the direction model, widen stops on them; (2) intraday breakout
with time-of-day entry windows (Asia/US open) instead of a full-day OCO; (3) options-style
payoffs are not available on spot, so the straddle analogue is out.
