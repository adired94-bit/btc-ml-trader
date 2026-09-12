# CLAUDE.md — BTC/USDT ML Trading Signal Platform

Execution standards for any engineer or autonomous agent working in this repository.

## 1. What this project is

A production-style Bitcoin (BTC/USDT, 1h) direction-forecasting and trading-signal
system:

| Layer | Path | Responsibility |
|-------|------|----------------|
| Config | `config.py` | Pydantic settings, `.env` overrides, paths |
| Data | `src/data/fetcher.py` | CCXT (Binance → Bybit → OKX → Kraken fallback) OHLCV + ticker |
| Data | `src/data/storage.py` | CSV cache with incremental refresh |
| Data | `src/data/processor.py` | RSI, EMA 20/50/200, ATR, Bollinger, VWAP, MACD, Volume Profile, features, labels |
| ML | `src/models/ensemble.py` | XGBoost + LightGBM soft-voting classifier, RandomForest range regressor |
| ML | `src/models/train.py` | Chronological split with gap, optional randomized search (TimeSeriesSplit), artifacts |
| ML | `src/models/predict.py` | Lazy model loading, prediction, signal + trade plan |
| Risk | `src/risk/management.py` | ATR stop-loss, 1:2 / 1:3 take-profits, position sizing, leverage cap, Kelly |
| Backtest | `src/backtest/engine.py` | Holdout & walk-forward simulation, Sharpe, win rate, max drawdown |
| API | `src/api/main.py` | FastAPI REST server |
| UI | `app.py` | Streamlit dashboard (thin client over the API) |
| Tests | `tests/` | pytest suite (network-free, synthetic data) |
| Memory | `SYSTEM_LEARNINGS.md` | Iteration log: OOS metrics, what worked, what did not |

## 2. Commands (Windows, from the repo root)

```bash
# environment
venv\Scripts\python.exe -m pip install -r requirements.txt

# tests (must be green before every commit)
venv\Scripts\python.exe -m pytest tests/

# data + training
venv\Scripts\python.exe -m src.models.train            # default params
venv\Scripts\python.exe -m src.models.train --tune     # randomized search

# backtests (results saved to models/backtest_<mode>.json)
venv\Scripts\python.exe -m src.backtest.engine --mode holdout
venv\Scripts\python.exe -m src.backtest.engine --mode walk_forward

# servers
venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
# or both at once
powershell -ExecutionPolicy Bypass -File run.ps1

# self-improvement loop
bash autoloop.sh            # AUTOLOOP_MAX_ITERATIONS / AUTOLOOP_SLEEP_SECONDS env vars
```

## 3. Non-negotiable rules

1. **Test immutability.** Never edit, delete, skip or weaken anything in `tests/` to make
   a run pass. Fix the code in `src/`. Adding *new* tests is encouraged.
2. **No look-ahead bias.** Any feature at bar *t* may only use bars `<= t`. Labels are the
   only forward-looking quantities and live in a separate frame. Signals computed on the
   close of bar *t* execute at the open of bar *t+1*. `tests/test_processor.py::
   test_features_have_no_look_ahead` guards this - keep it passing.
3. **Out-of-sample honesty.** Report metrics only on data the model never saw
   (chronological split with a `prediction_horizon` gap, or walk-forward folds). Never
   tune on the test window. The walk-forward Sharpe is the number that matters.
4. **Overfitting discipline.** Prefer fewer, regularised trees; use `TimeSeriesSplit` for
   any search; compare against buy-and-hold and against the previous iteration in
   `SYSTEM_LEARNINGS.md` before keeping a change.
5. **Git hygiene.** Commit after every working feature with a descriptive message. If a
   change breaks tests or degrades OOS metrics, `git revert`/`git checkout` it - do not
   leave the branch red. Never commit `venv/`, `data/*.csv`, `models/*.joblib`.
6. **No placeholders.** No `TODO`, `pass`-only functions, dummy returns or mocked
   business logic in `src/`. Everything must run end-to-end.
7. **Errors are handled, not hidden.** Catch specific exceptions, log with context via
   `src.logging_config.get_logger`, and surface HTTP errors with correct status codes.
8. **Risk rules are hard constraints.** Minimum reward:risk 1:2, stop-loss always
   ATR-based, position size always derived from `risk_per_trade_pct`, notional capped by
   `max_leverage`.

## 4. Coding conventions

* Python 3.12, type hints everywhere, `from __future__ import annotations`.
* Module-level docstring explaining purpose; dataclasses for structured results with
  `to_dict()` for JSON serialisation.
* Pure pandas/numpy indicator implementations (pandas_ta is used only as a test oracle).
* Config values come from `config.settings`; never hard-code symbols, paths or thresholds.
* Keep the API stateless apart from the explicit caches in `src/api/main.py`.
* Line length ≤ 120, imports sorted stdlib / third-party / local.

## 5. Self-improvement protocol (for autonomous iterations)

1. Read `SYSTEM_LEARNINGS.md` to know the current best OOS metrics and open hypotheses.
2. Pick **one** hypothesis (feature, label definition, model parameter, threshold, risk
   parameter). Implement it in `src/` only.
3. Run `pytest tests/`. Red → fix or revert.
4. Run `python -m src.models.train` then `python -m src.backtest.engine --mode walk_forward`.
5. Compare walk-forward Sharpe, max drawdown, win rate and profit factor with the current
   best. Better on Sharpe *without* worse drawdown → commit and record as new best.
   Otherwise revert the code change and record the negative result (negative results are
   valuable memory).
6. Append a dated entry to `SYSTEM_LEARNINGS.md` describing hypothesis, metrics, decision.
