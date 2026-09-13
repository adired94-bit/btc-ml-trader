# BTC/USDT ML Trading Signal Platform

Machine-learning driven Bitcoin (BTC/USDT, 1h) direction forecasts, trading signals with
ATR-based risk plans, walk-forward backtesting and a live dashboard.

```
Binance (CCXT) ─► CSV cache ─► indicators & features ─► XGBoost + LightGBM ensemble ─► signal + risk plan
                                                      └► RandomForest range model      ─► FastAPI ─► Streamlit
```

## Quick start (Windows)

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pytest tests/                 # 69 tests, no network needed
venv\Scripts\python.exe -m src.models.train              # downloads ~2y of candles, trains, saves models/
venv\Scripts\python.exe run.py                           # API on :8000 + dashboard on :8501
```

Or `powershell -ExecutionPolicy Bypass -File run.ps1`.

## Components

| Path | What it does |
|---|---|
| `config.py` | All settings (env / `.env` overridable). |
| `src/data/fetcher.py` | CCXT OHLCV + ticker with retries and exchange fallback (Binance → Bybit → OKX → Kraken). |
| `src/data/storage.py` | CSV cache, incremental refresh. |
| `src/data/processor.py` | RSI, EMA 20/50/200, ATR, Bollinger, VWAP, MACD, Volume Profile (POC / value area), 36 causal features, forward labels. |
| `src/models/ensemble.py` | `DirectionEnsemble` (XGBoost + LightGBM soft voting) and `RangeRegressor` (RandomForest max-up / max-down). |
| `src/models/train.py` | Chronological split with gap, optional `RandomizedSearchCV` over `TimeSeriesSplit`, OOS metrics, artifacts + `metadata.json`. |
| `src/models/predict.py` | Lazy model loading, latest prediction, LONG / SHORT / WAIT signal with sized trade plan. |
| `src/risk/management.py` | ATR stop, 1:2 and 1:3 take-profits, position sizing from % risk, leverage cap, Kelly / expectancy. |
| `src/backtest/engine.py` | Strategy backtest (holdout or rolling walk-forward): win rate, profit factor, Sharpe, Sortino, max drawdown, equity curve. |
| `src/backtest/walk_forward.py` | Day-by-day evaluation (train ≤ T, predict T+1): hit rate, MAE / RMSE of target price, P&L, failure-pattern analysis and self-improvement loop that appends to `SYSTEM_LEARNINGS.md`. |
| `src/api/main.py` | REST API (see below). |
| `app.py` | Streamlit dashboard: live chart, signal & risk plan, backtests, model info. |
| `autoloop.sh` | Infinite `claude --dangerously-skip-permissions` improvement loop following `CLAUDE.md`. |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, model availability, cache age |
| GET | `/market/ticker` | Live ticker |
| GET | `/market/ohlcv?limit=` | Cached candles |
| GET | `/indicators`, `/indicators/latest` | Indicator rows |
| GET | `/indicators/volume-profile?lookback=&bins=` | Volume profile with POC / value area |
| GET | `/predict/latest` | Direction probabilities + expected range |
| GET | `/signal/latest?threshold=&equity=&risk_per_trade_pct=&atr_multiplier=` | Actionable signal with trade plan |
| POST | `/risk/plan` | Stop / take-profit / size for a given side, entry and ATR |
| GET | `/model/info` · POST `/model/train` | Training metadata / background retrain |
| POST | `/backtest/run` · GET `/backtest/result?key=` · GET `/backtest/latest?mode=` | Backtests |

Interactive docs: `http://127.0.0.1:8000/docs`.

## Evaluation commands

```bash
venv\Scripts\python.exe -m src.backtest.engine --mode walk_forward
venv\Scripts\python.exe -m src.backtest.walk_forward          # daily walk-forward + self-learning
```

Results and learnings accumulate in `SYSTEM_LEARNINGS.md`; raw JSON lands in `models/`.

## Cloud / work from any device

See [DEPLOY.md](DEPLOY.md). Short version: push to GitHub (`publish.ps1`), then deploy `app.py` on
Streamlit Community Cloud. The dashboard runs in *embedded* mode there (no separate API process);
models and two years of data ship in the repo so the app starts trained.

## Disclaimer

Research software. Backtested performance is not indicative of future results; nothing here is
financial advice.
