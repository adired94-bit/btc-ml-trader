"""FastAPI REST server exposing market data, indicators, predictions, signals,
risk plans, backtests and model management.

Run with::

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from src import __version__
from src.backtest import engine as backtest_engine
from src.data import fetcher, storage
from src.data.processor import add_all_indicators, volume_profile
from src.logging_config import get_logger
from src.models import train as trainer
from src.models.predict import ModelNotTrainedError, Predictor, generate_signal
from src.risk.management import RiskError, RiskManager

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Shared state / caching
# ----------------------------------------------------------------------


class MarketState:
    """Thread-safe cache of OHLCV + indicator frames refreshed at most every ``ttl`` seconds."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._ohlcv: pd.DataFrame | None = None
        self._indicators: pd.DataFrame | None = None
        self._loaded_at = 0.0
        self._handle: fetcher.ExchangeHandle | None = None
        self.last_error: str | None = None

    def handle(self) -> fetcher.ExchangeHandle:
        if self._handle is None:
            self._handle = fetcher.connect()
        return self._handle

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            if not force and self._ohlcv is not None and time.time() - self._loaded_at < self.ttl:
                return
            try:
                df = storage.get_ohlcv()
                self._ohlcv = df
                self._indicators = add_all_indicators(df)
                self._loaded_at = time.time()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - keep serving stale data if the exchange hiccups
                self.last_error = str(exc)
                logger.error("Market refresh failed: %s", exc)
                if self._ohlcv is None:
                    raise

    def ohlcv(self) -> pd.DataFrame:
        self.refresh()
        assert self._ohlcv is not None
        return self._ohlcv

    def indicators(self) -> pd.DataFrame:
        self.refresh()
        assert self._indicators is not None
        return self._indicators

    @property
    def loaded_at(self) -> float:
        return self._loaded_at


class TrainingState:
    def __init__(self) -> None:
        self.running = False
        self.last_error: str | None = None
        self.last_report: dict[str, Any] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None


class BacktestCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, dict[str, Any]] = {}
        self.running = False
        self.last_error: str | None = None

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._results.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._results[key] = value


market = MarketState(ttl_seconds=60)
training = TrainingState()
backtests = BacktestCache()
predictor = Predictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("API starting (v%s) - warming market cache", __version__)
    try:
        market.refresh(force=True)
    except Exception as exc:  # noqa: BLE001 - the API must still start offline
        logger.error("Initial market load failed: %s", exc)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="BTC/USDT ML Signal API",
    version=__version__,
    description="Machine-learning driven Bitcoin direction forecasts, trading signals, risk plans and backtests.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------


class RiskPlanRequest(BaseModel):
    side: Literal["LONG", "SHORT"]
    entry_price: float = Field(gt=0)
    atr: float | None = Field(default=None, gt=0, description="Defaults to the latest ATR(14)")
    equity: float | None = Field(default=None, gt=0)
    risk_per_trade_pct: float | None = Field(default=None, gt=0, le=0.2)
    atr_multiplier: float | None = Field(default=None, gt=0)
    reward_risk_ratios: list[float] | None = None
    max_leverage: float | None = Field(default=None, gt=0)


class BacktestRequest(BaseModel):
    mode: Literal["holdout", "walk_forward"] = "holdout"
    threshold: float | None = Field(default=None, ge=0.34, le=0.99)
    take_profit_rr: float | None = Field(default=None, ge=1.0, le=10.0)
    atr_multiplier: float | None = Field(default=None, gt=0, le=10)
    risk_per_trade_pct: float | None = Field(default=None, gt=0, le=0.2)
    initial_equity: float | None = Field(default=None, gt=0)
    max_holding_bars: int | None = Field(default=None, ge=1, le=500)
    allow_short: bool = True
    train_window: int = Field(default=6_000, ge=500)
    test_window: int = Field(default=1_000, ge=50)


class TrainRequest(BaseModel):
    tune: bool = False
    n_iter: int = Field(default=12, ge=1, le=100)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    out.insert(0, "timestamp", [ts.isoformat() for ts in out.index])
    return out.where(pd.notna(out), None).to_dict(orient="records")


def _require_models() -> None:
    if not predictor.artifacts_exist():
        raise HTTPException(
            status_code=503,
            detail="Models not trained yet. POST /model/train or run `python -m src.models.train`.",
        )


# ----------------------------------------------------------------------
# Routes: health & market data
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "exchange": settings.exchange_id,
        "models_trained": predictor.artifacts_exist(),
        "training_running": training.running,
        "market_cache_age_s": round(time.time() - market.loaded_at, 1) if market.loaded_at else None,
        "market_error": market.last_error,
    }


@app.get("/market/ticker")
def ticker() -> dict[str, Any]:
    try:
        return fetcher.fetch_ticker(market.handle())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Ticker unavailable: {exc}") from exc


@app.get("/market/ohlcv")
def ohlcv(limit: int = Query(default=500, ge=10, le=20_000), refresh: bool = False) -> dict[str, Any]:
    try:
        if refresh:
            market.refresh(force=True)
        df = market.ohlcv().tail(limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc
    return {"symbol": settings.symbol, "timeframe": settings.timeframe, "count": len(df), "candles": _df_records(df)}


@app.get("/indicators")
def indicators(limit: int = Query(default=500, ge=10, le=20_000)) -> dict[str, Any]:
    try:
        df = market.indicators().tail(limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc
    return {"symbol": settings.symbol, "timeframe": settings.timeframe, "count": len(df), "rows": _df_records(df)}


@app.get("/indicators/latest")
def indicators_latest() -> dict[str, Any]:
    try:
        row = market.indicators().iloc[-1]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc
    payload = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    payload["timestamp"] = row.name.isoformat()
    return payload


@app.get("/indicators/volume-profile")
def volume_profile_endpoint(lookback: int = Query(default=240, ge=24, le=5_000), bins: int = Query(default=30, ge=5, le=200)) -> dict[str, Any]:
    try:
        df = market.ohlcv()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc
    vp = volume_profile(df, lookback=lookback, bins=bins)
    data = vp.to_dict()
    data.update({"lookback": lookback, "bins": bins, "as_of": df.index[-1].isoformat()})
    return data


# ----------------------------------------------------------------------
# Routes: prediction, signal, risk
# ----------------------------------------------------------------------


@app.get("/predict/latest")
def predict_latest() -> dict[str, Any]:
    _require_models()
    try:
        prediction, _ = predictor.predict_latest(market.ohlcv())
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return prediction.to_dict()


@app.get("/signal/latest")
def signal_latest(
    threshold: float | None = Query(default=None, ge=0.34, le=0.99),
    equity: float | None = Query(default=None, gt=0),
    risk_per_trade_pct: float | None = Query(default=None, gt=0, le=0.2),
    atr_multiplier: float | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    _require_models()
    try:
        rm = RiskManager(equity=equity, risk_per_trade_pct=risk_per_trade_pct, atr_multiplier=atr_multiplier)
        signal = generate_signal(market.ohlcv(), predictor, rm, threshold)
    except (ModelNotTrainedError, RiskError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {exc}") from exc
    return signal.to_dict()


@app.post("/risk/plan")
def risk_plan(req: RiskPlanRequest) -> dict[str, Any]:
    atr_value = req.atr
    if atr_value is None:
        try:
            atr_value = float(market.indicators()["atr_14"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"ATR unavailable: {exc}") from exc
    try:
        rm = RiskManager(
            equity=req.equity,
            risk_per_trade_pct=req.risk_per_trade_pct,
            atr_multiplier=req.atr_multiplier,
            reward_risk_ratios=req.reward_risk_ratios,
            max_leverage=req.max_leverage,
        )
        return rm.build_plan(req.side, req.entry_price, atr_value).to_dict()
    except RiskError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ----------------------------------------------------------------------
# Routes: model management
# ----------------------------------------------------------------------


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    meta = trainer.load_metadata()
    return {
        "trained": predictor.artifacts_exist(),
        "training_running": training.running,
        "training_started_at": training.started_at,
        "training_finished_at": training.finished_at,
        "training_error": training.last_error,
        "metadata": meta,
    }


def _train_job(tune: bool, n_iter: int) -> None:
    training.running = True
    training.started_at = pd.Timestamp.utcnow().isoformat()
    training.last_error = None
    try:
        report = trainer.train_models(market.ohlcv(), tune=tune, n_iter=n_iter)
        training.last_report = report.to_dict()
        predictor.load(force=True)
    except Exception as exc:  # noqa: BLE001
        training.last_error = str(exc)
        logger.exception("Training job failed")
    finally:
        training.running = False
        training.finished_at = pd.Timestamp.utcnow().isoformat()


@app.post("/model/train", status_code=202)
def train_model(req: TrainRequest, background: BackgroundTasks) -> dict[str, Any]:
    if training.running:
        raise HTTPException(status_code=409, detail="A training job is already running")
    background.add_task(_train_job, req.tune, req.n_iter)
    return {"status": "started", "tune": req.tune, "n_iter": req.n_iter}


# ----------------------------------------------------------------------
# Routes: backtesting
# ----------------------------------------------------------------------


def _backtest_key(req: BacktestRequest) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(req.model_dump().items()))


def _backtest_job(req: BacktestRequest, key: str) -> None:
    backtests.running = True
    backtests.last_error = None
    try:
        result = backtest_engine.run_backtest(
            df=market.ohlcv(),
            mode=req.mode,
            threshold=req.threshold,
            take_profit_rr=req.take_profit_rr,
            atr_multiplier=req.atr_multiplier,
            risk_per_trade_pct=req.risk_per_trade_pct,
            initial_equity=req.initial_equity,
            max_holding_bars=req.max_holding_bars,
            allow_short=req.allow_short,
            train_window=req.train_window,
            test_window=req.test_window,
        )
        backtests.put(key, result.to_dict())
    except Exception as exc:  # noqa: BLE001
        backtests.last_error = str(exc)
        logger.exception("Backtest job failed")
    finally:
        backtests.running = False


@app.post("/backtest/run", status_code=202)
def backtest_run(req: BacktestRequest, background: BackgroundTasks) -> dict[str, Any]:
    key = _backtest_key(req)
    cached = backtests.get(key)
    if cached is not None:
        return {"status": "ready", "key": key, "result": cached}
    if backtests.running:
        raise HTTPException(status_code=409, detail="A backtest is already running; poll /backtest/result")
    background.add_task(_backtest_job, req, key)
    return {"status": "started", "key": key}


@app.get("/backtest/result")
def backtest_result(key: str) -> dict[str, Any]:
    cached = backtests.get(key)
    if cached is None:
        if backtests.last_error:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {backtests.last_error}")
        return {"status": "running" if backtests.running else "unknown", "key": key}
    return {"status": "ready", "key": key, "result": cached}


@app.get("/backtest/latest")
def backtest_latest(mode: Literal["holdout", "walk_forward"] = "walk_forward") -> dict[str, Any]:
    """Most recent persisted CLI backtest (``python -m src.backtest.engine``)."""
    path = settings.models_dir / f"backtest_{mode}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No saved {mode} backtest; run `python -m src.backtest.engine --mode {mode}`")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
