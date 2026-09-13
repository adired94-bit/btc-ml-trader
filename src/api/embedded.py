"""In-process client with the same surface as the HTTP API.

Lets the Streamlit dashboard run as a single process (Streamlit Community Cloud,
Hugging Face Spaces, a laptop without the API running) by calling the route
functions of ``src.api.main`` directly instead of over HTTP.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import HTTPException

from src.api import main as api


class EmbeddedClient:
    """``get(path, params)`` / ``post(path, payload)`` mirroring the REST routes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        p = params or {}
        with self._lock:
            if path == "/health":
                return api.health()
            if path == "/market/ticker":
                return api.ticker()
            if path == "/market/ohlcv":
                return api.ohlcv(limit=int(p.get("limit", 500)), refresh=bool(p.get("refresh", False)))
            if path == "/indicators":
                return api.indicators(limit=int(p.get("limit", 500)))
            if path == "/indicators/latest":
                return api.indicators_latest()
            if path == "/indicators/volume-profile":
                return api.volume_profile_endpoint(lookback=int(p.get("lookback", 240)), bins=int(p.get("bins", 30)))
            if path == "/predict/latest":
                return api.predict_latest()
            if path == "/signal/latest":
                return api.signal_latest(
                    threshold=p.get("threshold"), equity=p.get("equity"),
                    risk_per_trade_pct=p.get("risk_per_trade_pct"), atr_multiplier=p.get("atr_multiplier"),
                )
            if path == "/model/info":
                return api.model_info()
            if path == "/backtest/result":
                return api.backtest_result(key=str(p.get("key", "")))
            if path == "/backtest/latest":
                return api.backtest_latest(mode=p.get("mode", "walk_forward"))
        raise HTTPException(status_code=404, detail=f"Unknown path {path}")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/risk/plan":
            return api.risk_plan(api.RiskPlanRequest(**payload))
        if path == "/model/train":
            req = api.TrainRequest(**payload)
            if api.training.running:
                raise HTTPException(status_code=409, detail="A training job is already running")
            # Flag before the thread starts so a rerun that arrives first sees "running", not "unknown".
            api.training.running = True
            threading.Thread(target=api._train_job, args=(req.tune, req.n_iter), daemon=True).start()
            return {"status": "started", "tune": req.tune, "n_iter": req.n_iter}
        if path == "/backtest/run":
            req = api.BacktestRequest(**payload)
            key = api._backtest_key(req)
            cached = api.backtests.get(key)
            if cached is not None:
                return {"status": "ready", "key": key, "result": cached}
            if api.backtests.running:
                raise HTTPException(status_code=409, detail="A backtest is already running")
            api.backtests.running = True
            threading.Thread(target=api._backtest_job, args=(req, key), daemon=True).start()
            return {"status": "started", "key": key}
        raise HTTPException(status_code=404, detail=f"Unknown path {path}")


_client: EmbeddedClient | None = None


def get_client() -> EmbeddedClient:
    global _client
    if _client is None:
        _client = EmbeddedClient()
    return _client
