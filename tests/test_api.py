"""API tests using FastAPI's TestClient with the exchange fully stubbed out."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.data.processor import add_all_indicators
from src.models.predict import Predictor


@pytest.fixture
def client(monkeypatch, ohlcv, trained_models_dir):
    # No network: seed the market cache with synthetic data and make refresh a no-op.
    monkeypatch.setattr(api.storage, "get_ohlcv", lambda *a, **k: ohlcv)
    api.market._ohlcv = ohlcv
    api.market._indicators = add_all_indicators(ohlcv)
    api.market._loaded_at = time.time()
    api.market.ttl = 10_000
    monkeypatch.setattr(api, "predictor", Predictor(models_dir=trained_models_dir))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models_trained"] is True


def test_ohlcv_and_indicators(client):
    r = client.get("/market/ohlcv", params={"limit": 50}).json()
    assert r["count"] == 50 and {"timestamp", "open", "close"} <= set(r["candles"][0])
    latest = client.get("/indicators/latest").json()
    assert "rsi_14" in latest and "timestamp" in latest
    rows = client.get("/indicators", params={"limit": 20}).json()
    assert rows["count"] == 20


def test_volume_profile_endpoint(client):
    vp = client.get("/indicators/volume-profile", params={"lookback": 120, "bins": 12}).json()
    assert len(vp["levels"]) == 12 and vp["value_area_low"] <= vp["poc"] <= vp["value_area_high"]


def test_prediction_and_signal(client):
    pred = client.get("/predict/latest").json()
    assert pytest.approx(pred["prob_down"] + pred["prob_flat"] + pred["prob_up"], abs=1e-6) == 1.0
    sig = client.get("/signal/latest", params={"threshold": 0.34, "equity": 2_000}).json()
    assert sig["action"] in {"LONG", "SHORT", "WAIT"}
    if sig["trade_plan"]:
        assert sig["trade_plan"]["equity"] == 2_000


def test_risk_plan_endpoint(client):
    r = client.post("/risk/plan", json={"side": "LONG", "entry_price": 50_000, "atr": 500, "equity": 10_000})
    assert r.status_code == 200
    plan = r.json()
    assert plan["stop_loss"] == pytest.approx(50_000 - 750)
    assert plan["take_profits"][0]["reward_risk"] == 2.0
    # ATR defaults to the latest indicator value when omitted
    assert client.post("/risk/plan", json={"side": "SHORT", "entry_price": 50_000}).status_code == 200
    # Validation errors surface as 422
    assert client.post("/risk/plan", json={"side": "LONG", "entry_price": -1}).status_code == 422


def test_model_info_and_backtest_status(client):
    info = client.get("/model/info").json()
    assert info["trained"] is True
    r = client.get("/backtest/result", params={"key": "nope"}).json()
    assert r["status"] in {"unknown", "running"}


def test_untrained_models_return_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "predictor", Predictor(models_dir=tmp_path))
    assert client.get("/signal/latest").status_code == 503
    assert client.get("/predict/latest").status_code == 503
