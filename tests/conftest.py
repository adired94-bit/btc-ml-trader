"""Shared fixtures: deterministic synthetic OHLCV data and small, fast models.

No test touches the network or the production ``models/`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.processor import build_dataset  # noqa: E402
from src.models.ensemble import DirectionEnsemble, RangeRegressor  # noqa: E402
from src.models.train import DIRECTION_MODEL_FILE, METADATA_FILE, RANGE_MODEL_FILE  # noqa: E402

FAST_XGB = {"n_estimators": 40, "max_depth": 3}
FAST_LGBM = {"n_estimators": 40, "num_leaves": 7}
FAST_RF = {"n_estimators": 30, "max_depth": 5}


def make_ohlcv(n: int = 2000, seed: int = 7, start_price: float = 50_000.0) -> pd.DataFrame:
    """Geometric random walk with alternating drift regimes so models can learn something."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-0.0006, 0.0, 0.0008], size=n // 100 + 1), 100)[:n]
    returns = regime + rng.normal(0, 0.006, n)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.0005, n))
    wick_up = np.abs(rng.normal(0, 0.003, n))
    wick_dn = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + wick_up)
    low = np.minimum(open_, close) * (1 - wick_dn)
    volume = rng.lognormal(mean=5.5, sigma=0.4, size=n) * (1 + 3 * np.abs(returns) / 0.006)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture(scope="session")
def dataset(ohlcv):
    return build_dataset(ohlcv)


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory, dataset) -> Path:
    """Train tiny models once per session into a temporary directory."""
    X, y, _ = dataset
    models_dir = tmp_path_factory.mktemp("models")
    DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X, y["direction"]).save(models_dir / DIRECTION_MODEL_FILE)
    RangeRegressor(FAST_RF).fit(X, y[RangeRegressor.TARGETS]).save(models_dir / RANGE_MODEL_FILE)
    (models_dir / METADATA_FILE).write_text('{"horizon": 4, "feature_names": []}', encoding="utf-8")
    return models_dir
