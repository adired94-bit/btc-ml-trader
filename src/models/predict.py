"""Inference: load persisted models, predict direction probabilities and the
expected price range, and turn them into an actionable trading signal with a
fully sized risk plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.data.processor import CLASS_NAMES, DOWN, FLAT, UP, latest_feature_row
from src.logging_config import get_logger
from src.models.ensemble import DirectionEnsemble, RangeRegressor
from src.models.train import DIRECTION_MODEL_FILE, METADATA_FILE, RANGE_MODEL_FILE, load_metadata
from src.risk.management import RiskManager

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when inference is requested before any model artifact exists."""


@dataclass
class Prediction:
    timestamp: str
    close: float
    prob_down: float
    prob_flat: float
    prob_up: float
    predicted_class: str
    confidence: float
    expected_max_up_pct: float
    expected_max_down_pct: float
    expected_high: float
    expected_low: float
    horizon_bars: int
    components: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    action: str  # LONG / SHORT / WAIT
    reason: str
    probability: float
    threshold: float
    prediction: Prediction
    indicators: dict[str, float]
    trade_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prediction"] = self.prediction.to_dict()
        return data


class Predictor:
    """Lazy-loading wrapper around the persisted model artifacts."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir or settings.models_dir)
        self._direction: DirectionEnsemble | None = None
        self._range: RangeRegressor | None = None
        self._metadata: dict[str, Any] | None = None
        self._mtime: float | None = None

    # ------------------------------------------------------------------
    def artifacts_exist(self) -> bool:
        return (self.models_dir / DIRECTION_MODEL_FILE).exists() and (self.models_dir / RANGE_MODEL_FILE).exists()

    def _current_mtime(self) -> float:
        return max(
            (self.models_dir / f).stat().st_mtime for f in (DIRECTION_MODEL_FILE, RANGE_MODEL_FILE)
        )

    def load(self, force: bool = False) -> None:
        if not self.artifacts_exist():
            raise ModelNotTrainedError(
                f"No trained models in {self.models_dir}. Run `python -m src.models.train` first."
            )
        mtime = self._current_mtime()
        if force or self._direction is None or self._mtime != mtime:
            self._direction = DirectionEnsemble.load(self.models_dir / DIRECTION_MODEL_FILE)
            self._range = RangeRegressor.load(self.models_dir / RANGE_MODEL_FILE)
            self._metadata = load_metadata(self.models_dir)
            self._mtime = mtime
            logger.info("Loaded model artifacts from %s", self.models_dir)

    @property
    def direction(self) -> DirectionEnsemble:
        self.load()
        assert self._direction is not None
        return self._direction

    @property
    def range_model(self) -> RangeRegressor:
        self.load()
        assert self._range is not None
        return self._range

    @property
    def metadata(self) -> dict[str, Any]:
        self.load()
        return self._metadata or {}

    # ------------------------------------------------------------------
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.direction.predict_proba(features)

    def predict_range(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.range_model.predict(features)

    def predict_latest(self, df: pd.DataFrame) -> tuple[Prediction, pd.Series]:
        """Prediction for the last closed candle in ``df`` plus its indicator row."""
        features, indicators = latest_feature_row(df)
        if features.isna().any(axis=None):
            raise ValueError("Insufficient history to compute every feature (need >= 300 bars)")
        proba = self.predict_proba(features)[0]
        rng = self.predict_range(features).iloc[0]
        components = self.direction.predict_proba_components(features)
        ind_row = indicators.iloc[0]
        close = float(ind_row["close"])
        cls = int(proba.argmax())
        prediction = Prediction(
            timestamp=indicators.index[0].isoformat(),
            close=close,
            prob_down=float(proba[DOWN]),
            prob_flat=float(proba[FLAT]),
            prob_up=float(proba[UP]),
            predicted_class=CLASS_NAMES[cls],
            confidence=float(proba[cls]),
            expected_max_up_pct=float(rng["future_max_up"] * 100),
            expected_max_down_pct=float(rng["future_max_down"] * 100),
            expected_high=close * (1 + float(rng["future_max_up"])),
            expected_low=close * (1 + float(rng["future_max_down"])),
            horizon_bars=int(self.metadata.get("horizon", settings.prediction_horizon)),
            components={
                name: {CLASS_NAMES[i]: float(p[0][i]) for i in (DOWN, FLAT, UP)}
                for name, p in components.items()
            },
        )
        return prediction, ind_row


def decide_action(prob_up: float, prob_down: float, threshold: float) -> tuple[str, str]:
    if prob_up >= threshold and prob_up > prob_down:
        return "LONG", f"P(up)={prob_up:.2f} >= {threshold:.2f} and exceeds P(down)={prob_down:.2f}"
    if prob_down >= threshold and prob_down > prob_up:
        return "SHORT", f"P(down)={prob_down:.2f} >= {threshold:.2f} and exceeds P(up)={prob_up:.2f}"
    return "WAIT", f"No side reaches the {threshold:.2f} confidence threshold (up={prob_up:.2f}, down={prob_down:.2f})"


def generate_signal(
    df: pd.DataFrame,
    predictor: Predictor,
    risk_manager: RiskManager | None = None,
    threshold: float | None = None,
) -> Signal:
    """Full pipeline: features -> probabilities -> action -> sized trade plan."""
    threshold = threshold if threshold is not None else settings.signal_probability_threshold
    risk_manager = risk_manager or RiskManager()
    prediction, ind = predictor.predict_latest(df)
    action, reason = decide_action(prediction.prob_up, prediction.prob_down, threshold)

    indicator_keys = [
        "close", "ema_20", "ema_50", "ema_200", "rsi_14", "atr_14", "atr_pct", "vwap",
        "bb_upper", "bb_mid", "bb_lower", "bb_width", "macd", "macd_signal", "macd_hist",
        "vp_poc", "vp_va_low", "vp_va_high", "volume",
    ]
    indicators = {k: float(ind[k]) for k in indicator_keys if k in ind.index and pd.notna(ind[k])}

    plan = None
    if action in ("LONG", "SHORT"):
        plan = risk_manager.build_plan(action, prediction.close, float(ind["atr_14"])).to_dict()

    probability = prediction.prob_up if action == "LONG" else prediction.prob_down if action == "SHORT" else max(prediction.prob_up, prediction.prob_down)
    return Signal(
        action=action,
        reason=reason,
        probability=float(probability),
        threshold=threshold,
        prediction=prediction,
        indicators=indicators,
        trade_plan=plan,
    )
