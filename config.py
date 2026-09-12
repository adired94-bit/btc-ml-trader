"""Central configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings.

    Every value can be overridden through an environment variable with the
    same name (case-insensitive) or through a ``.env`` file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Market data -------------------------------------------------------
    exchange_id: str = Field(default="binance", description="Primary CCXT exchange id")
    fallback_exchanges: list[str] = Field(
        default_factory=lambda: ["bybit", "okx", "kraken"],
        description="Exchanges tried in order when the primary one fails",
    )
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")
    history_candles: int = Field(default=17_520, description="~2 years of hourly candles")
    request_timeout_ms: int = Field(default=20_000)
    max_retries: int = Field(default=4)

    # --- Feature engineering / model -------------------------------------
    prediction_horizon: int = Field(default=4, description="Bars ahead the model predicts")
    direction_threshold_pct: float = Field(
        default=0.0015,
        description="Return magnitude (fraction) that separates UP / DOWN from FLAT",
    )
    direction_model: Literal["xgboost", "lightgbm"] = Field(default="xgboost")
    train_test_split: float = Field(default=0.8, description="Chronological train fraction")
    random_state: int = Field(default=42)

    # --- Risk management ---------------------------------------------------
    account_equity: float = Field(default=10_000.0)
    risk_per_trade_pct: float = Field(default=0.01, description="Fraction of equity risked per trade")
    atr_stop_multiplier: float = Field(default=1.5)
    reward_risk_ratios: list[float] = Field(default_factory=lambda: [2.0, 3.0])
    max_leverage: float = Field(default=3.0)
    signal_probability_threshold: float = Field(default=0.55)

    # --- Backtesting -------------------------------------------------------
    backtest_fee_pct: float = Field(default=0.0004, description="Taker fee per side")
    backtest_slippage_pct: float = Field(default=0.0002)
    backtest_max_holding_bars: int = Field(default=24)

    # --- Server ------------------------------------------------------------
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    api_url: str = Field(default="http://127.0.0.1:8000")
    dashboard_port: int = Field(default=8501)
    log_level: str = Field(default="INFO")

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.models_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def symbol_slug(self) -> str:
        return self.symbol.replace("/", "_").replace(":", "_")

    @property
    def ohlcv_cache_path(self) -> Path:
        return self.data_dir / f"{self.symbol_slug}_{self.timeframe}.csv"


settings = Settings()
settings.ensure_dirs()
