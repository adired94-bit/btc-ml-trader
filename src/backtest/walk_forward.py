"""Strict day-by-day walk-forward evaluation with a self-learning feedback loop.

Procedure (see ``run_walk_forward``):

1. The evaluation window runs from ``eval_start_days_ago`` (default 365) to
   ``eval_end_days_ago`` (default 90) relative to the last available candle.
2. For every UTC day *T* in that window:
   a. The models are (re)trained **only** on hourly bars whose 24-bar-ahead label
      is already known at the close of day *T* (zero look-ahead). Retraining
      happens every ``retrain_every_days`` days; in between, the most recent
      model - which has still only seen data ``<= T`` - is reused.
   b. At the last closed hourly bar of day *T* the models predict day *T+1*:
      direction (UP/DOWN with probability), a target close price and a risk
      level derived from the trailing volatility distribution.
   c. The prediction is compared with the realised close of day *T+1*.
   d. A ``DayRecord`` stores Hit/Miss, the error magnitude in %, the regime
      flags active on day *T* and the simulated trade outcome.
3. Metrics: directional accuracy (all days and traded days), MAE / RMSE on the
   target price (USD and %), simulated P&L with ATR stops and 1:2 take-profits,
   Sharpe, max drawdown, win rate, profit factor.

Self-learning (see ``self_improve``): the misses are analysed by regime
(volatility spikes, low volume, trend reversals, ...). Candidate adjustments -
stronger regularisation, regime-aware sample weights and regime trade filters -
are evaluated on the first half of the window and validated on the second half.
The winning configuration is re-run over the full window and everything is
appended to ``SYSTEM_LEARNINGS.md``.

CLI::

    python -m src.backtest.walk_forward             # full run + self-improvement
    python -m src.backtest.walk_forward --fast      # fewer trees, quicker
    python -m src.backtest.walk_forward --no-improve
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROJECT_ROOT, settings
from src.data import storage
from src.data.processor import DOWN, UP, add_all_indicators, build_extended_features, build_features
from src.logging_config import get_logger
from src.models.ensemble import DirectionEnsemble
from src.risk.management import RiskManager

logger = get_logger(__name__)

BARS_PER_DAY = 24
REGIME_FLAGS = ("high_volatility", "low_volatility", "low_volume", "volume_spike", "trend_up", "trend_down", "overbought", "oversold")
OUTCOME_FLAGS = ("trend_reversal", "large_move")

# Lighter boosters than production: the loop refits dozens of times.
WF_XGB_PARAMS: dict[str, Any] = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05}
WF_LGBM_PARAMS: dict[str, Any] = {"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.05}
WF_REG_PARAMS: dict[str, Any] = {
    "n_estimators": 200, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "min_child_samples": 30, "reg_lambda": 2.0, "verbose": -1, "n_jobs": -1,
    "random_state": settings.random_state,
}


# ----------------------------------------------------------------------
# Configuration & records
# ----------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    eval_start_days_ago: int = 365
    eval_end_days_ago: int = 90
    horizon_bars: int = BARS_PER_DAY
    retrain_every_days: int = 7
    min_train_rows: int = 1_000
    threshold: float = 0.55
    atr_multiplier: float = 1.5
    take_profit_rr: float = 2.0
    risk_per_trade_pct: float = 0.01
    initial_equity: float = 10_000.0
    fee_pct: float = settings.backtest_fee_pct
    slippage_pct: float = settings.backtest_slippage_pct
    xgb_params: dict[str, Any] = field(default_factory=lambda: dict(WF_XGB_PARAMS))
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(WF_LGBM_PARAMS))
    reg_params: dict[str, Any] = field(default_factory=lambda: dict(WF_REG_PARAMS))
    regime_weights: dict[str, float] = field(default_factory=dict)  # extra sample weight per regime flag
    skip_regimes: list[str] = field(default_factory=list)  # ex-ante regimes in which no trade is taken
    feature_set: str = "base"  # "base" (36 features) or "extended" (+ multi-day context)
    model: str = "ensemble"  # "ensemble" (XGB+LGBM) or "logreg" (regularised logistic regression)
    logreg_c: float = 0.05
    label: str = "baseline"

    def copy(self, **changes: Any) -> "WalkForwardConfig":
        data = asdict(self)
        data.update(changes)
        return WalkForwardConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DayRecord:
    day: str  # prediction made at the close of this day
    target_day: str  # the day being predicted
    train_end: str  # last bar the model was trained on
    close: float
    actual_close: float
    actual_return_pct: float
    predicted_direction: str
    actual_direction: str
    hit: bool
    prob_up: float
    prob_down: float
    confidence: float
    traded: bool
    target_price: float
    predicted_return_pct: float
    error_pct: float  # (target - actual) / actual * 100
    abs_error_pct: float
    risk_level: str
    atr_pct: float
    volume_ratio: float
    regimes: list[str]
    trade_side: str  # LONG / SHORT / NONE
    trade_pnl: float
    trade_return_pct: float
    trade_exit: str  # STOP / TAKE_PROFIT / CLOSE / NONE
    equity: float


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    records: list[DayRecord]
    metrics: dict[str, Any]
    failure_analysis: dict[str, Any]
    retrains: int
    runtime_seconds: float

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self.records])
        if not df.empty:
            df["day"] = pd.to_datetime(df["day"])
            df = df.set_index("day")
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "failure_analysis": self.failure_analysis,
            "retrains": self.retrains,
            "runtime_seconds": self.runtime_seconds,
            "records": [asdict(r) for r in self.records],
        }


# ----------------------------------------------------------------------
# Data preparation (all causal)
# ----------------------------------------------------------------------


@dataclass
class PreparedData:
    ohlcv: pd.DataFrame
    indicators: pd.DataFrame
    features: pd.DataFrame
    future_return: pd.Series  # close[t+h] / close[t] - 1
    direction: pd.Series  # UP / DOWN encoded 2 / 0 (never FLAT for the daily task)
    regimes: pd.DataFrame  # boolean flags per bar, computed from trailing data only
    daily_atr: pd.Series  # ATR(14) on daily bars, aligned to hourly index (ffilled)


def compute_regimes(indicators: pd.DataFrame, lookback: int = 24 * 90) -> pd.DataFrame:
    """Boolean regime flags per bar using only trailing information."""
    atr_pct = indicators["atr_pct"]
    vol_ratio = indicators["volume"] / indicators["volume"].rolling(24, min_periods=24).mean()
    # trailing percentile rank of ATR% within the previous ``lookback`` bars
    atr_rank = atr_pct.rolling(lookback, min_periods=24 * 20).rank(pct=True)
    flags = pd.DataFrame(index=indicators.index)
    flags["high_volatility"] = atr_rank >= 0.8
    flags["low_volatility"] = atr_rank <= 0.2
    flags["low_volume"] = vol_ratio < 0.7
    flags["volume_spike"] = vol_ratio > 2.0
    flags["trend_up"] = (indicators["ema_20"] > indicators["ema_50"]) & (indicators["ema_50"] > indicators["ema_200"])
    flags["trend_down"] = (indicators["ema_20"] < indicators["ema_50"]) & (indicators["ema_50"] < indicators["ema_200"])
    flags["overbought"] = indicators["rsi_14"] > 70
    flags["oversold"] = indicators["rsi_14"] < 30
    return flags.fillna(False).astype(bool)


def risk_level_from_rank(rank: float) -> str:
    if not np.isfinite(rank):
        return "MEDIUM"
    if rank >= 0.8:
        return "HIGH"
    if rank <= 0.3:
        return "LOW"
    return "MEDIUM"


def prepare_data(ohlcv: pd.DataFrame, horizon_bars: int = BARS_PER_DAY, feature_set: str = "base") -> PreparedData:
    indicators = add_all_indicators(ohlcv)
    features = build_extended_features(ohlcv, indicators) if feature_set == "extended" else build_features(ohlcv, indicators)
    close = ohlcv["close"]
    future_return = close.shift(-horizon_bars) / close - 1.0
    direction = pd.Series(np.where(future_return >= 0, UP, DOWN), index=ohlcv.index)
    direction[future_return.isna()] = -1
    regimes = compute_regimes(indicators)
    daily = ohlcv.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev_close = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    daily_atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    # A day's ATR becomes known at that day's close -> shift so bar t sees only completed days.
    daily_atr_hourly = daily_atr.reindex(ohlcv.index, method="ffill")
    return PreparedData(ohlcv, indicators, features, future_return, direction, regimes, daily_atr_hourly)


def day_close_bars(index: pd.DatetimeIndex) -> pd.Series:
    """Map each UTC day -> timestamp of its last hourly bar (bar open time)."""
    days = index.tz_convert("UTC").floor("D") if index.tz is not None else index.floor("D")
    return pd.Series(index, index=days).groupby(level=0).last()


# ----------------------------------------------------------------------
# Models used inside the loop
# ----------------------------------------------------------------------


class DailyModels:
    """Binary direction ensemble + return regressor trained on bars <= train_end."""

    def __init__(self, cfg: WalkForwardConfig) -> None:
        self.cfg = cfg
        if cfg.model == "logreg":
            self.direction = make_pipeline(
                StandardScaler(), LogisticRegression(C=cfg.logreg_c, max_iter=2_000, class_weight="balanced")
            )
        elif cfg.model == "ensemble":
            self.direction = DirectionEnsemble(xgb_params=cfg.xgb_params, lgbm_params=cfg.lgbm_params)
        else:
            raise ValueError(f"Unknown model: {cfg.model}")
        self.feature_names: list[str] = []
        self.regressor = lgb.LGBMRegressor(**cfg.reg_params)
        self.train_end: pd.Timestamp | None = None
        self.n_rows = 0

    def fit(self, data: PreparedData, train_end: pd.Timestamp) -> "DailyModels":
        mask = (data.features.index <= train_end) & (data.direction >= 0).to_numpy()
        X = data.features[mask].dropna()
        y_dir = data.direction.loc[X.index]
        y_ret = data.future_return.loc[X.index]
        if len(X) < self.cfg.min_train_rows:
            raise ValueError(f"Only {len(X)} training rows available at {train_end}; need {self.cfg.min_train_rows}")
        weights = np.ones(len(X))
        for flag, extra in self.cfg.regime_weights.items():
            if flag in data.regimes.columns and extra > 0:
                weights = weights * np.where(data.regimes.loc[X.index, flag].to_numpy(), 1.0 + extra, 1.0)
        self.feature_names = list(X.columns)
        if self.cfg.model == "logreg":
            self.direction.fit(X.to_numpy(dtype=np.float32), y_dir.to_numpy(), logisticregression__sample_weight=weights)
        else:
            self.direction.fit(X, y_dir, sample_weight=weights)
        self.regressor.fit(X.to_numpy(dtype=np.float32), y_ret.to_numpy(dtype=np.float32), sample_weight=weights)
        self.train_end = train_end
        self.n_rows = len(X)
        return self

    def predict(self, features_row: pd.DataFrame) -> tuple[float, float, float]:
        arr = features_row[self.feature_names].to_numpy(dtype=np.float32)
        if self.cfg.model == "logreg":
            proba = self.direction.predict_proba(arr)[0]
            classes = list(self.direction.classes_)
            p_up = float(proba[classes.index(UP)]) if UP in classes else 0.0
            p_down = float(proba[classes.index(DOWN)]) if DOWN in classes else 0.0
        else:
            proba = self.direction.predict_proba(features_row)[0]
            p_up, p_down = float(proba[UP]), float(proba[DOWN])
        ret = float(self.regressor.predict(arr)[0])
        return p_up, p_down, ret


# ----------------------------------------------------------------------
# Trade simulation for one day
# ----------------------------------------------------------------------


def simulate_day_trade(
    side: str, bars: pd.DataFrame, atr: float, equity: float, cfg: WalkForwardConfig
) -> tuple[float, float, str, float, float, float]:
    """Trade day T+1 hour by hour. Returns (pnl, return_pct, exit_reason, entry, stop, tp)."""
    rm = RiskManager(
        equity=equity, risk_per_trade_pct=cfg.risk_per_trade_pct, atr_multiplier=cfg.atr_multiplier,
        reward_risk_ratios=[cfg.take_profit_rr], max_leverage=settings.max_leverage,
    )
    raw_entry = float(bars["open"].iloc[0])
    entry = raw_entry * (1 + cfg.slippage_pct) if side == "LONG" else raw_entry * (1 - cfg.slippage_pct)
    plan = rm.build_plan(side, entry, atr)
    stop, tp, size = plan.stop_loss, plan.take_profits[0].price, plan.position_size
    exit_price, reason = None, "CLOSE"
    for _, bar in bars.iterrows():
        if side == "LONG":
            if bar["low"] <= stop:
                exit_price, reason = stop, "STOP"
                break
            if bar["high"] >= tp:
                exit_price, reason = tp, "TAKE_PROFIT"
                break
        else:
            if bar["high"] >= stop:
                exit_price, reason = stop, "STOP"
                break
            if bar["low"] <= tp:
                exit_price, reason = tp, "TAKE_PROFIT"
                break
    if exit_price is None:
        last = float(bars["close"].iloc[-1])
        exit_price = last * (1 - cfg.slippage_pct) if side == "LONG" else last * (1 + cfg.slippage_pct)
    gross = (exit_price - entry) * size if side == "LONG" else (entry - exit_price) * size
    pnl = gross - cfg.fee_pct * size * (entry + exit_price)
    return float(pnl), float(pnl / equity * 100), reason, entry, stop, tp


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def evaluation_days(index: pd.DatetimeIndex, cfg: WalkForwardConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    last = index[-1]
    start = (last - pd.Timedelta(days=cfg.eval_start_days_ago)).floor("D")
    end = (last - pd.Timedelta(days=cfg.eval_end_days_ago)).floor("D")
    if end <= start:
        raise ValueError("eval_end_days_ago must be smaller than eval_start_days_ago")
    return start, end


def run_walk_forward(
    ohlcv: pd.DataFrame, cfg: WalkForwardConfig | None = None, data: PreparedData | None = None, verbose: bool = True
) -> WalkForwardResult:
    cfg = cfg or WalkForwardConfig()
    t0 = time.perf_counter()
    data = data or prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    idx = ohlcv.index
    start, end = evaluation_days(idx, cfg)
    closes = day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    if not days:
        raise ValueError("No evaluation days inside the requested window")

    models: DailyModels | None = None
    last_train_day: pd.Timestamp | None = None
    retrains = 0
    equity = cfg.initial_equity
    records: list[DayRecord] = []
    atr_rank = data.indicators["atr_pct"].rolling(24 * 90, min_periods=24 * 20).rank(pct=True)
    pos = pd.Series(np.arange(len(idx)), index=idx)

    for day in days:
        ts = closes[day]  # last hourly bar of day T (its open time)
        i = int(pos[ts])
        if i + cfg.horizon_bars >= len(idx):
            break  # day T+1 not fully realised yet
        # Labels are known for bars whose horizon has elapsed by the close of day T.
        train_end = idx[i - cfg.horizon_bars]
        if models is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
            models = DailyModels(cfg).fit(data, train_end)
            last_train_day = day
            retrains += 1
            if verbose:
                logger.info("Retrained at %s on %d rows (train_end %s)", day.date(), models.n_rows, train_end)
        row = data.features.iloc[[i]]
        if row.isna().any(axis=None):
            continue
        p_up, p_down, r_hat = models.predict(row)
        close = float(data.ohlcv["close"].iloc[i])
        actual_close = float(data.ohlcv["close"].iloc[i + cfg.horizon_bars])
        actual_ret = (actual_close / close - 1) * 100
        pred_dir = "UP" if p_up >= p_down else "DOWN"
        actual_dir = "UP" if actual_ret >= 0 else "DOWN"
        confidence = max(p_up, p_down)
        target = close * (1 + r_hat)
        regimes_now = [f for f in REGIME_FLAGS if bool(data.regimes.iloc[i][f])]
        prev_day_ret = float(data.ohlcv["close"].iloc[i] / data.ohlcv["close"].iloc[max(0, i - cfg.horizon_bars)] - 1)
        outcome_flags = []
        if np.sign(prev_day_ret) != np.sign(actual_ret) and abs(prev_day_ret) > 0.005:
            outcome_flags.append("trend_reversal")
        if abs(actual_ret) > 3.0:
            outcome_flags.append("large_move")

        traded = confidence >= cfg.threshold and not any(f in cfg.skip_regimes for f in regimes_now)
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        daily_atr = float(data.daily_atr.iloc[i]) if np.isfinite(data.daily_atr.iloc[i]) else float("nan")
        if traded and np.isfinite(daily_atr) and daily_atr > 0:
            side = "LONG" if pred_dir == "UP" else "SHORT"
            bars = data.ohlcv.iloc[i + 1: i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = simulate_day_trade(side, bars, daily_atr, equity, cfg)
            equity += pnl
        else:
            traded = False

        records.append(
            DayRecord(
                day=day.isoformat(), target_day=(day + pd.Timedelta(hours=cfg.horizon_bars)).isoformat(), train_end=train_end.isoformat(),
                close=close, actual_close=actual_close, actual_return_pct=float(actual_ret),
                predicted_direction=pred_dir, actual_direction=actual_dir, hit=pred_dir == actual_dir,
                prob_up=p_up, prob_down=p_down, confidence=float(confidence), traded=traded,
                target_price=float(target), predicted_return_pct=float(r_hat * 100),
                error_pct=float((target - actual_close) / actual_close * 100), abs_error_pct=float(abs(target - actual_close) / actual_close * 100),
                risk_level=risk_level_from_rank(float(atr_rank.iloc[i])), atr_pct=float(data.indicators["atr_pct"].iloc[i]),
                volume_ratio=float(data.indicators["volume"].iloc[i] / max(data.indicators["volume"].iloc[max(0, i - 24):i].mean(), 1e-9)),
                regimes=regimes_now + outcome_flags, trade_side=side, trade_pnl=float(pnl), trade_return_pct=float(ret_pct),
                trade_exit=exit_reason, equity=float(equity),
            )
        )

    if not records:
        raise ValueError("Walk-forward produced no records (not enough realised data in the window)")
    metrics = compute_metrics(records, cfg)
    analysis = analyze_misses(records)
    runtime = time.perf_counter() - t0
    if verbose:
        logger.info(
            "Walk-forward [%s]: %d days, hit rate %.1f%%, traded hit rate %.1f%%, MAE %.2f%%, P&L %+.2f%%, Sharpe %.2f (%.0fs)",
            cfg.label, metrics["days"], metrics["directional_accuracy"] * 100, metrics["traded_directional_accuracy"] * 100,
            metrics["mae_pct"], metrics["total_return_pct"], metrics["sharpe_ratio"], runtime,
        )
    return WalkForwardResult(cfg, records, metrics, analysis, retrains, runtime)


# ----------------------------------------------------------------------
# Metrics & failure analysis
# ----------------------------------------------------------------------


def compute_metrics(records: list[DayRecord], cfg: WalkForwardConfig) -> dict[str, Any]:
    df = pd.DataFrame([asdict(r) for r in records])
    traded = df[df["traded"]]
    err_usd = df["target_price"] - df["actual_close"]
    daily_ret = df["trade_return_pct"] / 100
    sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if daily_ret.std() > 0 else 0.0
    equity = pd.concat([pd.Series([cfg.initial_equity]), df["equity"]]).reset_index(drop=True)
    max_dd = float(-(equity / equity.cummax() - 1).min() * 100)
    wins = traded[traded["trade_pnl"] > 0]
    losses = traded[traded["trade_pnl"] <= 0]
    gross_profit, gross_loss = float(wins["trade_pnl"].sum()), float(-losses["trade_pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return {
        "days": int(len(df)),
        "start": df["day"].iloc[0],
        "end": df["day"].iloc[-1],
        "directional_accuracy": float(df["hit"].mean()),
        "traded_days": int(len(traded)),
        "coverage": float(len(traded) / len(df)),
        "traded_directional_accuracy": float(traded["hit"].mean()) if len(traded) else 0.0,
        "up_predictions": int((df["predicted_direction"] == "UP").sum()),
        "down_predictions": int((df["predicted_direction"] == "DOWN").sum()),
        "actual_up_days": int((df["actual_direction"] == "UP").sum()),
        "mae_usd": float(err_usd.abs().mean()),
        "rmse_usd": float(np.sqrt((err_usd ** 2).mean())),
        "mae_pct": float(df["abs_error_pct"].mean()),
        "rmse_pct": float(np.sqrt((df["error_pct"] ** 2).mean())),
        "median_abs_error_pct": float(df["abs_error_pct"].median()),
        "naive_mae_pct": float(df["actual_return_pct"].abs().mean()),  # "tomorrow = today" benchmark
        "total_pnl": float(traded["trade_pnl"].sum()),
        "total_return_pct": float((df["equity"].iloc[-1] / cfg.initial_equity - 1) * 100),
        "final_equity": float(df["equity"].iloc[-1]),
        "buy_and_hold_return_pct": float((df["actual_close"].iloc[-1] / df["close"].iloc[0] - 1) * 100),
        "win_rate": float((traded["trade_pnl"] > 0).mean()) if len(traded) else 0.0,
        "profit_factor": float(pf),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "avg_trade_return_pct": float(traded["trade_return_pct"].mean()) if len(traded) else 0.0,
        "exit_reasons": {k: int(v) for k, v in traded["trade_exit"].value_counts().items()},
        "risk_level_distribution": {k: int(v) for k, v in df["risk_level"].value_counts().items()},
    }


def analyze_misses(records: list[DayRecord]) -> dict[str, Any]:
    """Compare hit rates across regimes to expose where the model fails."""
    df = pd.DataFrame([asdict(r) for r in records])
    overall_miss = float(1 - df["hit"].mean())
    patterns: list[dict[str, Any]] = []
    for flag in REGIME_FLAGS + OUTCOME_FLAGS:
        mask = df["regimes"].apply(lambda regs, f=flag: f in regs)
        n = int(mask.sum())
        if n == 0:
            continue
        miss_rate = float(1 - df.loc[mask, "hit"].mean())
        patterns.append({
            "pattern": flag, "days": n, "share_of_days": float(n / len(df)), "miss_rate": miss_rate,
            "lift_vs_overall": float(miss_rate - overall_miss), "ex_ante": flag in REGIME_FLAGS,
            "mean_abs_error_pct": float(df.loc[mask, "abs_error_pct"].mean()),
        })
    patterns.sort(key=lambda p: p["lift_vs_overall"], reverse=True)
    worst = [p for p in patterns if p["days"] >= 10 and p["lift_vs_overall"] > 0.03]
    for level in ("LOW", "MEDIUM", "HIGH"):
        sub = df[df["risk_level"] == level]
        if len(sub):
            patterns.append({"pattern": f"risk_{level.lower()}", "days": int(len(sub)), "share_of_days": float(len(sub) / len(df)),
                             "miss_rate": float(1 - sub["hit"].mean()), "lift_vs_overall": float(1 - sub["hit"].mean() - overall_miss),
                             "ex_ante": True, "mean_abs_error_pct": float(sub["abs_error_pct"].mean())})
    misses, hits = df[~df["hit"]], df[df["hit"]]
    return {
        "overall_miss_rate": overall_miss,
        "misses": int(len(misses)),
        "hits": int(len(hits)),
        "mean_abs_error_pct_on_misses": float(misses["abs_error_pct"].mean()) if len(misses) else 0.0,
        "mean_abs_error_pct_on_hits": float(hits["abs_error_pct"].mean()) if len(hits) else 0.0,
        "mean_atr_pct_on_misses": float(misses["atr_pct"].mean()) if len(misses) else 0.0,
        "mean_atr_pct_on_hits": float(hits["atr_pct"].mean()) if len(hits) else 0.0,
        "mean_confidence_on_misses": float(misses["confidence"].mean()) if len(misses) else 0.0,
        "mean_confidence_on_hits": float(hits["confidence"].mean()) if len(hits) else 0.0,
        "patterns": patterns,
        "worst_patterns": [p["pattern"] for p in worst],
    }


# ----------------------------------------------------------------------
# Self-learning feedback loop
# ----------------------------------------------------------------------


@dataclass
class FeedbackResult:
    baseline_tune: dict[str, Any]
    candidates: list[dict[str, Any]]
    chosen: WalkForwardConfig
    chosen_validation: dict[str, Any]
    baseline_validation: dict[str, Any]
    improved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_tune": self.baseline_tune, "candidates": self.candidates, "chosen": self.chosen.to_dict(),
            "chosen_validation": self.chosen_validation, "baseline_validation": self.baseline_validation, "improved": self.improved,
        }


def _score(metrics: dict[str, Any]) -> float:
    """Primary objective: directional accuracy on traded days; P&L breaks ties."""
    return metrics["traded_directional_accuracy"] + 0.001 * metrics["total_return_pct"]


def build_candidates(base: WalkForwardConfig, analysis: dict[str, Any]) -> list[WalkForwardConfig]:
    """Derive adjustment candidates from the failure analysis."""
    worst_ex_ante = [p for p in analysis["patterns"] if p["ex_ante"] and p["pattern"] in REGIME_FLAGS and p["days"] >= 10 and p["lift_vs_overall"] > 0.03]
    worst_ex_ante.sort(key=lambda p: p["lift_vs_overall"], reverse=True)
    candidates: list[WalkForwardConfig] = []
    candidates.append(base.copy(
        label="regularised",
        xgb_params={**base.xgb_params, "max_depth": 3, "min_child_weight": 10, "learning_rate": 0.03, "reg_lambda": 5.0},
        lgbm_params={**base.lgbm_params, "num_leaves": 7, "min_child_samples": 60, "learning_rate": 0.03, "reg_lambda": 5.0},
    ))
    if worst_ex_ante:
        top = [p["pattern"] for p in worst_ex_ante[:2]]
        candidates.append(base.copy(label="upweight_" + "+".join(top), regime_weights={**base.regime_weights, **{f: 1.5 for f in top}}))
        candidates.append(base.copy(label="skip_" + worst_ex_ante[0]["pattern"], skip_regimes=list({*base.skip_regimes, worst_ex_ante[0]["pattern"]})))
    candidates.append(base.copy(label="higher_threshold", threshold=min(0.9, base.threshold + 0.05)))
    return candidates


def self_improve(
    ohlcv: pd.DataFrame, base: WalkForwardConfig, data: PreparedData | None = None, candidates: list[WalkForwardConfig] | None = None, verbose: bool = True
) -> FeedbackResult:
    """Tune on the first half of the window, validate on the second half."""
    data = data or prepare_data(ohlcv, base.horizon_bars, base.feature_set)
    span = base.eval_start_days_ago - base.eval_end_days_ago
    mid = base.eval_end_days_ago + span // 2
    tune_cfg = base.copy(eval_end_days_ago=mid)
    val_cfg = base.copy(eval_start_days_ago=mid)

    base_tune = run_walk_forward(ohlcv, tune_cfg, data, verbose)
    base_val = run_walk_forward(ohlcv, val_cfg, data, verbose)
    candidates = candidates if candidates is not None else build_candidates(base, base_tune.failure_analysis)

    rows: list[dict[str, Any]] = [{"label": base.label, "tune": base_tune.metrics, "score": _score(base_tune.metrics)}]
    best_cfg, best_score = base, _score(base_tune.metrics)
    for cand in candidates:
        res = run_walk_forward(ohlcv, cand.copy(eval_end_days_ago=mid), data, verbose)
        score = _score(res.metrics)
        rows.append({"label": cand.label, "tune": res.metrics, "score": score})
        if score > best_score + 1e-9:
            best_cfg, best_score = cand, score

    chosen_val = base_val.metrics if best_cfg is base else run_walk_forward(ohlcv, best_cfg.copy(eval_start_days_ago=mid), data, verbose).metrics
    improved = best_cfg is not base and _score(chosen_val) >= _score(base_val.metrics)
    if not improved:
        best_cfg = base  # do not keep a change that fails validation
        chosen_val = base_val.metrics
    for row in rows:
        row["chosen"] = row["label"] == best_cfg.label
    return FeedbackResult(base_tune.metrics, rows, best_cfg, chosen_val, base_val.metrics, improved)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def format_report(result: WalkForwardResult) -> str:
    m, a, c = result.metrics, result.failure_analysis, result.config
    lines = [
        f"=== Walk-forward daily evaluation [{c.label}] {settings.symbol} ===",
        f"Window              : {m['start'][:10]} -> {m['end'][:10]} ({m['days']} days, {result.retrains} retrains, {result.runtime_seconds:.0f}s)",
        f"Directional accuracy: {m['directional_accuracy']:.1%} (all days) | {m['traded_directional_accuracy']:.1%} on {m['traded_days']} traded days ({m['coverage']:.0%} coverage)",
        f"Target price error  : MAE {m['mae_pct']:.2f}% ({m['mae_usd']:,.0f} USD) | RMSE {m['rmse_pct']:.2f}% ({m['rmse_usd']:,.0f} USD) | naive MAE {m['naive_mae_pct']:.2f}%",
        f"Simulated P&L       : {m['total_pnl']:+,.0f} USDT ({m['total_return_pct']:+.2f}%) | buy&hold {m['buy_and_hold_return_pct']:+.2f}%",
        f"Trades              : win rate {m['win_rate']:.1%}, profit factor {m['profit_factor']:.2f}, Sharpe {m['sharpe_ratio']:.2f}, max DD {m['max_drawdown_pct']:.2f}%, exits {m['exit_reasons']}",
        f"Misses              : {a['misses']} / {m['days']} ({a['overall_miss_rate']:.1%}); mean ATR% on misses {a['mean_atr_pct_on_misses'] * 100:.2f} vs hits {a['mean_atr_pct_on_hits'] * 100:.2f}",
        "Failure patterns (miss rate vs overall):",
    ]
    for p in a["patterns"][:8]:
        lines.append(f"  - {p['pattern']:<16} days={p['days']:>4} miss={p['miss_rate']:.1%} lift={p['lift_vs_overall']:+.1%} abs_err={p['mean_abs_error_pct']:.2f}%")
    return "\n".join(lines)


def learnings_entry(before: WalkForwardResult, after: WalkForwardResult | None, feedback: FeedbackResult | None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m0 = before.metrics
    out = [f"\n## {now} — Walk-forward daily evaluation ({m0['start'][:10]} → {m0['end'][:10]})\n"]
    out.append("### Baseline\n")
    out.append("| Metric | Value |\n|---|---|")
    out.append(f"| Days evaluated | {m0['days']} (retrains: {before.retrains}, every {before.config.retrain_every_days}d) |")
    out.append(f"| Directional accuracy (all / traded) | {m0['directional_accuracy']:.1%} / {m0['traded_directional_accuracy']:.1%} ({m0['traded_days']} trades) |")
    out.append(f"| Target price MAE / RMSE | {m0['mae_pct']:.2f}% / {m0['rmse_pct']:.2f}% ({m0['mae_usd']:,.0f} / {m0['rmse_usd']:,.0f} USD) |")
    out.append(f"| Naive (no-change) MAE | {m0['naive_mae_pct']:.2f}% |")
    out.append(f"| Simulated P&L | {m0['total_pnl']:+,.0f} USDT ({m0['total_return_pct']:+.2f}%), buy&hold {m0['buy_and_hold_return_pct']:+.2f}% |")
    out.append(f"| Win rate / profit factor | {m0['win_rate']:.1%} / {m0['profit_factor']:.2f} |")
    out.append(f"| Sharpe / max drawdown | {m0['sharpe_ratio']:.2f} / {m0['max_drawdown_pct']:.2f}% |")
    a = before.failure_analysis
    out.append("\n### Error analysis (misses)\n")
    out.append(f"* {a['misses']} misses out of {m0['days']} days ({a['overall_miss_rate']:.1%}).")
    out.append(f"* Mean ATR% on misses {a['mean_atr_pct_on_misses'] * 100:.2f} vs hits {a['mean_atr_pct_on_hits'] * 100:.2f}; mean confidence on misses {a['mean_confidence_on_misses']:.2f} vs hits {a['mean_confidence_on_hits']:.2f}.")
    out.append("* Miss rate by pattern (lift vs overall):")
    for p in a["patterns"][:10]:
        out.append(f"  * `{p['pattern']}`: {p['days']} days, miss {p['miss_rate']:.1%} ({p['lift_vs_overall']:+.1%}), abs err {p['mean_abs_error_pct']:.2f}%")
    if a["worst_patterns"]:
        out.append(f"* Worst patterns: {', '.join(a['worst_patterns'])}")
    if feedback is not None:
        out.append("\n### Self-learning feedback loop\n")
        out.append("Tuned on the first half of the window, validated on the second half (score = traded accuracy + 0.001 × return%).\n")
        out.append("| Candidate | Tune traded acc. | Tune return | Tune MAE | Score | Chosen |\n|---|---|---|---|---|---|")
        for row in feedback.candidates:
            t = row["tune"]
            out.append(f"| {row['label']} | {t['traded_directional_accuracy']:.1%} | {t['total_return_pct']:+.2f}% | {t['mae_pct']:.2f}% | {row['score']:.4f} | {'✅' if row['chosen'] else ''} |")
        bv, cv = feedback.baseline_validation, feedback.chosen_validation
        out.append(f"\nValidation half — baseline: traded acc {bv['traded_directional_accuracy']:.1%}, return {bv['total_return_pct']:+.2f}%, MAE {bv['mae_pct']:.2f}% | "
                   f"chosen ({feedback.chosen.label}): traded acc {cv['traded_directional_accuracy']:.1%}, return {cv['total_return_pct']:+.2f}%, MAE {cv['mae_pct']:.2f}%")
        out.append(f"\n**Decision:** {'adopted `' + feedback.chosen.label + '`' if feedback.improved else 'kept baseline (no candidate generalised to the validation half)'}.")
        if feedback.improved:
            ch = feedback.chosen
            out.append(f"Adjustments: xgb={json.dumps(ch.xgb_params)}, lgbm={json.dumps(ch.lgbm_params)}, regime_weights={ch.regime_weights}, skip_regimes={ch.skip_regimes}, threshold={ch.threshold}")
    if after is not None:
        m1 = after.metrics
        out.append("\n### Full-window re-run with the chosen configuration\n")
        out.append("| Metric | Baseline | Chosen |\n|---|---|---|")
        out.append(f"| Directional accuracy (all) | {m0['directional_accuracy']:.1%} | {m1['directional_accuracy']:.1%} |")
        out.append(f"| Directional accuracy (traded) | {m0['traded_directional_accuracy']:.1%} | {m1['traded_directional_accuracy']:.1%} |")
        out.append(f"| Trades | {m0['traded_days']} | {m1['traded_days']} |")
        out.append(f"| MAE / RMSE | {m0['mae_pct']:.2f}% / {m0['rmse_pct']:.2f}% | {m1['mae_pct']:.2f}% / {m1['rmse_pct']:.2f}% |")
        out.append(f"| P&L | {m0['total_return_pct']:+.2f}% | {m1['total_return_pct']:+.2f}% |")
        out.append(f"| Sharpe / max DD | {m0['sharpe_ratio']:.2f} / {m0['max_drawdown_pct']:.2f}% | {m1['sharpe_ratio']:.2f} / {m1['max_drawdown_pct']:.2f}% |")
    return "\n".join(out) + "\n"


def append_learnings(text: str, path: Path | None = None) -> Path:
    path = path or (PROJECT_ROOT / "SYSTEM_LEARNINGS.md")
    if not path.exists():
        path.write_text("# SYSTEM_LEARNINGS\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Appended walk-forward report to %s", path)
    return path


def save_json(payload: dict[str, Any], name: str = "walk_forward_report.json") -> Path:
    path = settings.models_dir / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-by-day walk-forward evaluation with self-improvement")
    parser.add_argument("--start-days-ago", type=int, default=365)
    parser.add_argument("--end-days-ago", type=int, default=90)
    parser.add_argument("--retrain-every", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--fast", action="store_true", help="fewer trees (quick smoke run)")
    parser.add_argument("--no-improve", action="store_true", help="skip the self-learning loop")
    parser.add_argument("--no-append", action="store_true", help="do not write to SYSTEM_LEARNINGS.md")
    args = parser.parse_args()

    cfg = WalkForwardConfig(
        eval_start_days_ago=args.start_days_ago, eval_end_days_ago=args.end_days_ago,
        retrain_every_days=args.retrain_every, threshold=args.threshold,
    )
    if args.fast:
        cfg.xgb_params = {**cfg.xgb_params, "n_estimators": 60}
        cfg.lgbm_params = {**cfg.lgbm_params, "n_estimators": 60}
        cfg.reg_params = {**cfg.reg_params, "n_estimators": 80}

    ohlcv = storage.get_ohlcv()
    data = prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    baseline = run_walk_forward(ohlcv, cfg, data)
    print("\n" + format_report(baseline))

    feedback = after = None
    if not args.no_improve:
        feedback = self_improve(ohlcv, cfg, data)
        if feedback.improved:
            after = run_walk_forward(ohlcv, feedback.chosen.copy(label=feedback.chosen.label), data)
            print("\n" + format_report(after))
        else:
            print("\nSelf-improvement: no candidate beat the baseline on the validation half; baseline kept.")

    entry = learnings_entry(baseline, after, feedback)
    if not args.no_append:
        append_learnings(entry)
    payload = {"baseline": baseline.to_dict(), "feedback": feedback.to_dict() if feedback else None, "improved_run": after.to_dict() if after else None}
    path = save_json(payload)
    print(f"\nSaved detailed JSON to {path}")


if __name__ == "__main__":
    main()
