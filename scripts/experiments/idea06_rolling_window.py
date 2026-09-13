"""Idea 06 - training-window memory: rolling windows vs time-decayed sample weights.

The production walk-forward loop trains on *all* history up to day T (expanding
window). This experiment asks whether the daily direction model generalises
better when it forgets old regimes:

* rolling windows - only the last 90 / 180 / 365 days of labelled bars;
* exponential time decay - expanding window, sample weight
  ``0.5 ** (age_days / half_life)`` with half-life 60 and 180 days.

Everything else (extended features, XGB+LGBM ensemble, retrain cadence, trade
simulation) is identical to the baseline so P&L is comparable. Tune half is the
older part of the evaluation window (365 -> 227 days ago), validation half the
newer part (227 -> 90 days ago); the validation half is never used to pick a
winner - the ranking is done on the tune half and reported on both.

    venv\\Scripts\\python.exe scripts\\experiments\\idea06_rolling_window.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea06")

# Fast boosters (shared CPU with nine other agents).
FAST_XGB = {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_LGBM = {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2}

EVAL_START, EVAL_END = 365, 90
MID = EVAL_END + (EVAL_START - EVAL_END) // 2  # 227


@dataclass(frozen=True)
class MemorySpec:
    label: str
    window_days: int | None = None  # None -> expanding window
    half_life_days: float | None = None  # None -> flat weights


SPECS: list[MemorySpec] = [
    MemorySpec("baseline_expanding"),
    MemorySpec("rolling_90d", window_days=90),
    MemorySpec("rolling_180d", window_days=180),
    MemorySpec("rolling_365d", window_days=365),
    MemorySpec("decay_hl60d", half_life_days=60.0),
    MemorySpec("decay_hl180d", half_life_days=180.0),
]


class MemoryDailyModels(WF.DailyModels):
    """DailyModels with an optional rolling training window and exponential time-decay weights.

    Look-ahead safety is inherited: the caller passes ``train_end`` = last bar whose 24-bar
    label is realised at the close of day T; we only *drop* older rows or *down-weight* them.
    """

    def __init__(self, cfg: WF.WalkForwardConfig, spec: MemorySpec) -> None:
        super().__init__(cfg)
        self.spec = spec

    def fit(self, data: WF.PreparedData, train_end: pd.Timestamp) -> "MemoryDailyModels":
        mask = (data.features.index <= train_end) & (data.direction >= 0).to_numpy()
        if self.spec.window_days is not None:
            mask &= np.asarray(data.features.index > train_end - pd.Timedelta(days=self.spec.window_days))
        X = data.features[mask].dropna()
        y_dir = data.direction.loc[X.index]
        y_ret = data.future_return.loc[X.index]
        if len(X) < self.cfg.min_train_rows:
            raise ValueError(f"Only {len(X)} training rows available at {train_end}; need {self.cfg.min_train_rows}")
        weights = np.ones(len(X))
        for flag, extra in self.cfg.regime_weights.items():
            if flag in data.regimes.columns and extra > 0:
                weights = weights * np.where(data.regimes.loc[X.index, flag].to_numpy(), 1.0 + extra, 1.0)
        if self.spec.half_life_days is not None:
            age_days = (train_end - X.index).total_seconds() / 86_400.0
            decay = np.power(0.5, np.asarray(age_days) / self.spec.half_life_days)
            weights = weights * decay / decay.mean()  # mean 1 keeps min_child_weight semantics comparable
        self.feature_names = list(X.columns)
        self.direction.fit(X, y_dir, sample_weight=weights)
        self.regressor.fit(X.to_numpy(dtype=np.float32), y_ret.to_numpy(dtype=np.float32), sample_weight=weights)
        self.train_end = train_end
        self.n_rows = len(X)
        self.effective_rows = float(weights.sum() ** 2 / np.square(weights).sum())  # Kish effective sample size
        return self


def run_loop(ohlcv: pd.DataFrame, cfg: WF.WalkForwardConfig, data: WF.PreparedData, spec: MemorySpec) -> dict[str, Any]:
    """Faithful copy of ``walk_forward.run_walk_forward`` with the model class swapped for ``MemoryDailyModels``."""
    t0 = time.perf_counter()
    idx = ohlcv.index
    start, end = WF.evaluation_days(idx, cfg)
    closes = WF.day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    models: MemoryDailyModels | None = None
    last_train_day: pd.Timestamp | None = None
    retrains, rows_seen, eff_seen = 0, [], []
    equity = cfg.initial_equity
    records: list[WF.DayRecord] = []
    atr_rank = data.indicators["atr_pct"].rolling(24 * 90, min_periods=24 * 20).rank(pct=True)
    pos = pd.Series(np.arange(len(idx)), index=idx)

    for day in days:
        ts = closes[day]
        i = int(pos[ts])
        if i + cfg.horizon_bars >= len(idx):
            break
        train_end = idx[i - cfg.horizon_bars]  # labels of bars <= train_end are realised at the close of day T
        if models is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
            models = MemoryDailyModels(cfg, spec).fit(data, train_end)
            last_train_day = day
            retrains += 1
            rows_seen.append(models.n_rows)
            eff_seen.append(models.effective_rows)
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
        regimes_now = [f for f in WF.REGIME_FLAGS if bool(data.regimes.iloc[i][f])]
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
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, daily_atr, equity, cfg)
            equity += pnl
        else:
            traded = False
        records.append(
            WF.DayRecord(
                day=day.isoformat(), target_day=(day + pd.Timedelta(hours=cfg.horizon_bars)).isoformat(), train_end=train_end.isoformat(),
                close=close, actual_close=actual_close, actual_return_pct=float(actual_ret),
                predicted_direction=pred_dir, actual_direction=actual_dir, hit=pred_dir == actual_dir,
                prob_up=p_up, prob_down=p_down, confidence=float(confidence), traded=traded,
                target_price=float(target), predicted_return_pct=float(r_hat * 100),
                error_pct=float((target - actual_close) / actual_close * 100), abs_error_pct=float(abs(target - actual_close) / actual_close * 100),
                risk_level=WF.risk_level_from_rank(float(atr_rank.iloc[i])), atr_pct=float(data.indicators["atr_pct"].iloc[i]),
                volume_ratio=float(data.indicators["volume"].iloc[i] / max(data.indicators["volume"].iloc[max(0, i - 24):i].mean(), 1e-9)),
                regimes=regimes_now + outcome_flags, trade_side=side, trade_pnl=float(pnl), trade_return_pct=float(ret_pct),
                trade_exit=exit_reason, equity=float(equity),
                momentum_direction="UP" if close >= float(data.ohlcv["close"].iloc[max(0, i - 720)]) else "DOWN",
            )
        )
    if not records:
        raise ValueError("no records produced")
    m = WF.compute_metrics(records, cfg)
    return {
        "days": m["days"], "hit": m["directional_accuracy"], "traded_days": m["traded_days"],
        "traded_hit": m["traded_directional_accuracy"], "return_pct": m["total_return_pct"],
        "sharpe": m["sharpe_ratio"], "max_dd": m["max_drawdown_pct"], "momentum_hit": m["momentum_30d_accuracy"],
        "always_up": m["always_up_accuracy"], "up_predictions": m["up_predictions"], "retrains": retrains,
        "mean_train_rows": float(np.mean(rows_seen)), "mean_effective_rows": float(np.mean(eff_seen)),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }


def fmt(label: str, half: str, m: dict[str, Any]) -> str:
    return (
        f"| {label} | {half} | {m['days']} | {m['hit']:.1%} | {m['traded_days']} | {m['traded_hit']:.1%} | "
        f"{m['return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_dd']:.2f}% | {m['mean_train_rows']:.0f} / {m['mean_effective_rows']:.0f} |"
    )


def main() -> None:
    t0 = time.perf_counter()
    ohlcv = storage.get_ohlcv()
    base = WF.WalkForwardConfig(
        feature_set="extended", model="ensemble", retrain_every_days=14,
        xgb_params=FAST_XGB, lgbm_params=FAST_LGBM, reg_params=FAST_REG, label="idea06",
    )
    data = WF.prepare_data(ohlcv, base.horizon_bars, base.feature_set)
    halves = {
        "tune": base.copy(eval_start_days_ago=EVAL_START, eval_end_days_ago=MID),
        "validation": base.copy(eval_start_days_ago=MID, eval_end_days_ago=EVAL_END),
    }
    results: dict[str, dict[str, Any]] = {}
    head = "| Config | Half | Days | Hit | Traded | Traded hit | Return | Sharpe | Max DD | Train rows / effective |\n|---|---|---|---|---|---|---|---|---|---|"
    print(head, flush=True)
    for spec in SPECS:
        results[spec.label] = {"spec": spec.__dict__}
        for half, cfg in halves.items():
            m = run_loop(ohlcv, cfg, data, spec)
            results[spec.label][half] = m
            print(fmt(spec.label, half, m), flush=True)

    # Rank on the tune half only (same score as scripts/improve.py); report validation for everything.
    def score(m: dict[str, Any]) -> float:
        return m["hit"] + 0.5 * m["traded_hit"] + 0.0005 * m["return_pct"]

    ranked = sorted(SPECS, key=lambda s: score(results[s.label]["tune"]), reverse=True)
    summary = {
        "tune_ranking": [s.label for s in ranked],
        "best_on_tune": ranked[0].label,
        "best_on_tune_validation_hit": results[ranked[0].label]["validation"]["hit"],
        "baseline_validation_hit": results["baseline_expanding"]["validation"]["hit"],
        "runtime_s": round(time.perf_counter() - t0, 1),
        "eval_window": {"start_days_ago": EVAL_START, "mid_days_ago": MID, "end_days_ago": EVAL_END},
        "data_range": [str(ohlcv.index[0]), str(ohlcv.index[-1])],
        "config": base.to_dict(),
    }
    out = settings.models_dir / "idea06_rolling_window.json"
    out.write_text(json.dumps({"results": results, "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"\nbest on tune: {summary['best_on_tune']} -> validation hit {summary['best_on_tune_validation_hit']:.1%} "
          f"(baseline validation hit {summary['baseline_validation_hit']:.1%}); runtime {summary['runtime_s']}s; saved {out}")


if __name__ == "__main__":
    main()
