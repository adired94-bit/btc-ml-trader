"""Idea 09 - trade with the trend only.

Keep the extended-features daily ensemble but only act on its signal when it agrees with the
EMA-200 trend (slope over the last 24 bars and price relative to the EMA). Compared against:

(a) the unfiltered model (baseline, threshold 0.55),
(b) a pure trend-following rule with no model (go with the EMA-200 slope every day),
(c) model + trend filter with threshold 0.50 (every model call that agrees with the trend).

Protocol: one walk-forward pass (365 -> 90 days ago, retrain every 14 days, <= 80 trees) gives the
per-day probabilities.  Every strategy - the baseline included - is then re-simulated from the same
DayRecords with ``simulate_day_trade`` so the comparison is apples to apples.  The window is split
into a tuning half (365 -> 227) and a validation half (227 -> 90); a small grid of filter variants
is selected on the tuning half only.

Run from the repo root::

    python scripts/experiments/idea09_trend_filter.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PROJECT_ROOT  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea09")

OUT_PATH = PROJECT_ROOT / "models" / "idea09_trend_filter.json"
TUNE_SPLIT_DAYS_AGO = 227


@dataclass
class Strategy:
    name: str
    threshold: float  # model confidence needed (0.50 == take every model call)
    use_model: bool  # False -> pure trend rule
    trend_filter: bool = True  # model strategies: only trade when the model agrees with the trend
    slope_bars: int = 24  # EMA-200 slope lookback in hourly bars
    require_price_side: bool = True  # also require close above/below EMA-200


# ----------------------------------------------------------------------
# Trend state per day (causal: uses only bars <= the prediction bar)
# ----------------------------------------------------------------------


def trend_direction(ind: pd.DataFrame, pos: int, slope_bars: int, require_price_side: bool) -> str:
    """Return "UP", "DOWN" or "NONE" for the trend state at bar ``pos``."""
    ema = ind["ema_200"]
    now, before = float(ema.iloc[pos]), float(ema.iloc[max(0, pos - slope_bars)])
    close = float(ind["close"].iloc[pos])
    if not (np.isfinite(now) and np.isfinite(before)):
        return "NONE"
    if now > before and (not require_price_side or close > now):
        return "UP"
    if now < before and (not require_price_side or close < now):
        return "DOWN"
    return "NONE"


# ----------------------------------------------------------------------
# Re-simulation from DayRecords
# ----------------------------------------------------------------------


def resimulate(
    records: list[WF.DayRecord], data: WF.PreparedData, cfg: WF.WalkForwardConfig, strat: Strategy
) -> dict[str, Any]:
    """Apply the strategy to each day and re-run the hour-by-hour trade simulation."""
    idx = data.ohlcv.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    equity = cfg.initial_equity
    rows: list[dict[str, Any]] = []
    for r in records:
        ts = pd.Timestamp(r.day)
        # the record's "day" is the UTC day; its last hourly bar is the prediction bar
        bar_ts = WF.day_close_bars(idx[(idx >= ts) & (idx < ts + pd.Timedelta(days=1))])
        i = int(pos[bar_ts.iloc[0]])
        trend = trend_direction(data.indicators, i, strat.slope_bars, strat.require_price_side)
        if strat.use_model:
            view = r.predicted_direction
            traded = r.confidence >= strat.threshold and (trend == view or not strat.trend_filter)
        else:
            view = trend
            traded = trend != "NONE"
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        atr = float(data.daily_atr.iloc[i])
        if traded and view in ("UP", "DOWN") and np.isfinite(atr) and atr > 0:
            side = "LONG" if view == "UP" else "SHORT"
            bars = data.ohlcv.iloc[i + 1: i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, atr, equity, cfg)
            equity += pnl
        else:
            traded = False
        rows.append(
            {
                "day": r.day, "view": view, "actual": r.actual_direction, "hit": view == r.actual_direction,
                "traded": traded, "side": side, "pnl": pnl, "ret_pct": ret_pct, "exit": exit_reason, "equity": equity,
                "trend": trend, "model_dir": r.predicted_direction, "confidence": r.confidence,
            }
        )
    return summarize(pd.DataFrame(rows), cfg)


def summarize(df: pd.DataFrame, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    traded = df[df["traded"]]
    daily_ret = df["ret_pct"] / 100
    sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if daily_ret.std() > 0 else 0.0
    equity = pd.concat([pd.Series([cfg.initial_equity]), df["equity"]]).reset_index(drop=True)
    max_dd = float(-(equity / equity.cummax() - 1).min() * 100)
    has_view = df[df["view"].isin(["UP", "DOWN"])]
    return {
        "days": int(len(df)),
        "directional_accuracy": float(has_view["hit"].mean()) if len(has_view) else float("nan"),
        "days_with_view": int(len(has_view)),
        "traded_days": int(len(traded)),
        "traded_directional_accuracy": float(traded["hit"].mean()) if len(traded) else float("nan"),
        "longs": int((traded["side"] == "LONG").sum()),
        "shorts": int((traded["side"] == "SHORT").sum()),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else float("nan"),
        "total_return_pct": float((df["equity"].iloc[-1] / cfg.initial_equity - 1) * 100),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "exit_reasons": {k: int(v) for k, v in traded["exit"].value_counts().items()},
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def score(m: dict[str, Any]) -> float:
    """Selection score on the tuning half: Sharpe, penalised when it rests on very few trades."""
    if m["traded_days"] < 20:
        return -99.0
    return m["sharpe_ratio"]


def fmt_row(name: str, half: str, m: dict[str, Any]) -> str:
    acc = m["directional_accuracy"] * 100
    tacc = m["traded_directional_accuracy"] * 100 if m["traded_days"] else float("nan")
    return (
        f"| {name} | {half} | {m['days']} | {acc:.1f}% | {m['traded_days']} | {tacc:.1f}% | "
        f"{m['total_return_pct']:+.2f}% | {m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.2f}% |"
    )


def main() -> None:
    t0 = time.perf_counter()
    ohlcv = storage.get_ohlcv()
    logger.info("OHLCV: %d bars %s -> %s", len(ohlcv), ohlcv.index[0], ohlcv.index[-1])

    cfg = WF.WalkForwardConfig(
        label="ext+ens", feature_set="extended", model="ensemble",
        eval_start_days_ago=365, eval_end_days_ago=90, retrain_every_days=14, threshold=0.55,
        xgb_params={"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 2},
        lgbm_params={"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.05, "n_jobs": 2},
        reg_params={**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2},
    )
    data = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    result = WF.run_walk_forward(ohlcv, cfg, data, verbose=True)
    logger.info("Walk-forward done: %d days, %d retrains, %.0fs", len(result.records), result.retrains, result.runtime_seconds)

    split = (ohlcv.index[-1] - pd.Timedelta(days=TUNE_SPLIT_DAYS_AGO)).floor("D")
    tune = [r for r in result.records if pd.Timestamp(r.day) < split]
    valid = [r for r in result.records if pd.Timestamp(r.day) >= split]
    logger.info("Tune half: %d days (< %s), validation half: %d days", len(tune), split.date(), len(valid))

    # The three requested comparisons + a small grid of filter variants (selected on the tune half only).
    core = [
        Strategy("baseline", 0.55, True, trend_filter=False),
        Strategy("model+trend@0.55", 0.55, True, True, 24, True),
        Strategy("pure_trend_slope24", 0.0, False, False, 24, False),
        Strategy("pure_trend_slope24+price", 0.0, False, False, 24, True),
        Strategy("model+trend@0.50", 0.50, True, True, 24, True),
    ]
    grid = [
        Strategy(f"model+trend@{thr:.2f}_s{sb}{'p' if ps else ''}", thr, True, True, sb, ps)
        for thr in (0.50, 0.55)
        for sb in (24, 72, 168)
        for ps in (True, False)
        if not (sb == 24 and ps)  # already in core
    ] + [Strategy(f"pure_trend_slope{sb}{'+price' if ps else ''}", 0.0, False, False, sb, ps) for sb in (72, 168) for ps in (False, True)]

    results: dict[str, dict[str, Any]] = {}
    for s in core + grid:
        results[s.name] = {
            "strategy": dict(s.__dict__),
            "tune": resimulate(tune, data, cfg, s),
            "validation": resimulate(valid, data, cfg, s),
        }
    best_grid = max(grid, key=lambda s: score(results[s.name]["tune"]))

    header = "| Strategy | Half | Days | Dir acc | Traded | Traded acc | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|"
    lines = ["## Core comparison (tune half)", header]
    lines += [fmt_row(s.name, "tune", results[s.name]["tune"]) for s in core]
    lines += ["", "## Core comparison (validation half)", header]
    lines += [fmt_row(s.name, "validation", results[s.name]["validation"]) for s in core]
    lines += ["", f"## Grid (tune half) - best by Sharpe: {best_grid.name}", header]
    lines += [fmt_row(s.name, "tune", results[s.name]["tune"]) for s in sorted(grid, key=lambda s: -score(results[s.name]["tune"]))]
    lines += ["", "## Grid winner on validation", header, fmt_row(best_grid.name, "validation", results[best_grid.name]["validation"])]
    report = "\n".join(lines)
    print(report)

    # Agreement diagnostics: how often does the model agree with the trend, and how good is each when they agree?
    diag = {}
    for half_name, recs in (("tune", tune), ("validation", valid)):
        idx = data.ohlcv.index
        pos = pd.Series(np.arange(len(idx)), index=idx)
        agree = hit_agree = hit_model_disagree = disagree = 0
        for r in recs:
            ts = pd.Timestamp(r.day)
            bar_ts = WF.day_close_bars(idx[(idx >= ts) & (idx < ts + pd.Timedelta(days=1))])
            tr = trend_direction(data.indicators, int(pos[bar_ts.iloc[0]]), 24, True)
            if tr == r.predicted_direction:
                agree += 1
                hit_agree += int(r.hit)
            elif tr != "NONE":
                disagree += 1
                hit_model_disagree += int(r.hit)
        diag[half_name] = {
            "agree_days": agree, "model_acc_when_agree": hit_agree / agree if agree else None,
            "disagree_days": disagree, "model_acc_when_disagree": hit_model_disagree / disagree if disagree else None,
        }
    print("\nAgreement diagnostics:", json.dumps(diag, indent=2))

    payload = {
        "idea": "09 trend filter (EMA-200 slope + price side) on the extended-features ensemble",
        "config": cfg.to_dict(),
        "walk_forward_days": len(result.records),
        "retrains": result.retrains,
        "walk_forward_runtime_s": result.runtime_seconds,
        "tune_split": split.isoformat(),
        "tune_days": len(tune),
        "validation_days": len(valid),
        "core_strategies": [s.name for s in core],
        "grid_best_on_tune": best_grid.name,
        "results": results,
        "agreement_diagnostics": diag,
        "report_markdown": report,
        "total_runtime_s": time.perf_counter() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    logger.info("Saved %s (total %.0fs)", OUT_PATH, payload["total_runtime_s"])


if __name__ == "__main__":
    main()
