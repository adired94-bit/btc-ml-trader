"""Idea 02 - probability calibration before thresholding.

Hypothesis: the ensemble's raw P(up) is not calibrated, so a fixed 0.55 confidence
threshold trades on noise. Re-map P(up) with a calibrator (isotonic regression or
Platt scaling) fitted on a rolling window of the model's *own past out-of-sample*
predictions (previous 90 / 180 evaluation days), then trade only when the calibrated
probability of the chosen side is >= 0.55 / 0.60.

Protocol (same for every idea in this campaign):

* 2-year cache, evaluation window 365 -> 90 days ago. Tuning half 365 -> 227,
  validation half 227 -> 90 (same day sets as ``scripts/improve.py``).
* One walk-forward pass per primary model collects out-of-sample predictions from
  ``365 + 180`` days ago so that the calibrator already has history on the first
  tuning day. Retrain every 14 days, 80 trees, n_jobs=2.
* Baseline and calibrated variants use the *identical* predictions and the identical
  trade simulation (``walk_forward.simulate_day_trade``); only the trade filter (and,
  for the ``cal``-side variants, the direction) differs. That makes the comparison
  apples to apples.
* Zero look-ahead: the calibrator used at the close of day T is fitted on predictions
  made on days d <= T-1, whose outcomes (close of d+1 <= T) are known at that time.

Run::

    venv\\Scripts\\python.exe scripts\\experiments\\idea02_calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea02")

EVAL_START, EVAL_END = 365, 90
MID = EVAL_END + (EVAL_START - EVAL_END) // 2  # 227, as in scripts/improve.py
CAL_WINDOWS = (90, 180)
HISTORY_DAYS = max(CAL_WINDOWS)  # walk-forward starts this much earlier to warm the calibrator
MIN_CAL_POINTS = 60
THRESHOLDS = (0.55, 0.60)
METHODS = ("isotonic", "platt")
SIDE_MODES = ("raw", "cal")  # raw: direction from the raw model; cal: direction from the calibrated probability
MIN_TRADES_FOR_SELECTION = 20
BUCKETS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)
FAST = {"n_estimators": 80, "n_jobs": 2}


def fast_cfg(**changes: Any) -> WF.WalkForwardConfig:
    base = WF.WalkForwardConfig(
        eval_start_days_ago=EVAL_START + HISTORY_DAYS,
        eval_end_days_ago=EVAL_END,
        retrain_every_days=14,
        xgb_params={**WF.WF_XGB_PARAMS, **FAST},
        lgbm_params={**WF.WF_LGBM_PARAMS, **FAST},
        reg_params={**WF.WF_REG_PARAMS, **FAST},
    )
    return base.copy(**changes)


# ----------------------------------------------------------------------
# 1. Collect out-of-sample predictions (no trading yet)
# ----------------------------------------------------------------------


@dataclass
class Pred:
    day: pd.Timestamp
    i: int
    train_end: pd.Timestamp
    close: float
    actual_close: float
    actual_ret: float
    actual_dir: str
    p_up: float
    p_down: float
    r_hat: float
    atr: float
    atr_rank: float
    volume_ratio: float
    regimes: list[str]
    momentum_direction: str


def collect_predictions(ohlcv: pd.DataFrame, cfg: WF.WalkForwardConfig, data: WF.PreparedData) -> tuple[list[Pred], int]:
    """Mirror of ``walk_forward.run_walk_forward`` that only records the model output."""
    idx = ohlcv.index
    start, end = WF.evaluation_days(idx, cfg)
    closes = WF.day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    atr_rank = data.indicators["atr_pct"].rolling(24 * 90, min_periods=24 * 20).rank(pct=True)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    models: WF.DailyModels | None = None
    last_train_day: pd.Timestamp | None = None
    retrains = 0
    preds: list[Pred] = []
    for day in days:
        ts = closes[day]
        i = int(pos[ts])
        if i + cfg.horizon_bars >= len(idx):
            break
        train_end = idx[i - cfg.horizon_bars]  # labels of bars <= train_end are known at the close of day T
        if models is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
            models = WF.DailyModels(cfg).fit(data, train_end)
            last_train_day = day
            retrains += 1
        row = data.features.iloc[[i]]
        if row.isna().any(axis=None):
            continue
        p_up, p_down, r_hat = models.predict(row)
        close = float(data.ohlcv["close"].iloc[i])
        actual_close = float(data.ohlcv["close"].iloc[i + cfg.horizon_bars])
        actual_ret = (actual_close / close - 1) * 100
        prev_day_ret = float(close / data.ohlcv["close"].iloc[max(0, i - cfg.horizon_bars)] - 1)
        regimes_now = [f for f in WF.REGIME_FLAGS if bool(data.regimes.iloc[i][f])]
        if np.sign(prev_day_ret) != np.sign(actual_ret) and abs(prev_day_ret) > 0.005:
            regimes_now.append("trend_reversal")
        if abs(actual_ret) > 3.0:
            regimes_now.append("large_move")
        atr = float(data.daily_atr.iloc[i]) if np.isfinite(data.daily_atr.iloc[i]) else float("nan")
        vol = data.indicators["volume"]
        preds.append(
            Pred(
                day=day, i=i, train_end=train_end, close=close, actual_close=actual_close, actual_ret=float(actual_ret),
                actual_dir="UP" if actual_ret >= 0 else "DOWN", p_up=p_up, p_down=p_down, r_hat=r_hat, atr=atr,
                atr_rank=float(atr_rank.iloc[i]), volume_ratio=float(vol.iloc[i] / max(vol.iloc[max(0, i - 24):i].mean(), 1e-9)),
                regimes=regimes_now,
                momentum_direction="UP" if close >= float(data.ohlcv["close"].iloc[max(0, i - 720)]) else "DOWN",
            )
        )
    return preds, retrains


# ----------------------------------------------------------------------
# 2. Trade simulation from a decision table (identical to the walk-forward loop)
# ----------------------------------------------------------------------


@dataclass
class Decision:
    pred_dir: str
    confidence: float
    traded: bool
    p_up: float
    p_down: float


def simulate(preds: list[Pred], decisions: list[Decision], data: WF.PreparedData, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    equity = cfg.initial_equity
    records: list[WF.DayRecord] = []
    for p, d in zip(preds, decisions):
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        traded = d.traded
        if traded and np.isfinite(p.atr) and p.atr > 0:
            side = "LONG" if d.pred_dir == "UP" else "SHORT"
            bars = data.ohlcv.iloc[p.i + 1: p.i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, p.atr, equity, cfg)
            equity += pnl
        else:
            traded = False
        target = p.close * (1 + p.r_hat)
        records.append(
            WF.DayRecord(
                day=p.day.isoformat(), target_day=(p.day + pd.Timedelta(hours=cfg.horizon_bars)).isoformat(),
                train_end=p.train_end.isoformat(), close=p.close, actual_close=p.actual_close, actual_return_pct=p.actual_ret,
                predicted_direction=d.pred_dir, actual_direction=p.actual_dir, hit=d.pred_dir == p.actual_dir,
                prob_up=d.p_up, prob_down=d.p_down, confidence=d.confidence, traded=traded, target_price=float(target),
                predicted_return_pct=float(p.r_hat * 100), error_pct=float((target - p.actual_close) / p.actual_close * 100),
                abs_error_pct=float(abs(target - p.actual_close) / p.actual_close * 100),
                risk_level=WF.risk_level_from_rank(p.atr_rank), atr_pct=float("nan"), volume_ratio=p.volume_ratio,
                regimes=p.regimes, trade_side=side, trade_pnl=float(pnl), trade_return_pct=float(ret_pct),
                trade_exit=exit_reason, equity=float(equity), momentum_direction=p.momentum_direction,
            )
        )
    return WF.compute_metrics(records, cfg)


# ----------------------------------------------------------------------
# 3. Calibrators fitted on the rolling window of past OOS predictions
# ----------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    if len(np.unique(y)) < 2:  # degenerate window: fall back to the base rate
        rate = float(y.mean())
        return lambda _x: rate
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(p, y)
        return lambda x: float(iso.predict(np.array([x]))[0])
    if method == "platt":
        lr = LogisticRegression(C=1.0, max_iter=1_000).fit(_logit(p).reshape(-1, 1), y)
        return lambda x: float(lr.predict_proba(_logit(np.array([x])).reshape(-1, 1))[0, 1])
    raise ValueError(method)


def rolling_calibrated(preds: list[Pred], method: str, window_days: int) -> np.ndarray:
    """Calibrated P(up) for every prediction; NaN while the history is too short."""
    days = np.array([p.day for p in preds])
    raw = np.array([p.p_up for p in preds])
    y = np.array([1.0 if p.actual_dir == "UP" else 0.0 for p in preds])
    out = np.full(len(preds), np.nan)
    for k, p in enumerate(preds):
        lo, hi = p.day - pd.Timedelta(days=window_days), p.day - pd.Timedelta(days=1)
        mask = (days >= lo) & (days <= hi)  # outcome of day d is known at close of d+1 <= T
        if mask.sum() < MIN_CAL_POINTS:
            continue
        out[k] = fit_calibrator(method, raw[mask], y[mask])(p.p_up)
    return out


def decisions_from(preds: list[Pred], p_cal: np.ndarray | None, threshold: float, side_mode: str) -> list[Decision]:
    out: list[Decision] = []
    for k, p in enumerate(preds):
        raw_dir = "UP" if p.p_up >= p.p_down else "DOWN"
        if p_cal is None:  # baseline: raw confidence
            conf = max(p.p_up, p.p_down)
            out.append(Decision(raw_dir, conf, conf >= threshold, p.p_up, p.p_down))
            continue
        c = p_cal[k]
        if not np.isfinite(c):
            out.append(Decision(raw_dir, 0.5, False, p.p_up, p.p_down))
            continue
        if side_mode == "raw":
            conf = c if raw_dir == "UP" else 1 - c
            out.append(Decision(raw_dir, float(conf), conf >= threshold, float(c), float(1 - c)))
        else:
            cal_dir = "UP" if c >= 0.5 else "DOWN"
            conf = max(c, 1 - c)
            out.append(Decision(cal_dir, float(conf), conf >= threshold, float(c), float(1 - c)))
    return out


# ----------------------------------------------------------------------
# 4. Calibration diagnostics
# ----------------------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(p)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else float("nan")


def reliability(p: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    m = np.isfinite(p)
    p, y = p[m], y[m]
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(sel.sum())
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})", "n": n,
            "mean_pred_up": round(float(p[sel].mean()), 4) if n else None,
            "realised_up": round(float(y[sel].mean()), 4) if n else None,
        })
    return rows


def confidence_vs_hit(p_up: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Raw confidence buckets vs realised hit rate (the 'confidence is not accuracy' check)."""
    m = np.isfinite(p_up)
    p_up, y = p_up[m], y[m]
    conf = np.maximum(p_up, 1 - p_up)
    hit = ((p_up >= 0.5) == (y == 1)).astype(float)
    rows = []
    for lo, hi in ((0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)):
        sel = (conf >= lo) & (conf < hi)
        rows.append({"confidence": f"[{lo:.2f},{min(hi, 1.0):.2f})", "n": int(sel.sum()), "hit_rate": round(float(hit[sel].mean()), 4) if sel.any() else None})
    return rows


# ----------------------------------------------------------------------
# 5. Campaign
# ----------------------------------------------------------------------


def score(m: dict[str, Any]) -> float:
    return m["directional_accuracy"] + 0.5 * m["traded_directional_accuracy"] + 0.0005 * m["total_return_pct"]


def row(label: str, half: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label, "half": half, "days": m["days"], "hit": round(m["directional_accuracy"], 4),
        "traded_days": m["traded_days"], "traded_hit": round(m["traded_directional_accuracy"], 4),
        "return_pct": round(m["total_return_pct"], 2), "sharpe": round(m["sharpe_ratio"], 2),
        "max_dd": round(m["max_drawdown_pct"], 2), "score": round(score(m), 4),
    }


def md_table(rows: list[dict[str, Any]]) -> str:
    head = "| Config | Half | Days | Hit (all) | Traded | Traded hit | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|"
    body = [
        f"| {r['label']} | {r['half']} | {r['days']} | {r['hit']:.1%} | {r['traded_days']} | {r['traded_hit']:.1%} | {r['return_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% |"
        for r in rows
    ]
    return "\n".join([head, *body])


def half_slice(preds: list[Pred], ohlcv_index: pd.DatetimeIndex, start_ago: int, end_ago: int) -> list[int]:
    cfg = WF.WalkForwardConfig(eval_start_days_ago=start_ago, eval_end_days_ago=end_ago)
    start, end = WF.evaluation_days(ohlcv_index, cfg)
    return [k for k, p in enumerate(preds) if start <= p.day <= end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="base,extended", help="comma-separated feature sets for the primary model")
    args = parser.parse_args()
    t0 = time.perf_counter()

    ohlcv = storage.load_cached()
    if ohlcv is None:
        ohlcv = storage.get_ohlcv()
    logger.info("OHLCV %s -> %s (%d bars)", ohlcv.index[0], ohlcv.index[-1], len(ohlcv))
    idx = ohlcv.index

    halves = {"tune": (EVAL_START, MID), "validation": (MID, EVAL_END)}
    all_rows: dict[str, list[dict[str, Any]]] = {"tune": [], "validation": []}
    diagnostics: dict[str, Any] = {}
    sim_cfg = fast_cfg()  # fees, slippage, ATR stop, R:R are the project defaults

    for feature_set in [s.strip() for s in args.models.split(",") if s.strip()]:
        cfg = fast_cfg(feature_set=feature_set, label=f"{feature_set}+ens")
        t1 = time.perf_counter()
        data = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
        preds, retrains = collect_predictions(ohlcv, cfg, data)
        logger.info("%s: %d OOS predictions, %d retrains, %.0fs", cfg.label, len(preds), retrains, time.perf_counter() - t1)

        calibrated = {(m, w): rolling_calibrated(preds, m, w) for m in METHODS for w in CAL_WINDOWS}
        raw_p = np.array([p.p_up for p in preds])
        y = np.array([1.0 if p.actual_dir == "UP" else 0.0 for p in preds])

        for half, (s_ago, e_ago) in halves.items():
            ks = half_slice(preds, idx, s_ago, e_ago)
            sub = [preds[k] for k in ks]
            if not sub:
                raise ValueError(f"no predictions in {half} half")
            # baseline: raw confidence thresholds
            for thr in THRESHOLDS:
                m = simulate(sub, decisions_from(sub, None, thr, "raw"), data, sim_cfg)
                all_rows[half].append(row(f"{cfg.label} raw>={thr:.2f}", half, m))
            # calibrated variants
            for (method, w), pc in calibrated.items():
                pc_sub = pc[ks]
                for side_mode in SIDE_MODES:
                    for thr in THRESHOLDS:
                        m = simulate(sub, decisions_from(sub, pc_sub, thr, side_mode), data, sim_cfg)
                        all_rows[half].append(row(f"{cfg.label} {method}{w} {side_mode}-side>={thr:.2f}", half, m))
            # diagnostics
            diag = {
                "n": len(ks),
                "brier_raw": round(brier(raw_p[ks], y[ks]), 4),
                "brier_constant_0.5": 0.25,
                "brier_base_rate": round(float(np.mean((y[ks].mean() - y[ks]) ** 2)), 4),
                "reliability_raw": reliability(raw_p[ks], y[ks]),
                "confidence_vs_hit_raw": confidence_vs_hit(raw_p[ks], y[ks]),
            }
            for (method, w), pc in calibrated.items():
                diag[f"brier_{method}{w}"] = round(brier(pc[ks], y[ks]), 4)
                diag[f"reliability_{method}{w}"] = reliability(pc[ks], y[ks])
                diag[f"cal_mean_{method}{w}"] = round(float(np.nanmean(pc[ks])), 4)
                diag[f"cal_std_{method}{w}"] = round(float(np.nanstd(pc[ks])), 4)
            diagnostics[f"{cfg.label}/{half}"] = diag
        # keep raw predictions for later analysis
        diagnostics[f"{cfg.label}/predictions"] = [
            {"day": p.day.isoformat(), "p_up": round(p.p_up, 4), "actual_dir": p.actual_dir, "actual_ret": round(p.actual_ret, 4),
             **{f"cal_{m}{w}": (round(float(pc[k]), 4) if np.isfinite(pc[k]) else None) for (m, w), pc in calibrated.items()}}
            for k, p in enumerate(preds)
        ]

    # selection on the tuning half only
    tune = all_rows["tune"]
    val = {r["label"]: r for r in all_rows["validation"]}
    eligible = [r for r in tune if r["traded_days"] >= MIN_TRADES_FOR_SELECTION and "raw>=" not in r["label"]]
    ranked = sorted(eligible, key=lambda r: r["score"], reverse=True)
    top = ranked[:3]
    baselines = [r for r in tune if "raw>=" in r["label"]]

    print("\n### Tuning half - baselines\n" + md_table(baselines))
    print("\n### Tuning half - top calibrated variants (>= %d trades)\n" % MIN_TRADES_FOR_SELECTION + md_table(ranked[:8]))
    print("\n### Validation half - baselines\n" + md_table([val[r["label"]] for r in baselines]))
    print("\n### Validation half - the top-3 tuned variants\n" + md_table([val[r["label"]] for r in top]))
    print("\n### Validation half - all calibrated variants\n" + md_table(sorted(all_rows["validation"], key=lambda r: r["label"])))
    for key, diag in diagnostics.items():
        if key.endswith("/predictions"):
            continue
        print(f"\n### Diagnostics {key}: n={diag['n']} Brier raw={diag['brier_raw']} base-rate={diag['brier_base_rate']} "
              + " ".join(f"{m}{w}={diag[f'brier_{m}{w}']}" for m in METHODS for w in CAL_WINDOWS))
        print("confidence vs hit (raw):", diag["confidence_vs_hit_raw"])
        print("reliability raw:", [(r["bucket"], r["n"], r["realised_up"]) for r in diag["reliability_raw"] if r["n"]])
        print("reliability iso90:", [(r["bucket"], r["n"], r["realised_up"]) for r in diag["reliability_isotonic90"] if r["n"]])
        print("reliability platt90:", [(r["bucket"], r["n"], r["realised_up"]) for r in diag["reliability_platt90"] if r["n"]])

    runtime = time.perf_counter() - t0
    out = settings.models_dir / "idea02_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "idea": "02 probability calibration before thresholding",
        "protocol": {"eval_start": EVAL_START, "eval_end": EVAL_END, "mid": MID, "history_days": HISTORY_DAYS,
                     "cal_windows": CAL_WINDOWS, "min_cal_points": MIN_CAL_POINTS, "thresholds": THRESHOLDS,
                     "fast_params": FAST, "retrain_every_days": 14, "sim_config": sim_cfg.to_dict()},
        "tune": tune, "validation": all_rows["validation"], "tune_ranked_eligible": ranked, "top3_labels": [r["label"] for r in top],
        "diagnostics": diagnostics, "runtime_seconds": round(runtime, 1),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out} ({runtime:.0f}s)")


if __name__ == "__main__":
    main()
