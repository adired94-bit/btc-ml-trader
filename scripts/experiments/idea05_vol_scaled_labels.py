"""Idea 05 - volatility-scaled three-class labels for the daily direction forecast.

Instead of labelling every hourly bar UP/DOWN by the sign of its 24h-ahead return, the
return is compared with the *trailing* daily ATR% (shifted by one day, so bar t only sees
fully completed days):

    UP   if ret_24h >  +k * ATR%_daily(t)
    DOWN if ret_24h <  -k * ATR%_daily(t)
    FLAT otherwise

for k in {0.15, 0.3, 0.5, 0.8} (0.15 was added after a smoke run showed FLAT is already 44% of
bars at k = 0.3, because daily ATR% is ~2.5x the typical 24h move). The 3-class ``DirectionEnsemble`` (XGB + LGBM) is trained
walk-forward exactly like ``src.backtest.walk_forward.run_walk_forward`` (same days, same
``train_end = idx[i - horizon]`` rule, same retrain cadence) and a directional call is made
every day from P(UP) vs P(DOWN); a trade is taken only when max(P(UP), P(DOWN)) >= threshold.
The threshold is tuned on the older half of the window only. The baseline is the same loop
with k = 0 (plain sign labels, i.e. the existing binary walk-forward) and is cross-checked
against ``run_walk_forward`` to prove the loop reproduces the production evaluation.

    venv\\Scripts\\python.exe scripts\\experiments\\idea05_vol_scaled_labels.py
    venv\\Scripts\\python.exe scripts\\experiments\\idea05_vol_scaled_labels.py --quick   # smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.data.processor import DOWN, FLAT, UP  # noqa: E402
from src.logging_config import get_logger  # noqa: E402
from src.models.ensemble import DirectionEnsemble  # noqa: E402

logger = get_logger("idea05")
logging.getLogger("src.models.ensemble").setLevel(logging.WARNING)

# Fast, CPU-friendly settings shared by every run (baseline included).
FAST_XGB = {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_LGBM = {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2}
BASE_CFG = WF.WalkForwardConfig(
    eval_start_days_ago=365, eval_end_days_ago=90, retrain_every_days=14, threshold=0.55,
    xgb_params=FAST_XGB, lgbm_params=FAST_LGBM, reg_params=FAST_REG, feature_set="extended", label="baseline",
)
THRESHOLD_GRID: tuple[float | str, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, "argmax")  # "argmax": trade iff top class is not FLAT
MIN_TRADES_FOR_TUNING = 25  # a threshold that trades fewer days than this on the tune half is not selectable
OUT_JSON = settings.models_dir / "idea05_vol_scaled_labels.json"


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------


def causal_daily_atr_pct(ohlcv: pd.DataFrame) -> pd.Series:
    """Daily ATR(14) as a fraction of the close, known at the *previous* day's close.

    ``prepare_data`` ffills the daily ATR without shifting, so intraday bars of day D would see
    day D's full range. For labels every hourly bar must only use completed days -> shift(1).
    """
    daily = ohlcv.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    atr_hourly = atr.reindex(ohlcv.index, method="ffill")
    return atr_hourly / ohlcv["close"]


def make_labels(future_return: pd.Series, atr_pct: pd.Series, k: float) -> pd.Series:
    """0=DOWN, 1=FLAT, 2=UP (k > 0) or 0/2 by sign (k == 0). -1 where the label is unknown."""
    if k <= 0:
        lab = pd.Series(np.where(future_return >= 0, UP, DOWN), index=future_return.index)
        lab[future_return.isna()] = -1
        return lab.astype(int)
    band = k * atr_pct
    lab = pd.Series(FLAT, index=future_return.index, dtype=int)
    lab[future_return > band] = UP
    lab[future_return < -band] = DOWN
    lab[future_return.isna() | band.isna()] = -1
    return lab


# ----------------------------------------------------------------------
# Walk-forward loop that stores raw class probabilities per day
# ----------------------------------------------------------------------


@dataclass
class RawDay:
    day: pd.Timestamp
    i: int
    train_end: pd.Timestamp
    close: float
    actual_close: float
    actual_return_pct: float
    actual_label: int  # 3-class (or 2-class for k=0) label of the prediction bar
    p_down: float
    p_flat: float
    p_up: float
    daily_atr: float  # unshifted, same series the production loop feeds the trade simulator
    regimes: list[str]
    momentum_direction: str


def run_loop(
    data: WF.PreparedData, labels: pd.Series, cfg: WF.WalkForwardConfig, balance_classes: bool = True, tag: str = ""
) -> tuple[list[RawDay], dict[str, Any]]:
    """Mirror of ``run_walk_forward`` (same days, train_end, retrain cadence) without the trade layer."""
    idx = data.ohlcv.index
    start, end = WF.evaluation_days(idx, cfg)
    closes = WF.day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    pos = pd.Series(np.arange(len(idx)), index=idx)
    model: DirectionEnsemble | None = None
    feature_names: list[str] = []
    last_train_day: pd.Timestamp | None = None
    retrains, n_rows, class_share = 0, 0, {}
    raws: list[RawDay] = []
    t0 = time.perf_counter()
    for day in days:
        ts = closes[day]
        i = int(pos[ts])
        if i + cfg.horizon_bars >= len(idx):
            break
        train_end = idx[i - cfg.horizon_bars]  # labels fully realised at the close of day T
        if model is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
            mask = (data.features.index <= train_end) & (labels >= 0).to_numpy()
            X = data.features[mask].dropna()
            y = labels.loc[X.index]
            if len(X) < cfg.min_train_rows:
                raise ValueError(f"Only {len(X)} training rows at {train_end}")
            model = DirectionEnsemble(xgb_params=cfg.xgb_params, lgbm_params=cfg.lgbm_params, balance_classes=balance_classes)
            model.fit(X, y)
            feature_names = list(X.columns)
            last_train_day, retrains, n_rows = day, retrains + 1, len(X)
            class_share = {str(c): round(float((y == c).mean()), 4) for c in (DOWN, FLAT, UP)}
        row = data.features.iloc[[i]]
        if row.isna().any(axis=None):
            continue
        proba = model.predict_proba(row[feature_names])[0]
        close = float(data.ohlcv["close"].iloc[i])
        actual_close = float(data.ohlcv["close"].iloc[i + cfg.horizon_bars])
        raws.append(RawDay(
            day=day, i=i, train_end=train_end, close=close, actual_close=actual_close,
            actual_return_pct=(actual_close / close - 1) * 100, actual_label=int(labels.iloc[i]),
            p_down=float(proba[DOWN]), p_flat=float(proba[FLAT]), p_up=float(proba[UP]),
            daily_atr=float(data.daily_atr.iloc[i]) if np.isfinite(data.daily_atr.iloc[i]) else float("nan"),
            regimes=[f for f in WF.REGIME_FLAGS if bool(data.regimes.iloc[i][f])],
            momentum_direction="UP" if close >= float(data.ohlcv["close"].iloc[max(0, i - 720)]) else "DOWN",
        ))
    info = {"retrains": retrains, "last_train_rows": n_rows, "last_train_class_share": class_share, "runtime_s": round(time.perf_counter() - t0, 1)}
    logger.info("loop %s: %d days, %d retrains, %d rows, class share %s (%.0fs)", tag, len(raws), retrains, n_rows, class_share, info["runtime_s"])
    return raws, info


# ----------------------------------------------------------------------
# Trade layer + metrics (same simulator and metric code as production)
# ----------------------------------------------------------------------


def evaluate(raws: list[RawDay], threshold: float | str, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    """Directional call every day from P(UP) vs P(DOWN); trade when max(P(UP), P(DOWN)) >= threshold
    (or, for threshold == "argmax", when the most likely class is UP or DOWN rather than FLAT)."""
    equity = cfg.initial_equity
    records: list[WF.DayRecord] = []
    hits3, traded_hits3, argmax_flat = [], [], 0
    for r in raws:
        pred_dir = "UP" if r.p_up >= r.p_down else "DOWN"
        actual_dir = "UP" if r.actual_return_pct >= 0 else "DOWN"
        confidence = max(r.p_up, r.p_down)
        pred_label = UP if pred_dir == "UP" else DOWN
        hit3 = r.actual_label == pred_label  # FLAT-actual counts as a miss for a directional call
        hits3.append(hit3)
        argmax_flat += int(r.p_flat > confidence)
        gate = confidence > r.p_flat if threshold == "argmax" else confidence >= float(threshold)
        traded = gate and np.isfinite(r.daily_atr) and r.daily_atr > 0
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        if traded:
            side = "LONG" if pred_dir == "UP" else "SHORT"
            bars = DATA_OHLCV.iloc[r.i + 1: r.i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, r.daily_atr, equity, cfg)
            equity += pnl
            traded_hits3.append(hit3)
        records.append(WF.DayRecord(
            day=r.day.isoformat(), target_day=(r.day + pd.Timedelta(hours=cfg.horizon_bars)).isoformat(), train_end=r.train_end.isoformat(),
            close=r.close, actual_close=r.actual_close, actual_return_pct=float(r.actual_return_pct),
            predicted_direction=pred_dir, actual_direction=actual_dir, hit=pred_dir == actual_dir,
            prob_up=r.p_up, prob_down=r.p_down, confidence=float(confidence), traded=traded,
            target_price=r.close, predicted_return_pct=0.0,  # no return regressor in this experiment
            error_pct=float((r.close - r.actual_close) / r.actual_close * 100), abs_error_pct=float(abs(r.close - r.actual_close) / r.actual_close * 100),
            risk_level="MEDIUM", atr_pct=float(r.daily_atr / r.close) if np.isfinite(r.daily_atr) else 0.0, volume_ratio=1.0,
            regimes=r.regimes, trade_side=side, trade_pnl=float(pnl), trade_return_pct=float(ret_pct), trade_exit=exit_reason,
            equity=float(equity), momentum_direction=r.momentum_direction,
        ))
    m = WF.compute_metrics(records, cfg)
    m["threshold"] = threshold
    m["hit_3class"] = float(np.mean(hits3))
    m["traded_hit_3class"] = float(np.mean(traded_hits3)) if traded_hits3 else 0.0
    m["actual_flat_share"] = float(np.mean([r.actual_label == FLAT for r in raws]))
    m["argmax_flat_share"] = float(argmax_flat / len(raws))
    m["mean_p_flat"] = float(np.mean([r.p_flat for r in raws]))
    for key in ("mae_usd", "rmse_usd", "mae_pct", "rmse_pct", "median_abs_error_pct", "risk_level_distribution"):
        m.pop(key, None)
    return m


def score(m: dict[str, Any]) -> float:
    return WF._score(m)  # traded accuracy + 0.001 x return%


def fmt_thr(t: float | str) -> str:
    return t if isinstance(t, str) else f"{t:.2f}"


def pick_threshold(raws: list[RawDay], cfg: WF.WalkForwardConfig) -> tuple[float | str, list[dict[str, Any]]]:
    grid = [evaluate(raws, thr, cfg) for thr in THRESHOLD_GRID]
    eligible = [m for m in grid if m["traded_days"] >= MIN_TRADES_FOR_TUNING]
    best = max(eligible or grid, key=score)
    return best["threshold"], grid


def short(label: str, half: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label, "half": half, "threshold": m["threshold"], "days": m["days"],
        "hit": round(m["directional_accuracy"], 4), "hit_3class": round(m["hit_3class"], 4),
        "momentum_hit": round(m["momentum_30d_accuracy"], 4), "always_up": round(m["always_up_accuracy"], 4),
        "trades": m["traded_days"], "traded_hit": round(m["traded_directional_accuracy"], 4),
        "traded_hit_3class": round(m["traded_hit_3class"], 4), "return_pct": round(m["total_return_pct"], 2),
        "sharpe": round(m["sharpe_ratio"], 2), "max_dd": round(m["max_drawdown_pct"], 2),
        "actual_flat_share": round(m["actual_flat_share"], 3), "score": round(score(m), 4),
    }


def md_table(rows: list[dict[str, Any]]) -> str:
    head = ("| Config | Half | Thr | Days | Hit (sign) | Hit (3-class) | Mom 30d | Trades | Traded hit | Traded hit 3c | Return | Sharpe | Max DD |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    body = [
        f"| {r['label']} | {r['half']} | {fmt_thr(r['threshold'])} | {r['days']} | {r['hit']:.1%} | {r['hit_3class']:.1%} | {r['momentum_hit']:.1%} | "
        f"{r['trades']} | {r['traded_hit']:.1%} | {r['traded_hit_3class']:.1%} | {r['return_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% |"
        for r in rows
    ]
    return "\n".join([head, *body])


# ----------------------------------------------------------------------
# Campaign
# ----------------------------------------------------------------------

DATA_OHLCV: pd.DataFrame  # set in main(); the trade simulator needs the raw bars of day T+1


def main() -> None:
    global DATA_OHLCV
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke run: fewer trees, 60-day halves")
    parser.add_argument("--no-crosscheck", action="store_true", help="skip the run_walk_forward reproduction check")
    args = parser.parse_args()
    t_start = time.perf_counter()

    ohlcv = storage.get_ohlcv()
    DATA_OHLCV = ohlcv
    base = BASE_CFG
    if args.quick:
        base = base.copy(eval_start_days_ago=210, xgb_params={**FAST_XGB, "n_estimators": 30}, lgbm_params={**FAST_LGBM, "n_estimators": 30})
    span = base.eval_start_days_ago - base.eval_end_days_ago
    mid = base.eval_end_days_ago + span // 2
    tune_cfg, val_cfg = base.copy(eval_end_days_ago=mid), base.copy(eval_start_days_ago=mid)
    logger.info("Window: tune %d->%d days ago, validation %d->%d days ago; last bar %s", base.eval_start_days_ago, mid, mid, base.eval_end_days_ago, ohlcv.index[-1])

    data = WF.prepare_data(ohlcv, base.horizon_bars, base.feature_set)
    atr_pct = causal_daily_atr_pct(ohlcv)
    label_sets = {0.0: make_labels(data.future_return, atr_pct, 0.0)}
    for k in (0.15, 0.3, 0.5, 0.8):
        label_sets[k] = make_labels(data.future_return, atr_pct, k)
    label_stats = {
        str(k): {str(c): round(float((lab[lab >= 0] == c).mean()), 4) for c in (DOWN, FLAT, UP)} for k, lab in label_sets.items()
    }
    logger.info("Label class shares over all realised bars: %s", label_stats)

    # Experiments: (label, k, balance_classes)
    exps = [("baseline_binary", 0.0, True), ("k0.15_atr", 0.15, True), ("k0.3_atr", 0.3, True), ("k0.5_atr", 0.5, True), ("k0.8_atr", 0.8, True)]

    results: dict[str, Any] = {"window": {"tune": [base.eval_start_days_ago, mid], "validation": [mid, base.eval_end_days_ago], "last_bar": str(ohlcv.index[-1])},
                               "config": base.to_dict(), "label_stats": label_stats, "threshold_grid": list(THRESHOLD_GRID), "experiments": {}}
    tune_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    tune_raws: dict[str, list[RawDay]] = {}

    # --- tune half: run every experiment, sweep the threshold ---
    for label, k, bal in exps:
        raws, info = run_loop(data, label_sets[k], tune_cfg, balance_classes=bal, tag=f"{label}/tune")
        tune_raws[label] = raws
        thr, grid = pick_threshold(raws, tune_cfg)
        results["experiments"][label] = {"k": k, "balance_classes": bal, "tune_info": info, "tune_grid": grid, "chosen_threshold": thr}
        rows = [short(label, "tune", next(g for g in grid if g["threshold"] == thr))]
        if label == "baseline_binary":  # also report the production threshold for reference
            rows.insert(0, short(label + "@0.55", "tune", next(g for g in grid if g["threshold"] == 0.55)))
        for r in rows:
            tune_rows.append(r)
            print(f"[tune] {r['label']:<22} thr {fmt_thr(r['threshold'])} hit {r['hit']:.1%} (3c {r['hit_3class']:.1%}) traded {r['traded_hit']:.1%} ({r['trades']}) ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:.2f} flat_actual {r['actual_flat_share']:.0%}", flush=True)
        print("       grid: " + ", ".join(f"{fmt_thr(g['threshold'])}->{g['traded_directional_accuracy']:.1%}/{g['traded_days']}" for g in grid), flush=True)

    # --- cross-check: the k=0 loop must reproduce run_walk_forward at threshold 0.55 ---
    if not args.no_crosscheck:
        ref = WF.run_walk_forward(ohlcv, tune_cfg, data, verbose=False).metrics
        mine = evaluate(tune_raws["baseline_binary"], 0.55, tune_cfg)
        results["crosscheck"] = {"run_walk_forward": {kk: ref[kk] for kk in ("days", "directional_accuracy", "traded_days", "traded_directional_accuracy", "total_return_pct")},
                                 "this_loop": {kk: mine[kk] for kk in ("days", "directional_accuracy", "traded_days", "traded_directional_accuracy", "total_return_pct")}}
        print(f"[check] run_walk_forward: hit {ref['directional_accuracy']:.4f} traded {ref['traded_directional_accuracy']:.4f} ({ref['traded_days']}) ret {ref['total_return_pct']:+.2f}% | "
              f"this loop: hit {mine['directional_accuracy']:.4f} traded {mine['traded_directional_accuracy']:.4f} ({mine['traded_days']}) ret {mine['total_return_pct']:+.2f}%", flush=True)

    # --- choose the idea config on the tune half BEFORE touching validation ---
    idea_tune = [r for r in tune_rows if r["label"] != "baseline_binary" and "@" not in r["label"]]
    chosen = max(idea_tune, key=lambda r: r["score"])["label"]
    results["chosen_on_tune"] = chosen
    print(f"[tune] chosen idea config by tune score: {chosen} (thr {fmt_thr(results['experiments'][chosen]['chosen_threshold'])})", flush=True)

    # --- validation half: every experiment at its tune-chosen threshold (pre-registered: all are reported) ---
    for label, k, bal in exps:
        raws, info = run_loop(data, label_sets[k], val_cfg, balance_classes=bal, tag=f"{label}/val")
        thr = results["experiments"][label]["chosen_threshold"]
        grid = [evaluate(raws, t, val_cfg) for t in THRESHOLD_GRID]
        results["experiments"][label]["val_info"] = info
        results["experiments"][label]["val_grid"] = grid
        rows = [short(label, "validation", next(g for g in grid if g["threshold"] == thr))]
        if label == "baseline_binary":
            rows.insert(0, short(label + "@0.55", "validation", next(g for g in grid if g["threshold"] == 0.55)))
        for r in rows:
            val_rows.append(r)
            print(f"[val ] {r['label']:<22} thr {fmt_thr(r['threshold'])} hit {r['hit']:.1%} (3c {r['hit_3class']:.1%}) traded {r['traded_hit']:.1%} ({r['trades']}) ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:.2f}", flush=True)
        print("       grid: " + ", ".join(f"{fmt_thr(g['threshold'])}->{g['traded_directional_accuracy']:.1%}/{g['traded_days']}" for g in grid), flush=True)

    results["tune_rows"], results["val_rows"] = tune_rows, val_rows
    results["runtime_s"] = round(time.perf_counter() - t_start, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("\n### Tune half\n" + md_table(tune_rows) + "\n\n### Validation half\n" + md_table(val_rows))
    print(f"\nchosen on tune: {chosen}; runtime {results['runtime_s']:.0f}s; saved {OUT_JSON}")


if __name__ == "__main__":
    main()
