"""Idea 01 - Meta-labeling on top of the daily direction ensemble.

Primary model: the walk-forward ``DailyModels`` ensemble (XGB + LGBM, extended features),
retrained every 14 days exactly as in ``src/backtest/walk_forward.py``.

Secondary ("meta") model: a small LightGBM binary classifier whose label is
"the primary model was right". It is trained only on *out-of-sample* primary
predictions: whenever the primary is retrained at day D_k (train_end_k), the
previous primary model (trained up to train_end_{k-1}) scores every hourly bar in
(train_end_{k-1}, train_end_k].  Those bars have a fully realised 24-bar label at
the close of D_k, and none of them was seen by the model that scored them, so the
pool is out-of-sample and free of look-ahead.  The meta model is fit on that pool
(features + p_up/confidence/predicted direction/regime flags) at every retrain and
is asked P(right) for the day-close bar of every evaluation day.

Trading rule: trade only when P(right) >= META_THRESHOLD (0.55), optionally combined
with the primary confidence gate.  Everything is compared against the plain primary
(confidence >= 0.55) computed on the *same* primary predictions, plus the official
``run_walk_forward`` baseline on each half as a cross-check.

Tune half: eval days 365 -> 227 days ago.  Validation half: 227 -> 90.  The primary
loop is warmed up 13 retrain cycles (182 days) before the tune window so the meta
pool is non-empty from the first evaluation day.

    venv\\Scripts\\python.exe scripts\\experiments\\idea01_meta_labeling.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea01_meta_labeling")

EVAL_START, EVAL_END = 365, 90
RETRAIN_EVERY = 14
WARMUP_CYCLES = 13  # primary OOS predictions collected before the tune window (13 * 14 = 182 days)
PRIMARY_THRESHOLD = 0.55
META_THRESHOLD = 0.55
META_SWEEP = (0.50, 0.52, 0.55, 0.58, 0.60)

FAST_XGB = {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_LGBM = {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2}
META_PARAMS: dict[str, Any] = {
    "n_estimators": 80, "num_leaves": 7, "learning_rate": 0.05, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "min_child_samples": 100, "reg_lambda": 5.0, "objective": "binary",
    "verbose": -1, "n_jobs": 2, "random_state": settings.random_state,
}
META_EXTRA = ["p_up", "confidence", "pred_up", "r_hat", *WF.REGIME_FLAGS]


def primary_config() -> WF.WalkForwardConfig:
    return WF.WalkForwardConfig(
        eval_start_days_ago=EVAL_START, eval_end_days_ago=EVAL_END, retrain_every_days=RETRAIN_EVERY,
        threshold=PRIMARY_THRESHOLD, feature_set="extended", xgb_params=FAST_XGB, lgbm_params=FAST_LGBM,
        reg_params=FAST_REG, label="ext+ens_fast",
    )


@dataclass
class DayPred:
    """One evaluation day: primary + meta outputs, realised outcome, everything needed to simulate a trade."""

    day: pd.Timestamp
    i: int  # position of the day-close bar in the hourly index
    p_up: float
    p_down: float
    confidence: float
    pred_dir: str
    actual_dir: str
    hit: bool
    p_right_hourly: float  # meta model trained on hourly OOS rows
    p_right_daily: float  # meta model trained on day-close OOS rows only
    meta_rows_hourly: int
    meta_rows_daily: int
    daily_atr: float


# ----------------------------------------------------------------------
# Meta-model helpers
# ----------------------------------------------------------------------


def score_chunk(models: WF.DailyModels, data: WF.PreparedData, lo: int, hi: int) -> pd.DataFrame:
    """Out-of-sample primary scores for hourly bars idx[lo:hi] (exclusive hi) plus the realised meta label."""
    X = data.features.iloc[lo:hi]
    valid = ~X.isna().any(axis=1) & (data.direction.iloc[lo:hi] >= 0).to_numpy()
    X = X[valid]
    if X.empty:
        return pd.DataFrame()
    proba = models.direction.predict_proba(X)
    r_hat = models.regressor.predict(X.to_numpy(dtype=np.float32))
    p_up, p_down = proba[:, WF.UP], proba[:, WF.DOWN]
    pred_up = p_up >= p_down
    actual_up = (data.direction.loc[X.index] == WF.UP).to_numpy()
    out = X.copy()
    out["p_up"] = p_up
    out["confidence"] = np.maximum(p_up, p_down)
    out["pred_up"] = pred_up.astype(float)
    out["r_hat"] = r_hat
    for flag in WF.REGIME_FLAGS:
        out[flag] = data.regimes.loc[X.index, flag].astype(float).to_numpy()
    out["meta_label"] = (pred_up == actual_up).astype(int)
    return out


def fit_meta(pool: pd.DataFrame) -> lgb.LGBMClassifier | None:
    if len(pool) < 300 or pool["meta_label"].nunique() < 2:
        return None
    X = pool.drop(columns=["meta_label"])
    clf = lgb.LGBMClassifier(**META_PARAMS)
    clf.fit(X.to_numpy(dtype=np.float32), pool["meta_label"].to_numpy())
    return clf


def meta_row(models: WF.DailyModels, data: WF.PreparedData, i: int, p_up: float, p_down: float, r_hat: float) -> np.ndarray:
    row = data.features.iloc[i].to_dict()
    row.update({"p_up": p_up, "confidence": max(p_up, p_down), "pred_up": float(p_up >= p_down), "r_hat": r_hat})
    for flag in WF.REGIME_FLAGS:
        row[flag] = float(data.regimes.iloc[i][flag])
    cols = [*models.feature_names, *META_EXTRA]
    return np.array([[row[c] for c in cols]], dtype=np.float32)


# ----------------------------------------------------------------------
# Primary + meta walk-forward
# ----------------------------------------------------------------------


def run_primary_with_meta(ohlcv: pd.DataFrame, data: WF.PreparedData, cfg: WF.WalkForwardConfig, verbose: bool = True) -> tuple[list[DayPred], dict[str, Any]]:
    idx = ohlcv.index
    warm_cfg = cfg.copy(eval_start_days_ago=cfg.eval_start_days_ago + WARMUP_CYCLES * cfg.retrain_every_days)
    start, end = WF.evaluation_days(idx, warm_cfg)
    closes = WF.day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    pos = pd.Series(np.arange(len(idx)), index=idx)
    day_close_index = pd.DatetimeIndex(closes.to_numpy())

    models: WF.DailyModels | None = None
    meta_h: lgb.LGBMClassifier | None = None
    meta_d: lgb.LGBMClassifier | None = None
    pool_h: list[pd.DataFrame] = []
    pool_d: list[pd.DataFrame] = []
    n_pool_h = n_pool_d = 0
    last_train_day: pd.Timestamp | None = None
    prev_train_end_i: int | None = None
    retrains = 0
    preds: list[DayPred] = []
    auc_log: list[dict[str, Any]] = []

    for day in days:
        ts = closes[day]
        i = int(pos[ts])
        if i + cfg.horizon_bars >= len(idx):
            break
        train_end_i = i - cfg.horizon_bars
        train_end = idx[train_end_i]
        if models is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
            if models is not None and prev_train_end_i is not None:
                # Bars (prev_train_end, train_end] are OOS for the *previous* primary and fully labelled now.
                chunk = score_chunk(models, data, prev_train_end_i + 1, train_end_i + 1)
                if not chunk.empty:
                    pool_h.append(chunk)
                    daily_mask = chunk.index.isin(day_close_index)
                    if daily_mask.any():
                        pool_d.append(chunk[daily_mask])
                    pool_hf = pd.concat(pool_h)
                    pool_df = pd.concat(pool_d) if pool_d else pd.DataFrame()
                    n_pool_h, n_pool_d = len(pool_hf), len(pool_df)
                    # honest running OOS AUC of the *previous* meta model on the new chunk
                    if meta_h is not None and chunk["meta_label"].nunique() > 1:
                        p = meta_h.predict_proba(chunk.drop(columns=["meta_label"]).to_numpy(dtype=np.float32))[:, 1]
                        auc_log.append({"day": day.date().isoformat(), "rows": int(len(chunk)), "auc": float(roc_auc_score(chunk["meta_label"], p)),
                                        "base_rate": float(chunk["meta_label"].mean())})
                    meta_h = fit_meta(pool_hf)
                    meta_d = fit_meta(pool_df) if len(pool_df) else None
            models = WF.DailyModels(cfg).fit(data, train_end)
            prev_train_end_i = train_end_i
            last_train_day = day
            retrains += 1
            if verbose:
                logger.info("Retrain %d at %s: primary rows %d, meta pool hourly %d / daily %d", retrains, day.date(), models.n_rows, n_pool_h, n_pool_d)
        row = data.features.iloc[[i]]
        if row.isna().any(axis=None):
            continue
        p_up, p_down, r_hat = models.predict(row)
        close = float(data.ohlcv["close"].iloc[i])
        actual_close = float(data.ohlcv["close"].iloc[i + cfg.horizon_bars])
        actual_ret = actual_close / close - 1
        pred_dir = "UP" if p_up >= p_down else "DOWN"
        actual_dir = "UP" if actual_ret >= 0 else "DOWN"
        mrow = meta_row(models, data, i, p_up, p_down, r_hat)
        pr_h = float(meta_h.predict_proba(mrow)[0, 1]) if meta_h is not None else float("nan")
        pr_d = float(meta_d.predict_proba(mrow)[0, 1]) if meta_d is not None else float("nan")
        atr = float(data.daily_atr.iloc[i]) if np.isfinite(data.daily_atr.iloc[i]) else float("nan")
        preds.append(DayPred(day, i, float(p_up), float(p_down), float(max(p_up, p_down)), pred_dir, actual_dir, pred_dir == actual_dir,
                             pr_h, pr_d, n_pool_h, n_pool_d, atr))
    return preds, {"retrains": retrains, "meta_oos_auc_by_chunk": auc_log}


# ----------------------------------------------------------------------
# Evaluation of a trade filter on a set of days (same trade simulation as walk_forward)
# ----------------------------------------------------------------------


def evaluate(preds: list[DayPred], trade_mask: list[bool], data: WF.PreparedData, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    equity = cfg.initial_equity
    rets, hits, traded_hits, eq_curve = [], [], [], [cfg.initial_equity]
    for p, trade in zip(preds, trade_mask):
        hits.append(p.hit)
        ret_pct = 0.0
        if trade and np.isfinite(p.daily_atr) and p.daily_atr > 0:
            side = "LONG" if p.pred_dir == "UP" else "SHORT"
            bars = data.ohlcv.iloc[p.i + 1: p.i + 1 + cfg.horizon_bars]
            pnl, ret_pct, _, _, _, _ = WF.simulate_day_trade(side, bars, p.daily_atr, equity, cfg)
            equity += pnl
            traded_hits.append(p.hit)
        rets.append(ret_pct / 100)
        eq_curve.append(equity)
    r = pd.Series(rets)
    eq = pd.Series(eq_curve)
    return {
        "days": len(preds),
        "accuracy": float(np.mean(hits)) if hits else 0.0,
        "traded_days": len(traded_hits),
        "coverage": float(len(traded_hits) / len(preds)) if preds else 0.0,
        "traded_accuracy": float(np.mean(traded_hits)) if traded_hits else 0.0,
        "total_return_pct": float((equity / cfg.initial_equity - 1) * 100),
        "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
        "max_drawdown_pct": float(-(eq / eq.cummax() - 1).min() * 100),
    }


def official_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "days": m["days"], "accuracy": m["directional_accuracy"], "traded_days": m["traded_days"], "coverage": m["coverage"],
        "traded_accuracy": m["traded_directional_accuracy"], "total_return_pct": m["total_return_pct"],
        "sharpe": m["sharpe_ratio"], "max_drawdown_pct": m["max_drawdown_pct"],
    }


def md_table(rows: list[tuple[str, str, dict[str, Any]]]) -> str:
    head = "| Variant | Half | Days | Acc (all) | Traded | Traded acc | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|"
    body = [
        f"| {label} | {half} | {m['days']} | {m['accuracy']:.1%} | {m['traded_days']} ({m['coverage']:.0%}) | {m['traded_accuracy']:.1%} | "
        f"{m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.2f}% |"
        for label, half, m in rows
    ]
    return "\n".join([head, *body])


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-official", action="store_true", help="skip the run_walk_forward cross-check baselines")
    args = parser.parse_args()
    t0 = time.perf_counter()

    ohlcv = storage.get_ohlcv()
    cfg = primary_config()
    data = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    logger.info("Data: %d candles %s -> %s, %d features", len(ohlcv), ohlcv.index[0], ohlcv.index[-1], data.features.shape[1])

    preds, info = run_primary_with_meta(ohlcv, data, cfg)
    t_primary = time.perf_counter() - t0
    logger.info("Primary+meta loop done: %d prediction days, %d retrains, %.0fs", len(preds), info["retrains"], t_primary)

    start, end = WF.evaluation_days(ohlcv.index, cfg)
    mid = (ohlcv.index[-1] - pd.Timedelta(days=EVAL_END + (EVAL_START - EVAL_END) // 2)).floor("D")
    halves = {
        "tune": [p for p in preds if start <= p.day < mid],
        "validation": [p for p in preds if mid <= p.day <= end],
    }
    logger.info("Tune %s..%s (%d days) | validation %s..%s (%d days)", start.date(), mid.date(), len(halves["tune"]), mid.date(), end.date(), len(halves["validation"]))

    def variants(ps: list[DayPred], meta_thr: float) -> dict[str, list[bool]]:
        return {
            "baseline (conf>=0.55)": [p.confidence >= PRIMARY_THRESHOLD for p in ps],
            f"meta_hourly>={meta_thr:.2f}": [np.isfinite(p.p_right_hourly) and p.p_right_hourly >= meta_thr for p in ps],
            f"meta_hourly>={meta_thr:.2f} & conf>=0.55": [np.isfinite(p.p_right_hourly) and p.p_right_hourly >= meta_thr and p.confidence >= PRIMARY_THRESHOLD for p in ps],
            f"meta_daily>={meta_thr:.2f}": [np.isfinite(p.p_right_daily) and p.p_right_daily >= meta_thr for p in ps],
        }

    results: dict[str, Any] = {"tune": {}, "validation": {}}
    table_rows: list[tuple[str, str, dict[str, Any]]] = []
    for half, ps in halves.items():
        for label, mask in variants(ps, META_THRESHOLD).items():
            m = evaluate(ps, mask, data, cfg)
            results[half][label] = m
            table_rows.append((label, half, m))

    # Threshold sweep: selected on tune only, then reported on validation.
    sweep: dict[str, Any] = {}
    for thr in META_SWEEP:
        for gate in ("meta_only", "meta_and_conf"):
            key = f"{gate}@{thr:.2f}"
            sweep[key] = {}
            for half, ps in halves.items():
                mask = [np.isfinite(p.p_right_hourly) and p.p_right_hourly >= thr and (gate == "meta_only" or p.confidence >= PRIMARY_THRESHOLD) for p in ps]
                sweep[key][half] = evaluate(ps, mask, data, cfg)
    eligible = {k: v for k, v in sweep.items() if v["tune"]["traded_days"] >= 30}
    best_key = max(eligible, key=lambda k: eligible[k]["tune"]["traded_accuracy"] + 0.001 * eligible[k]["tune"]["total_return_pct"]) if eligible else None
    if best_key:
        table_rows.append((f"sweep-best-on-tune [{best_key}]", "tune", sweep[best_key]["tune"]))
        table_rows.append((f"sweep-best-on-tune [{best_key}]", "validation", sweep[best_key]["validation"]))

    # Meta calibration diagnostics: accuracy of the primary by P(right) bucket, per half.
    calib: dict[str, Any] = {}
    for half, ps in halves.items():
        df = pd.DataFrame([{"p_right": p.p_right_hourly, "hit": p.hit, "conf": p.confidence} for p in ps]).dropna()
        buckets = pd.cut(df["p_right"], [0, 0.45, 0.5, 0.55, 0.6, 1.0], include_lowest=True)
        grp = df.groupby(buckets, observed=True)["hit"].agg(["count", "mean"])
        calib[half] = {str(k): {"days": int(v["count"]), "accuracy": float(v["mean"])} for k, v in grp.iterrows()}
        calib[half]["auc_p_right_vs_hit"] = float(roc_auc_score(df["hit"], df["p_right"])) if df["hit"].nunique() > 1 else float("nan")
        calib[half]["auc_conf_vs_hit"] = float(roc_auc_score(df["hit"], df["conf"])) if df["hit"].nunique() > 1 else float("nan")
        calib[half]["p_right_mean"] = float(df["p_right"].mean())
        calib[half]["p_right_std"] = float(df["p_right"].std())

    official: dict[str, Any] = {}
    if not args.skip_official:
        tune_cfg = cfg.copy(eval_end_days_ago=EVAL_END + (EVAL_START - EVAL_END) // 2, label="official_tune")
        val_cfg = cfg.copy(eval_start_days_ago=EVAL_END + (EVAL_START - EVAL_END) // 2, label="official_validation")
        for half, c in (("tune", tune_cfg), ("validation", val_cfg)):
            res = WF.run_walk_forward(ohlcv, c, data, verbose=False)
            official[half] = official_metrics(res.metrics)
            table_rows.append(("official run_walk_forward baseline", half, official[half]))

    runtime = time.perf_counter() - t0
    table = md_table(table_rows)
    print("\n" + table)
    print("\nMeta OOS AUC per 14-day chunk (previous meta model scoring the freshly labelled chunk):")
    for a in info["meta_oos_auc_by_chunk"]:
        print(f"  {a['day']}: rows={a['rows']} auc={a['auc']:.3f} base_rate={a['base_rate']:.3f}")
    print("\nCalibration of P(right) on evaluation days:")
    print(json.dumps(calib, indent=2))
    print(f"\nRuntime {runtime:.0f}s (primary+meta loop {t_primary:.0f}s)")

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "meta_params": META_PARAMS,
        "meta_threshold": META_THRESHOLD,
        "warmup_cycles": WARMUP_CYCLES,
        "split": {"tune_start": str(start.date()), "mid": str(mid.date()), "end": str(end.date())},
        "results": results,
        "sweep": sweep,
        "sweep_best_on_tune": best_key,
        "official_baseline": official,
        "calibration": calib,
        "meta_oos_auc_by_chunk": info["meta_oos_auc_by_chunk"],
        "retrains": info["retrains"],
        "runtime_seconds": runtime,
        "markdown_table": table,
        "days": [{**asdict(p), "day": p.day.isoformat()} for p in preds if start <= p.day <= end],
    }
    out = settings.models_dir / "idea01_meta_labeling.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
