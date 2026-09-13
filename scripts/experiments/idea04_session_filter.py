"""Idea 04 - *when* to predict, not just what.

Keeps the extended-features XGB+LGBM daily model untouched and varies only the
decision time and a calendar filter:

* decision hour H (UTC): the prediction is made at H:00 using the hourly bar that
  closes at H:00 (open time (H-1) % 24) and targets the close 24 bars later.
  H = 0 is exactly the existing walk-forward (last bar of the UTC day).
* calendar filters applied to *trading only* (the model still predicts every day):
  ``no_weekend``  - do not trade when the 24-h window opens on a Saturday/Sunday,
  ``no_bigmove``  - do not trade after |trailing 24-h return| > 3 %,
  ``both``        - both filters.

Zero look-ahead: the ensemble is retrained every ``retrain_every_days`` days at the
23:00 bar (same schedule as ``run_walk_forward``) on bars whose 24-bar label is
already realised at that time; every decision hour reuses the latest such model,
which therefore never saw data beyond the decision bar.

Tune on the older half (365 -> 227 days ago), validate on the newer half
(227 -> 90 days ago). Baseline = decision hour 0, no filter, which is checked
against ``walk_forward.run_walk_forward`` on the same window.

    venv\\Scripts\\python.exe scripts\\experiments\\idea04_session_filter.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

logger = get_logger("idea04")

FAST_XGB = {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_LGBM = {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2}
BIG_MOVE_PCT = 3.0
FILTERS = ("none", "no_weekend", "no_bigmove", "both")
STORY_HOURS = (0, 8, 13, 14, 16)  # 00 = UTC day close, 08 = Asia close, 13/14 = around US open (13:30), 16
MIN_TRADES = 40


def base_config(**changes: Any) -> WF.WalkForwardConfig:
    return WF.WalkForwardConfig(
        feature_set="extended", model="ensemble", retrain_every_days=14, label="ext+ens_fast",
        xgb_params=dict(FAST_XGB), lgbm_params=dict(FAST_LGBM), reg_params=dict(FAST_REG),
    ).copy(**changes)


# ----------------------------------------------------------------------
# One pass over a window: fit on the hour-0 schedule, predict at every hour
# ----------------------------------------------------------------------


def predict_all_hours(data: WF.PreparedData, cfg: WF.WalkForwardConfig, hours: list[int]) -> tuple[pd.DataFrame, int]:
    """Return one row per (decision bar, hour) with the raw prediction and everything needed to trade later."""
    idx = data.ohlcv.index
    start, end = WF.evaluation_days(idx, cfg)
    closes = WF.day_close_bars(idx)
    days = [d for d in closes.index if start <= d <= end]
    pos = pd.Series(np.arange(len(idx)), index=idx)
    h = cfg.horizon_bars
    close = data.ohlcv["close"]

    # events sorted in time: (timestamp, hour). Hour 0 uses the 23:00 bar of day D (as run_walk_forward does).
    events: list[tuple[pd.Timestamp, int, pd.Timestamp]] = []
    for day in days:
        for H in hours:
            ts = closes[day] if H == 0 else day + pd.Timedelta(hours=H - 1)
            if ts in pos.index:
                events.append((ts, H, day))
    events.sort(key=lambda e: (e[0], e[1]))

    models: WF.DailyModels | None = None
    last_train_day: pd.Timestamp | None = None
    retrains = 0
    rows: list[dict[str, Any]] = []
    for ts, H, day in events:
        i = int(pos[ts])
        if i + h >= len(idx):
            continue
        if H == 0:  # retrain schedule identical to run_walk_forward
            if models is None or last_train_day is None or (day - last_train_day).days >= cfg.retrain_every_days:
                train_end = idx[i - h]
                models = WF.DailyModels(cfg).fit(data, train_end)
                last_train_day = day
                retrains += 1
                logger.info("Retrained at %s on %d rows (train_end %s)", day.date(), models.n_rows, train_end)
        if models is None:
            continue  # hours > 0 on the very first day have no model yet
        assert models.train_end is not None and models.train_end + pd.Timedelta(hours=h) <= ts, "look-ahead guard"
        row = data.features.iloc[[i]]
        if row.isna().any(axis=None):
            continue
        p_up, p_down, r_hat = models.predict(row)
        c0, c1 = float(close.iloc[i]), float(close.iloc[i + h])
        actual_ret = (c1 / c0 - 1) * 100
        prev_ret = (c0 / float(close.iloc[max(0, i - h)]) - 1) * 100
        atr = float(data.daily_atr.iloc[i])
        rows.append({
            "hour": H, "i": i, "ts": ts, "day": day, "window_open": idx[i + 1],
            "weekday": int(idx[i + 1].dayofweek), "close": c0, "actual_close": c1, "actual_return_pct": actual_ret,
            "prev_return_pct": prev_ret, "prob_up": p_up, "prob_down": p_down, "confidence": max(p_up, p_down),
            "predicted_direction": "UP" if p_up >= p_down else "DOWN", "actual_direction": "UP" if actual_ret >= 0 else "DOWN",
            "target_price": c0 * (1 + r_hat), "predicted_return_pct": r_hat * 100,
            "daily_atr": atr if np.isfinite(atr) else float("nan"),
            "train_end": models.train_end, "momentum_direction": "UP" if c0 >= float(close.iloc[max(0, i - 720)]) else "DOWN",
        })
    df = pd.DataFrame(rows)
    df["hit"] = df["predicted_direction"] == df["actual_direction"]
    return df, retrains


def passes(df: pd.DataFrame, flt: str) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if flt in ("no_weekend", "both"):
        ok &= ~df["weekday"].isin([5, 6])
    if flt in ("no_bigmove", "both"):
        ok &= df["prev_return_pct"].abs() <= BIG_MOVE_PCT
    return ok


def evaluate(df_hour: pd.DataFrame, flt: str, data: WF.PreparedData, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    """Trade one decision hour with a calendar filter using walk_forward.simulate_day_trade and compute_metrics."""
    ok = passes(df_hour, flt)
    equity = cfg.initial_equity
    records: list[WF.DayRecord] = []
    for (_, r), allowed in zip(df_hour.iterrows(), ok):
        traded = bool(allowed) and r["confidence"] >= cfg.threshold and np.isfinite(r["daily_atr"]) and r["daily_atr"] > 0
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        if traded:
            side = "LONG" if r["predicted_direction"] == "UP" else "SHORT"
            i = int(r["i"])
            bars = data.ohlcv.iloc[i + 1: i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, float(r["daily_atr"]), equity, cfg)
            equity += pnl
        records.append(WF.DayRecord(
            day=r["ts"].isoformat(), target_day=(r["ts"] + pd.Timedelta(hours=cfg.horizon_bars)).isoformat(),
            train_end=r["train_end"].isoformat(), close=r["close"], actual_close=r["actual_close"],
            actual_return_pct=r["actual_return_pct"], predicted_direction=r["predicted_direction"],
            actual_direction=r["actual_direction"], hit=bool(r["hit"]), prob_up=r["prob_up"], prob_down=r["prob_down"],
            confidence=r["confidence"], traded=traded, target_price=r["target_price"],
            predicted_return_pct=r["predicted_return_pct"],
            error_pct=(r["target_price"] - r["actual_close"]) / r["actual_close"] * 100,
            abs_error_pct=abs(r["target_price"] - r["actual_close"]) / r["actual_close"] * 100,
            risk_level="MEDIUM", atr_pct=float("nan"), volume_ratio=float("nan"), regimes=[], trade_side=side,
            trade_pnl=pnl, trade_return_pct=ret_pct, trade_exit=exit_reason, equity=equity,
            momentum_direction=r["momentum_direction"],
        ))
    m = WF.compute_metrics(records, cfg)
    return {
        "days": m["days"], "hit": round(m["directional_accuracy"], 4), "traded": m["traded_days"],
        "traded_hit": round(m["traded_directional_accuracy"], 4), "return_pct": round(m["total_return_pct"], 2),
        "sharpe": round(m["sharpe_ratio"], 2), "max_dd": round(m["max_drawdown_pct"], 2),
        "always_up": round(m["always_up_accuracy"], 4), "momentum_hit": round(m["momentum_30d_accuracy"], 4),
        "filtered_out": int((~ok).sum()),
    }


def grid(df: pd.DataFrame, data: WF.PreparedData, cfg: WF.WalkForwardConfig, hours: list[int]) -> list[dict[str, Any]]:
    out = []
    for H in hours:
        sub = df[df["hour"] == H].sort_values("ts")
        if sub.empty:
            continue
        for flt in FILTERS:
            out.append({"hour": H, "filter": flt, **evaluate(sub, flt, data, cfg)})
    return out


def by_weekday(df: pd.DataFrame, cfg: WF.WalkForwardConfig, hour: int) -> list[dict[str, Any]]:
    sub = df[df["hour"] == hour]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for wd, g in sub.groupby("weekday"):
        conf = g[g["confidence"] >= cfg.threshold]
        rows.append({
            "weekday": names[int(wd)], "days": int(len(g)), "hit": round(float(g["hit"].mean()), 4),
            "confident": int(len(conf)), "confident_hit": round(float(conf["hit"].mean()), 4) if len(conf) else None,
            "mean_abs_return": round(float(g["actual_return_pct"].abs().mean()), 3),
        })
    return rows


def after_bigmove(df: pd.DataFrame, cfg: WF.WalkForwardConfig, hour: int) -> dict[str, Any]:
    sub = df[df["hour"] == hour]
    big = sub[sub["prev_return_pct"].abs() > BIG_MOVE_PCT]
    calm = sub[sub["prev_return_pct"].abs() <= BIG_MOVE_PCT]
    conf_big, conf_calm = big[big["confidence"] >= cfg.threshold], calm[calm["confidence"] >= cfg.threshold]
    return {
        "big_days": int(len(big)), "big_hit": round(float(big["hit"].mean()), 4) if len(big) else None,
        "big_confident": int(len(conf_big)), "big_confident_hit": round(float(conf_big["hit"].mean()), 4) if len(conf_big) else None,
        "calm_days": int(len(calm)), "calm_hit": round(float(calm["hit"].mean()), 4),
        "calm_confident": int(len(conf_calm)), "calm_confident_hit": round(float(conf_calm["hit"].mean()), 4) if len(conf_calm) else None,
    }


def fmt(r: dict[str, Any]) -> str:
    return (f"H{r['hour']:02d} {r['filter']:<11} days {r['days']:>3} hit {r['hit']:.1%} traded {r['traded']:>3} "
            f"traded_hit {r['traded_hit']:.1%} ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:+.2f} dd {r['max_dd']:.2f}%")


def pick(rows: list[dict[str, Any]], hours: tuple[int, ...] | None = None, exclude_baseline: bool = True) -> dict[str, Any]:
    cands = [r for r in rows if r["traded"] >= MIN_TRADES and (hours is None or r["hour"] in hours)]
    if exclude_baseline:
        cands = [r for r in cands if not (r["hour"] == 0 and r["filter"] == "none")] or cands
    return max(cands, key=lambda r: (r["traded_hit"], r["sharpe"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", default="all", help="'all' (0-23) or comma list, e.g. 0,8,13,14,16")
    parser.add_argument("--no-crosscheck", action="store_true", help="skip the run_walk_forward equality check")
    args = parser.parse_args()
    hours = list(range(24)) if args.hours == "all" else [int(x) for x in args.hours.split(",")]
    if 0 not in hours:
        hours = [0, *hours]
    t0 = time.perf_counter()

    ohlcv = storage.get_ohlcv()
    base = base_config()
    span = base.eval_start_days_ago - base.eval_end_days_ago
    mid = base.eval_end_days_ago + span // 2  # 227
    halves = {"tune": base.copy(eval_end_days_ago=mid), "validation": base.copy(eval_start_days_ago=mid)}
    data = WF.prepare_data(ohlcv, base.horizon_bars, base.feature_set)
    logger.info("Prepared %d bars, %d features (%.0fs)", len(ohlcv), data.features.shape[1], time.perf_counter() - t0)

    result: dict[str, Any] = {"config": base.to_dict(), "hours": hours, "filters": list(FILTERS), "big_move_pct": BIG_MOVE_PCT, "halves": {}}
    for half, cfg in halves.items():
        t1 = time.perf_counter()
        df, retrains = predict_all_hours(data, cfg, hours)
        rows = grid(df, data, cfg, hours)
        logger.info("%s: %d predictions, %d retrains, grid of %d combos (%.0fs)", half, len(df), retrains, len(rows), time.perf_counter() - t1)
        print(f"\n=== {half} ({cfg.eval_start_days_ago}->{cfg.eval_end_days_ago} days ago) ===")
        for r in rows:
            if r["hour"] in STORY_HOURS:
                print(fmt(r))
        crosscheck = None
        if not args.no_crosscheck and half == "tune":
            ref = WF.run_walk_forward(ohlcv, cfg, data, verbose=False).metrics
            mine = next(r for r in rows if r["hour"] == 0 and r["filter"] == "none")
            crosscheck = {
                "run_walk_forward": {k: ref[k] for k in ("days", "directional_accuracy", "traded_days", "traded_directional_accuracy", "total_return_pct", "sharpe_ratio", "max_drawdown_pct")},
                "idea04_hour0_none": mine,
            }
            print(f"cross-check run_walk_forward: days {ref['days']} hit {ref['directional_accuracy']:.4f} traded {ref['traded_days']} "
                  f"traded_hit {ref['traded_directional_accuracy']:.4f} ret {ref['total_return_pct']:+.2f}% | mine: days {mine['days']} hit {mine['hit']:.4f} "
                  f"traded {mine['traded']} traded_hit {mine['traded_hit']:.4f} ret {mine['return_pct']:+.2f}%")
        result["halves"][half] = {
            "window": {"eval_start_days_ago": cfg.eval_start_days_ago, "eval_end_days_ago": cfg.eval_end_days_ago},
            "retrains": retrains, "grid": rows,
            "by_hour_all_days": [{"hour": r["hour"], "days": r["days"], "hit": r["hit"], "traded": r["traded"], "traded_hit": r["traded_hit"], "always_up": r["always_up"]} for r in rows if r["filter"] == "none"],
            "by_weekday": {f"hour_{H}": by_weekday(df, cfg, H) for H in STORY_HOURS if H in hours},
            "after_bigmove": {f"hour_{H}": after_bigmove(df, cfg, H) for H in STORY_HOURS if H in hours},
            "crosscheck": crosscheck,
        }

    # Selection on the tuning half only, then read off the validation half.
    tune_rows, val_rows = result["halves"]["tune"]["grid"], result["halves"]["validation"]["grid"]

    def find(rows: list[dict[str, Any]], H: int, flt: str) -> dict[str, Any]:
        return next(r for r in rows if r["hour"] == H and r["filter"] == flt)

    picks = {
        "baseline": (0, "none"),
        "best_filter_at_hour0": (0, pick([r for r in tune_rows if r["hour"] == 0])["filter"]),
        "best_story_hour_no_filter": (pick([r for r in tune_rows if r["filter"] == "none"], STORY_HOURS)["hour"], "none"),
        "best_story_hour+filter": tuple(pick(tune_rows, STORY_HOURS)[k] for k in ("hour", "filter")),
        "best_any_hour+filter": tuple(pick(tune_rows)[k] for k in ("hour", "filter")),
    }
    comparison = []
    for name, (H, flt) in picks.items():
        comparison.append({"name": name, "hour": H, "filter": flt, "tune": find(tune_rows, H, flt), "validation": find(val_rows, H, flt)})
    result["comparison"] = comparison
    # Stability of the hour effect: rank correlation between halves of the per-hour accuracy.
    th = pd.Series({r["hour"]: r["hit"] for r in tune_rows if r["filter"] == "none"})
    vh = pd.Series({r["hour"]: r["hit"] for r in val_rows if r["filter"] == "none"})
    common = th.index.intersection(vh.index)
    result["hour_hit_rank_corr_tune_vs_val"] = round(float(th[common].corr(vh[common], method="spearman")), 3) if len(common) > 2 else None
    tt = pd.Series({r["hour"]: r["traded_hit"] for r in tune_rows if r["filter"] == "none"})
    vt = pd.Series({r["hour"]: r["traded_hit"] for r in val_rows if r["filter"] == "none"})
    result["hour_traded_hit_rank_corr_tune_vs_val"] = round(float(tt[common].corr(vt[common], method="spearman")), 3) if len(common) > 2 else None
    result["runtime_seconds"] = round(time.perf_counter() - t0, 1)

    print("\n=== Selection (made on tune) -> validation ===")
    head = "| Variant | Hour | Filter | Half | Days | Hit | Traded | Traded hit | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for c in comparison:
        for half in ("tune", "validation"):
            m = c[half]
            lines.append(f"| {c['name']} | {c['hour']:02d} | {c['filter']} | {half} | {m['days']} | {m['hit']:.1%} | {m['traded']} | {m['traded_hit']:.1%} | {m['return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_dd']:.2f}% |")
    print("\n".join(lines))
    print(f"\nhour effect rank corr (all-days hit) tune vs val: {result['hour_hit_rank_corr_tune_vs_val']}; traded hit: {result['hour_traded_hit_rank_corr_tune_vs_val']}")
    for half in ("tune", "validation"):
        print(f"\n[{half}] by weekday at hour 0:")
        for r in result["halves"][half]["by_weekday"]["hour_0"]:
            print(f"  {r['weekday']} days {r['days']:>3} hit {r['hit']:.1%} confident {r['confident']:>3} conf_hit {r['confident_hit']}")
        print(f"[{half}] after |ret|>3% at hour 0: {result['halves'][half]['after_bigmove']['hour_0']}")

    out = settings.models_dir / "idea04_session_filter.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out}  runtime {result['runtime_seconds']}s")


if __name__ == "__main__":
    main()
