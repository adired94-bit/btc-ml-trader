"""Idea 07 - multi-horizon agreement filter for the daily direction forecast.

Three walk-forward daily models are trained with prediction horizons of 24h, 72h and 168h
(extended features, identical hyper-parameters). At the close of every day T each model
emits prob_up / prob_down for its own horizon; the labels used to train the horizon-h model
at day T are restricted (inside ``run_walk_forward``) to bars whose horizon has fully
elapsed by the close of day T, so there is zero look-ahead in any of the three.

The 24h direction is then traded only when the longer horizons agree with it:

* ``all3``  - 24h, 72h and 168h all predict the same side (and conf24 >= threshold)
* ``2of3``  - at least one of 72h / 168h agrees with the 24h side (and conf24 >= threshold)
* ``24+72`` / ``24+168`` - pairwise agreement variants (used for tuning only)

The trade simulation (``simulate_day_trade``) and ``compute_metrics`` are the ones from
``src.backtest.walk_forward`` so P&L is directly comparable with the plain 24h ensemble.

Protocol: window 365 -> 90 days ago, split at the midpoint (227 days ago). Variant / threshold
selection happens on the older (tune) half only; the newer (validation) half is scored once.

Run from the repo root::

    venv\\Scripts\\python.exe scripts\\experiments\\idea07_multi_horizon.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402

HORIZONS = (24, 72, 168)
FAST_XGB = {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 2}
FAST_LGBM = {"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.05, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 40, "n_jobs": 2}
OUT_PATH = PROJECT_ROOT / "models" / "idea07_multi_horizon.json"

STRATEGIES = {
    "baseline": lambda d: np.ones(len(d), dtype=bool),
    "all3": lambda d: (d["dir_24"] == d["dir_72"]) & (d["dir_24"] == d["dir_168"]),
    "2of3": lambda d: (d["dir_24"] == d["dir_72"]) | (d["dir_24"] == d["dir_168"]),
    "24+72": lambda d: d["dir_24"] == d["dir_72"],
    "24+168": lambda d: d["dir_24"] == d["dir_168"],
    # sanity control: trade only when the long horizons DISAGREE with 24h (should be worse if the idea has merit)
    "disagree": lambda d: (d["dir_24"] != d["dir_72"]) & (d["dir_24"] != d["dir_168"]),
}


def base_config() -> WF.WalkForwardConfig:
    return WF.WalkForwardConfig(
        eval_start_days_ago=365, eval_end_days_ago=90, retrain_every_days=14, threshold=0.55,
        feature_set="extended", xgb_params=dict(FAST_XGB), lgbm_params=dict(FAST_LGBM), reg_params=dict(FAST_REG),
        label="ext24_fast",
    )


def horizon_frame(res: WF.WalkForwardResult, h: int) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in res.records])
    out = pd.DataFrame({
        "day": pd.to_datetime(df["day"]),
        f"prob_up_{h}": df["prob_up"], f"conf_{h}": df["confidence"], f"dir_{h}": df["predicted_direction"],
        f"hit_{h}": df["hit"], f"train_end_{h}": pd.to_datetime(df["train_end"]),
    })
    return out.set_index("day")


def resimulate(
    records: list[WF.DayRecord], trade_mask: pd.Series, data: WF.PreparedData, cfg: WF.WalkForwardConfig
) -> list[WF.DayRecord]:
    """Re-run the day-trade simulation on the 24h records with an external trade mask (equity restarts)."""
    idx = data.ohlcv.index
    closes = WF.day_close_bars(idx)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    equity = cfg.initial_equity
    out: list[WF.DayRecord] = []
    for rec in records:
        day = pd.Timestamp(rec.day)
        take = bool(trade_mask.get(day, False)) and rec.confidence >= cfg.threshold
        i = int(pos[closes[day]])
        daily_atr = float(data.daily_atr.iloc[i])
        side, pnl, ret_pct, exit_reason = "NONE", 0.0, 0.0, "NONE"
        if take and np.isfinite(daily_atr) and daily_atr > 0:
            side = "LONG" if rec.predicted_direction == "UP" else "SHORT"
            bars = data.ohlcv.iloc[i + 1: i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, daily_atr, equity, cfg)
            equity += pnl
        else:
            take = False
        out.append(replace(rec, traded=take, trade_side=side, trade_pnl=float(pnl), trade_return_pct=float(ret_pct),
                           trade_exit=exit_reason, equity=float(equity)))
    return out


def summarize(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "days": m["days"], "accuracy": round(m["directional_accuracy"], 4), "traded_days": m["traded_days"],
        "traded_accuracy": round(m["traded_directional_accuracy"], 4), "return_pct": round(m["total_return_pct"], 3),
        "sharpe": round(m["sharpe_ratio"], 3), "max_dd_pct": round(m["max_drawdown_pct"], 3),
        "win_rate": round(m["win_rate"], 4), "profit_factor": round(m["profit_factor"], 3),
        "always_up": round(m["always_up_accuracy"], 4), "momentum_30d": round(m["momentum_30d_accuracy"], 4),
    }


def evaluate(
    name: str, mask: pd.Series, threshold: float, records: list[WF.DayRecord], data: WF.PreparedData,
    cfg: WF.WalkForwardConfig,
) -> dict[str, Any]:
    c = cfg.copy(threshold=threshold, label=name)
    recs = resimulate(records, mask, data, c)
    return {"name": name, "threshold": threshold, **summarize(WF.compute_metrics(recs, c))}


def md_table(rows: list[dict[str, Any]]) -> str:
    head = "| Config | Half | thr | Days | Acc (all) | Traded | Traded acc | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['half']} | {r['threshold']:.2f} | {r['days']} | {r['accuracy']:.1%} | {r['traded_days']} | "
            f"{r['traded_accuracy']:.1%} | {r['return_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd_pct']:.2f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("src", "src.backtest.walk_forward", "src.models.ensemble"):
        logging.getLogger(name).setLevel(logging.WARNING)
    t0 = time.perf_counter()
    ohlcv = storage.get_ohlcv()
    print(f"OHLCV: {len(ohlcv)} bars {ohlcv.index[0]} -> {ohlcv.index[-1]}")
    cfg = base_config()

    # 1) three independent walk-forward runs, one per horizon, over the full 365 -> 90 window
    results: dict[int, WF.WalkForwardResult] = {}
    prepared: dict[int, WF.PreparedData] = {}
    for h in HORIZONS:
        th = time.perf_counter()
        prepared[h] = WF.prepare_data(ohlcv, h, "extended")
        results[h] = WF.run_walk_forward(ohlcv, cfg.copy(horizon_bars=h, label=f"ext_h{h}"), data=prepared[h], verbose=False)
        m = results[h].metrics
        print(
            f"h={h:3d}: {m['days']} days, own-horizon acc {m['directional_accuracy']:.1%}, traded acc {m['traded_directional_accuracy']:.1%} "
            f"on {m['traded_days']} days, retrains {results[h].retrains}, {time.perf_counter() - th:.0f}s"
        )

    # 2) align by day; verify zero look-ahead on the training cut for every horizon
    frames = [horizon_frame(results[h], h) for h in HORIZONS]
    joined = frames[0].join(frames[1:], how="inner")
    idx = ohlcv.index
    closes = WF.day_close_bars(idx)
    for h in HORIZONS:
        day_close_ts = closes.reindex(joined.index)
        gap_hours = (day_close_ts - joined[f"train_end_{h}"]).dt.total_seconds() / 3600
        assert (gap_hours >= h).all(), f"look-ahead: horizon {h} trained on bars closer than {h}h to the day close"
    print(f"Aligned days: {len(joined)} (per-horizon: {[len(f) for f in frames]}); train-cut check passed for all horizons")
    agree_all = float(STRATEGIES["all3"](joined).mean())
    agree_2 = float(STRATEGIES["2of3"](joined).mean())
    print(f"Agreement rates: all3 {agree_all:.1%}, 2of3 {agree_2:.1%}")

    # 3) split: tune = older half [start, mid), validation = newer half [mid, end]
    last = idx[-1]
    span = cfg.eval_start_days_ago - cfg.eval_end_days_ago
    mid_days_ago = cfg.eval_end_days_ago + span // 2
    mid = (last - pd.Timedelta(days=mid_days_ago)).floor("D")
    recs24 = [r for r in results[24].records if pd.Timestamp(r.day) in joined.index]
    halves = {
        "tune": [r for r in recs24 if pd.Timestamp(r.day) < mid],
        "validation": [r for r in recs24 if pd.Timestamp(r.day) >= mid],
    }
    print(f"Split at {mid.date()}: tune {len(halves['tune'])} days, validation {len(halves['validation'])} days")

    # sanity: baseline re-simulation over the full window must reproduce run_walk_forward's own P&L
    full_base = resimulate(recs24, pd.Series(True, index=joined.index), prepared[24], cfg)
    ref = results[24].metrics
    chk = WF.compute_metrics(full_base, cfg)
    print(f"Resim check (full window): return {chk['total_return_pct']:+.2f}% vs run_walk_forward {ref['total_return_pct']:+.2f}%, "
          f"traded {chk['traded_days']} vs {ref['traded_days']}")

    # 4) tuning grid on the older half only
    thresholds = (0.50, 0.55, 0.60)
    tune_rows: list[dict[str, Any]] = []
    for name, fn in STRATEGIES.items():
        mask = pd.Series(fn(joined), index=joined.index)
        for thr in thresholds:
            tune_rows.append({**evaluate(name, mask, thr, halves["tune"], prepared[24], cfg), "half": "tune"})
    print("\n### Tuning half (older)\n" + md_table(tune_rows))

    def score(r: dict[str, Any]) -> float:
        # same spirit as scripts/improve.py: accuracy first, P&L as a tiebreaker; need a minimum sample
        if r["traded_days"] < 20:
            return -1.0
        return r["traded_accuracy"] + 0.0005 * r["return_pct"]

    candidates = [r for r in tune_rows if r["name"] not in ("baseline", "disagree")]
    best_tune = max(candidates, key=score)
    print(f"\nBest on tune (min 20 trades): {best_tune['name']} thr={best_tune['threshold']:.2f} "
          f"traded acc {best_tune['traded_accuracy']:.1%} on {best_tune['traded_days']} days")

    # 5) validation half: baseline, the two pre-registered variants at 0.55, and the tune-selected config
    val_specs = [("baseline", 0.55), ("all3", 0.55), ("2of3", 0.55), ("disagree", 0.55)]
    if (best_tune["name"], best_tune["threshold"]) not in val_specs:
        val_specs.append((best_tune["name"], best_tune["threshold"]))
    val_rows: list[dict[str, Any]] = []
    for name, thr in val_specs:
        mask = pd.Series(STRATEGIES[name](joined), index=joined.index)
        val_rows.append({**evaluate(name, mask, thr, halves["validation"], prepared[24], cfg), "half": "validation"})
    print("\n### Validation half (newer, never used for selection)\n" + md_table(val_rows))

    # per-horizon standalone accuracy on each half (their own horizon), for context
    horizon_acc: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        hj = joined[f"hit_{h}"]
        horizon_acc[str(h)] = {"tune": round(float(hj[hj.index < mid].mean()), 4), "validation": round(float(hj[hj.index >= mid].mean()), 4)}
    print("\nStandalone own-horizon accuracy:", json.dumps(horizon_acc))

    runtime = time.perf_counter() - t0
    payload = {
        "idea": "07 multi-horizon agreement (24h traded only when 72h/168h models agree)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime, 1),
        "config": {**cfg.to_dict(), "horizons": list(HORIZONS), "split_day": mid.isoformat()},
        "aligned_days": int(len(joined)),
        "agreement_rate": {"all3": agree_all, "2of3": agree_2},
        "horizon_standalone_accuracy": horizon_acc,
        "horizon_runs": {str(h): {"retrains": results[h].retrains, **summarize(results[h].metrics)} for h in HORIZONS},
        "resim_check": {"resim_return_pct": chk["total_return_pct"], "walk_forward_return_pct": ref["total_return_pct"]},
        "tune": tune_rows,
        "best_tune": best_tune,
        "validation": val_rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} ({runtime:.0f}s)")


if __name__ == "__main__":
    main()
