"""Idea 10 - mean reversion at extremes (research experiment, not production code).

Hypothesis: on days where the hourly RSI-14 at the daily close is > 70 / < 30, or Bollinger %B is
> 1 / < 0, the next-day return tends to revert. SYSTEM_LEARNINGS shows the model misses 85 % of
"overbought" days and 62 % of "oversold" days, so a specialised rule might beat it there.

Rules compared against the plain extended-features ensemble (``baseline``):

* ``fade_always``   - on extreme days trade the fade (SHORT when overbought, LONG when oversold)
                      regardless of the model; elsewhere trade the model as usual.
* ``fade_agree``    - on extreme days trade the fade only if the model direction agrees; elsewhere
                      trade the model as usual.
* ``suppress``      - never trade on extreme days; trade the model elsewhere.
* ``fade_only``     - fade on extreme days and nothing else (isolates the fade edge).

Protocol: one walk-forward pass (``run_walk_forward``) over the 365->90 days-ago window with light
boosters; the per-day model probabilities are then post-processed by every rule and the trades are
re-simulated with ``simulate_day_trade`` from a fresh 10 000 equity on each half. The rule family and
the extreme definition are chosen on the older half (365->227) and validated on the newer half
(227->90). Baseline numbers are produced by the very same re-simulation, so they are comparable.

Run from the repo root::

    venv\\Scripts\\python.exe scripts/experiments/idea10_mean_reversion.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea10")

EVAL_START, EVAL_SPLIT, EVAL_END = 365, 227, 90
RULES = ("baseline", "fade_always", "fade_agree", "suppress", "fade_only")
EXTREME_GRID: list[dict[str, Any]] = [
    {"name": f"{src}_rsi{hi}_bb{bb:g}", "source": src, "rsi_hi": hi, "rsi_lo": 100 - hi, "bb_hi": bb, "bb_lo": 1 - bb}
    for src, hi, bb in product(("rsi", "bb", "either", "both"), (65, 70, 75), (1.0, 1.05))
    if not (src == "rsi" and bb != 1.0) and not (src == "bb" and hi != 70)  # avoid duplicate configs
]


# ----------------------------------------------------------------------
# Walk-forward pass
# ----------------------------------------------------------------------


def light_config() -> WF.WalkForwardConfig:
    xgb = dict(WF.WF_XGB_PARAMS, n_estimators=80, n_jobs=2)
    lgbm = dict(WF.WF_LGBM_PARAMS, n_estimators=80, n_jobs=2, verbose=-1)
    reg = dict(WF.WF_REG_PARAMS, n_estimators=80, n_jobs=2)
    return WF.WalkForwardConfig(
        eval_start_days_ago=EVAL_START, eval_end_days_ago=EVAL_END, retrain_every_days=14, feature_set="extended",
        xgb_params=xgb, lgbm_params=lgbm, reg_params=reg, label="ext+ens_light",
    )


def load_data() -> pd.DataFrame:
    ohlcv = storage.get_ohlcv()
    return ohlcv.tail(settings.history_candles) if len(ohlcv) > settings.history_candles else ohlcv


# ----------------------------------------------------------------------
# Extreme flags & rule application
# ----------------------------------------------------------------------


def extreme_state(rsi: float, pct_b: float, spec: dict[str, Any]) -> str:
    """Return 'overbought', 'oversold' or '' for the day-close bar."""
    rsi_ob, rsi_os = rsi > spec["rsi_hi"], rsi < spec["rsi_lo"]
    bb_ob, bb_os = pct_b > spec["bb_hi"], pct_b < spec["bb_lo"]
    src = spec["source"]
    if src == "rsi":
        ob, os_ = rsi_ob, rsi_os
    elif src == "bb":
        ob, os_ = bb_ob, bb_os
    elif src == "either":
        ob, os_ = rsi_ob or bb_ob, rsi_os or bb_os
    else:  # both
        ob, os_ = rsi_ob and bb_ob, rsi_os and bb_os
    if ob and not os_:
        return "overbought"
    if os_ and not ob:
        return "oversold"
    return ""


def decide(rule: str, rec: WF.DayRecord, state: str, threshold: float) -> tuple[str, str]:
    """Return (predicted_direction, trade_side) for one day under ``rule``."""
    model_dir = rec.predicted_direction
    model_side = ("LONG" if model_dir == "UP" else "SHORT") if rec.confidence >= threshold else "NONE"
    if not state or rule == "baseline":
        return model_dir, model_side
    fade_dir = "DOWN" if state == "overbought" else "UP"
    fade_side = "SHORT" if fade_dir == "DOWN" else "LONG"
    if rule == "fade_always" or rule == "fade_only":
        return fade_dir, fade_side
    if rule == "fade_agree":
        return (fade_dir, fade_side) if model_dir == fade_dir else (model_dir, "NONE")
    if rule == "suppress":
        return model_dir, "NONE"
    raise ValueError(rule)


def resimulate(
    rule: str, records: list[WF.DayRecord], data: WF.PreparedData, pos: pd.Series, spec: dict[str, Any],
    cfg: WF.WalkForwardConfig,
) -> dict[str, Any]:
    equity = cfg.initial_equity
    rows: list[dict[str, Any]] = []
    for rec in records:
        i = int(pos[pd.Timestamp(rec.day)])
        rsi = float(data.indicators["rsi_14"].iloc[i])
        pct_b = float(data.indicators["bb_pct_b"].iloc[i])
        state = extreme_state(rsi, pct_b, spec) if np.isfinite(rsi) and np.isfinite(pct_b) else ""
        pred_dir, side = decide(rule, rec, state, cfg.threshold)
        if rule == "fade_only" and not state:
            side = "NONE"
        daily_atr = float(data.daily_atr.iloc[i])
        pnl, ret_pct, exit_reason = 0.0, 0.0, "NONE"
        if side != "NONE" and np.isfinite(daily_atr) and daily_atr > 0:
            bars = data.ohlcv.iloc[i + 1: i + 1 + cfg.horizon_bars]
            pnl, ret_pct, exit_reason, _, _, _ = WF.simulate_day_trade(side, bars, daily_atr, equity, cfg)
            equity += pnl
        else:
            side = "NONE"
        rows.append({
            "day": rec.day, "state": state, "pred_dir": pred_dir, "actual_dir": rec.actual_direction,
            "hit": pred_dir == rec.actual_direction, "traded": side != "NONE", "side": side, "pnl": pnl,
            "ret_pct": ret_pct, "exit": exit_reason, "equity": equity, "actual_ret": rec.actual_return_pct,
            "model_dir": rec.predicted_direction, "model_conf": rec.confidence,
        })
    return summarise(pd.DataFrame(rows), cfg)


def summarise(df: pd.DataFrame, cfg: WF.WalkForwardConfig) -> dict[str, Any]:
    traded = df[df["traded"]]
    daily_ret = df["ret_pct"] / 100
    sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if daily_ret.std() > 0 else 0.0
    equity = pd.concat([pd.Series([cfg.initial_equity]), df["equity"]]).reset_index(drop=True)
    ext = df[df["state"] != ""]
    fade_dir = np.where(ext["state"] == "overbought", "DOWN", "UP")
    fade_hits = (fade_dir == ext["actual_dir"].to_numpy()) if len(ext) else np.array([], dtype=bool)
    ob, os_ = ext[ext["state"] == "overbought"], ext[ext["state"] == "oversold"]
    return {
        "days": int(len(df)),
        "directional_accuracy": float(df["hit"].mean()),
        "traded_days": int(len(traded)),
        "traded_directional_accuracy": float(traded["hit"].mean()) if len(traded) else 0.0,
        "total_return_pct": float((equity.iloc[-1] / cfg.initial_equity - 1) * 100),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": float(-(equity / equity.cummax() - 1).min() * 100),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "extreme_days": int(len(ext)),
        "overbought_days": int(len(ob)),
        "oversold_days": int(len(os_)),
        "fade_hit_rate": float(fade_hits.mean()) if len(ext) else float("nan"),
        "fade_hit_rate_overbought": float((ob["actual_dir"] == "DOWN").mean()) if len(ob) else float("nan"),
        "fade_hit_rate_oversold": float((os_["actual_dir"] == "UP").mean()) if len(os_) else float("nan"),
        "model_hit_rate_on_extremes": float((ext["model_dir"] == ext["actual_dir"]).mean()) if len(ext) else float("nan"),
        "extreme_mean_next_ret_pct": float(ext["actual_ret"].mean()) if len(ext) else float("nan"),
        "extreme_traded_days": int(traded["state"].ne("").sum()),
        "extreme_trade_pnl": float(traded.loc[traded["state"] != "", "pnl"].sum()),
        "exit_reasons": {k: int(v) for k, v in traded["exit"].value_counts().items()},
    }


def score(m: dict[str, Any]) -> float:
    """Selection score on the tuning half: Sharpe first, accuracy as tie-breaker."""
    return m["sharpe_ratio"] + 2.0 * (m["traded_directional_accuracy"] - 0.5)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def fmt(label: str, half: str, m: dict[str, Any]) -> str:
    fh = "n/a" if not np.isfinite(m["fade_hit_rate"]) else f"{m['fade_hit_rate'] * 100:.1f}%"
    return (
        f"| {label} | {half} | {m['days']} | {m['directional_accuracy'] * 100:.1f}% | {m['traded_days']} | "
        f"{m['traded_directional_accuracy'] * 100:.1f}% | {m['total_return_pct']:+.2f}% | {m['sharpe_ratio']:.2f} | "
        f"{m['max_drawdown_pct']:.2f}% | {m['extreme_days']} | {fh} |"
    )


HEADER = (
    "| Config | Half | Days | Hit rate | Traded | Traded hit | Return | Sharpe | Max DD | Extreme days | Fade hit |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    t0 = time.perf_counter()
    cfg = light_config()
    ohlcv = load_data()
    logger.info("Loaded %d candles %s -> %s", len(ohlcv), ohlcv.index[0], ohlcv.index[-1])
    data = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    result = WF.run_walk_forward(ohlcv, cfg, data=data, verbose=True)
    wf_seconds = time.perf_counter() - t0
    logger.info("Walk-forward done in %.0fs (%d retrains)", wf_seconds, result.retrains)

    pos = pd.Series(np.arange(len(ohlcv)), index=ohlcv.index)
    split_day = (ohlcv.index[-1] - pd.Timedelta(days=EVAL_SPLIT)).floor("D")
    tune = [r for r in result.records if pd.Timestamp(r.day) < split_day]
    valid = [r for r in result.records if pd.Timestamp(r.day) >= split_day]
    logger.info("Tune half: %d days, validation half: %d days (split %s)", len(tune), len(valid), split_day.date())

    # --- tuning half: every rule x every extreme definition ---------------------------------
    no_extreme = {"name": "none", "source": "rsi", "rsi_hi": 101, "rsi_lo": -1, "bb_hi": 99, "bb_lo": -99}
    base_tune = resimulate("baseline", tune, data, pos, no_extreme, cfg)
    base_valid = resimulate("baseline", valid, data, pos, no_extreme, cfg)
    tune_rows: list[dict[str, Any]] = []
    for spec in EXTREME_GRID:
        for rule in RULES[1:]:
            m = resimulate(rule, tune, data, pos, spec, cfg)
            tune_rows.append({"rule": rule, "extreme": spec["name"], "score": score(m), **m})
    tune_df = pd.DataFrame(tune_rows).sort_values("score", ascending=False)

    # --- pick the best extreme definition per rule family on the tuning half, validate it ----
    chosen: dict[str, dict[str, Any]] = {}
    for rule in RULES[1:]:
        best = tune_df[tune_df["rule"] == rule].iloc[0]
        spec = next(s for s in EXTREME_GRID if s["name"] == best["extreme"])
        chosen[rule] = {
            "extreme": spec, "tune": {k: v for k, v in best.items() if k not in ("rule", "extreme", "score")},
            "validation": resimulate(rule, valid, data, pos, spec, cfg),
        }
    # Also validate the canonical definition from the task (RSI 70/30 or %B 1/0) for every rule.
    canonical = next(s for s in EXTREME_GRID if s["name"] == "either_rsi70_bb1")
    canonical_res = {
        rule: {"tune": resimulate(rule, tune, data, pos, canonical, cfg), "validation": resimulate(rule, valid, data, pos, canonical, cfg)}
        for rule in RULES[1:]
    }

    # --- report ----------------------------------------------------------------------------
    lines = [HEADER, fmt("baseline (ext+ens)", "tune", base_tune), fmt("baseline (ext+ens)", "validation", base_valid)]
    for rule, res in canonical_res.items():
        lines.append(fmt(f"{rule} [canonical either_rsi70_bb1]", "tune", res["tune"]))
        lines.append(fmt(f"{rule} [canonical either_rsi70_bb1]", "validation", res["validation"]))
    for rule, res in chosen.items():
        lines.append(fmt(f"{rule} [tuned {res['extreme']['name']}]", "tune", res["tune"]))
        lines.append(fmt(f"{rule} [tuned {res['extreme']['name']}]", "validation", res["validation"]))
    report = "\n".join(lines)
    print("\n" + report + "\n")
    print("Top 10 tuning-half configs by score:")
    print(tune_df[["rule", "extreme", "score", "traded_days", "traded_directional_accuracy", "total_return_pct", "sharpe_ratio",
                   "extreme_days", "fade_hit_rate"]].head(10).to_string(index=False))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "window": {"eval_start_days_ago": EVAL_START, "split_days_ago": EVAL_SPLIT, "eval_end_days_ago": EVAL_END,
                   "split_day": split_day.isoformat(), "tune_days": len(tune), "validation_days": len(valid)},
        "walk_forward_metrics_full_window": result.metrics,
        "walk_forward_retrains": result.retrains,
        "baseline": {"tune": base_tune, "validation": base_valid},
        "canonical_rules": canonical_res,
        "tuned_rules": chosen,
        "tuning_grid": tune_df.to_dict(orient="records"),
        "report_markdown": report,
        "runtime_seconds": time.perf_counter() - t0,
        "walk_forward_seconds": wf_seconds,
    }
    out = PROJECT_ROOT / "models" / "idea10_mean_reversion.json"
    out.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    logger.info("Saved %s (total %.0fs)", out, payload["runtime_seconds"])


if __name__ == "__main__":
    main()
