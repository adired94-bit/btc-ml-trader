"""Systematic improvement campaign for the daily direction forecast.

Every experiment is scored on the *tuning half* of the evaluation window
(older half). The best configurations are then re-scored on the *validation
half* (newer half) they were never selected on, and the winner is finally run
over the full window. Results are appended to SYSTEM_LEARNINGS.md and saved as
JSON. Nothing here touches the tests.

    venv\\Scripts\\python.exe scripts\\improve.py            # full campaign
    venv\\Scripts\\python.exe scripts\\improve.py --quick    # fewer experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("improve")

REG_STRONG = {
    "xgb_params": {**WF.WF_XGB_PARAMS, "max_depth": 3, "min_child_weight": 20, "learning_rate": 0.03, "reg_lambda": 10.0, "subsample": 0.7},
    "lgbm_params": {**WF.WF_LGBM_PARAMS, "num_leaves": 7, "min_child_samples": 100, "learning_rate": 0.03, "reg_lambda": 10.0},
}
REGIME_W = {"trend_up": 1.5, "high_volatility": 1.5}
LONG_SUBSET = ("baseline", "base+regime_w", "ext+ens", "ext+ens+regime_w", "ext+ens_strongreg", "base+logreg", "ext+ens_h72")
ONCHAIN_SUBSET = ("baseline", "base+regime_w", "oc+ens", "oc+ens+regime_w", "oc+ens_strongreg", "oc+logreg")
PATTERN_SUBSET = ("baseline", "base+regime_w", "pat+ens", "pat+ens+regime_w", "pat+ens_strongreg", "pat+logreg")
FLOW_SUBSET = ("baseline", "base+regime_w", "flow+ens", "flow+ens+regime_w", "flow+ens_strongreg", "flow+logreg", "flow+ens+retrain7d")


def experiments(quick: bool) -> list[WF.WalkForwardConfig]:
    base = WF.WalkForwardConfig()
    exps = [
        base.copy(label="baseline"),
        base.copy(label="base+regime_w", regime_weights=REGIME_W),
        base.copy(label="ext+ens", feature_set="extended"),
        base.copy(label="ext+ens+regime_w", feature_set="extended", regime_weights=REGIME_W),
        base.copy(label="ext+ens_strongreg", feature_set="extended", **REG_STRONG),
        base.copy(label="base+logreg", model="logreg"),
        base.copy(label="ext+logreg_c0.05", feature_set="extended", model="logreg", logreg_c=0.05),
        base.copy(label="ext+logreg_c0.01", feature_set="extended", model="logreg", logreg_c=0.01),
        base.copy(label="ext+logreg_c0.5", feature_set="extended", model="logreg", logreg_c=0.5),
        base.copy(label="ext+ens+retrain1d", feature_set="extended", retrain_every_days=1),
        base.copy(label="ext+ens_h72", feature_set="extended", horizon_bars=72),
        base.copy(label="ext+logreg_h72", feature_set="extended", model="logreg", horizon_bars=72),
        base.copy(label="ext+ens_h168", feature_set="extended", horizon_bars=168),
        base.copy(label="ext+logreg_h168", feature_set="extended", model="logreg", horizon_bars=168),
        base.copy(label="oc+ens", feature_set="onchain"),
        base.copy(label="oc+ens+regime_w", feature_set="onchain", regime_weights=REGIME_W),
        base.copy(label="oc+ens_strongreg", feature_set="onchain", **REG_STRONG),
        base.copy(label="oc+logreg", feature_set="onchain", model="logreg"),
        base.copy(label="pat+ens", feature_set="patterns"),
        base.copy(label="pat+ens+regime_w", feature_set="patterns", regime_weights=REGIME_W),
        base.copy(label="pat+ens_strongreg", feature_set="patterns", **REG_STRONG),
        base.copy(label="pat+logreg", feature_set="patterns", model="logreg"),
        base.copy(label="flow+ens", feature_set="flow"),
        base.copy(label="flow+ens+regime_w", feature_set="flow", regime_weights=REGIME_W),
        base.copy(label="flow+ens_strongreg", feature_set="flow", **REG_STRONG),
        base.copy(label="flow+logreg", feature_set="flow", model="logreg"),
        base.copy(label="flow+ens+retrain7d", feature_set="flow", retrain_every_days=7),
    ]
    if quick:
        exps = [e for e in exps if e.label in ("baseline", "ext+ens", "ext+logreg_c0.05", "ext+logreg_h72")]
    return exps


def score(m: dict) -> float:
    """Primary: hit rate on all days; secondary: traded hit rate and P&L."""
    return m["directional_accuracy"] + 0.5 * m["traded_directional_accuracy"] + 0.0005 * m["total_return_pct"]


def run(ohlcv, cfg: WF.WalkForwardConfig, cache: dict) -> WF.WalkForwardResult:
    key = (cfg.horizon_bars, cfg.feature_set)
    if key not in cache:
        cache[key] = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    return WF.run_walk_forward(ohlcv, cfg, cache[key], verbose=False)


def row(label: str, half: str, m: dict) -> dict:
    return {
        "label": label, "half": half, "days": m["days"], "hit": round(m["directional_accuracy"], 4),
        "momentum_hit": round(m["momentum_30d_accuracy"], 4), "always_up": round(m["always_up_accuracy"], 4),
        "traded_hit": round(m["traded_directional_accuracy"], 4), "trades": m["traded_days"],
        "mae_pct": round(m["mae_pct"], 3), "naive_mae_pct": round(m["naive_mae_pct"], 3),
        "return_pct": round(m["total_return_pct"], 2), "sharpe": round(m["sharpe_ratio"], 2),
        "max_dd": round(m["max_drawdown_pct"], 2), "score": round(score(m), 4),
    }


def md_table(rows: list[dict]) -> str:
    head = "| Config | Half | Days | Hit rate | Momentum 30d | Always UP | Traded hit | Trades | MAE | Naive MAE | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    body = [
        f"| {r['label']} | {r['half']} | {r['days']} | {r['hit']:.1%} | {r['momentum_hit']:.1%} | {r['always_up']:.1%} | {r['traded_hit']:.1%} | {r['trades']} | {r['mae_pct']:.2f}% | {r['naive_mae_pct']:.2f}% | {r['return_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% |"
        for r in rows
    ]
    return "\n".join([head, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--top", type=int, default=3, help="configs promoted to the validation half")
    parser.add_argument("--long", action="store_true", help="use the 6-year research cache and a ~3-year evaluation window")
    parser.add_argument("--flow", action="store_true", help="with --long: order-flow / derivatives feature experiments")
    parser.add_argument("--patterns", action="store_true", help="with --long: candlestick-pattern feature experiments")
    parser.add_argument("--onchain", action="store_true", help="with --long: on-chain / sentiment / macro feature experiments")
    args = parser.parse_args()
    t0 = time.perf_counter()

    if args.long:
        from scripts.fetch_long_history import LONG_CACHE

        ohlcv = storage.load_cached(LONG_CACHE)
        if ohlcv is None:
            raise SystemExit("Run scripts/fetch_long_history.py first")
        window = {"eval_start_days_ago": 1_100, "eval_end_days_ago": 90, "retrain_every_days": 21 if (args.flow or args.patterns or args.onchain) else 14}
        subset = ONCHAIN_SUBSET if args.onchain else PATTERN_SUBSET if args.patterns else FLOW_SUBSET if args.flow else LONG_SUBSET
        exps = []
        for e in experiments(args.quick):
            if e.label not in subset:
                continue
            overrides = dict(window)
            if "retrain" in e.label:  # experiments that *are* about retrain cadence keep their own value
                overrides.pop("retrain_every_days")
            exps.append(e.copy(**overrides))
    else:
        ohlcv = storage.get_ohlcv()
        exps = experiments(args.quick)
    base = exps[0]
    span = base.eval_start_days_ago - base.eval_end_days_ago
    mid = base.eval_end_days_ago + span // 2
    cache: dict = {}

    tune_rows: list[dict] = []
    for cfg in exps:
        try:
            res = run(ohlcv, cfg.copy(eval_end_days_ago=mid), cache)
        except Exception as exc:  # noqa: BLE001 - one broken experiment must not stop the campaign
            logger.error("Experiment %s failed: %s", cfg.label, exc)
            continue
        r = row(cfg.label, "tune", res.metrics)
        tune_rows.append(r)
        print(f"[tune] {cfg.label:<22} hit {r['hit']:.1%} (mom {r['momentum_hit']:.1%}) traded {r['traded_hit']:.1%} ({r['trades']}) MAE {r['mae_pct']:.2f}% ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:.2f}", flush=True)

    ranked = sorted(tune_rows, key=lambda r: r["score"], reverse=True)
    promoted = [r["label"] for r in ranked[: args.top]]
    if "baseline" not in promoted:
        promoted.append("baseline")
    by_label = {c.label: c for c in exps}

    val_rows: list[dict] = []
    for label in promoted:
        res = run(ohlcv, by_label[label].copy(eval_start_days_ago=mid), cache)
        r = row(label, "validation", res.metrics)
        val_rows.append(r)
        print(f"[val ] {label:<22} hit {r['hit']:.1%} traded {r['traded_hit']:.1%} ({r['trades']}) MAE {r['mae_pct']:.2f}% ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:.2f}", flush=True)

    base_val = next(r for r in val_rows if r["label"] == "baseline")
    best_val = max(val_rows, key=lambda r: r["score"])
    winner = best_val["label"] if best_val["score"] > base_val["score"] else "baseline"
    full = run(ohlcv, by_label[winner].copy(label=winner), cache)
    full_row = row(winner, "full", full.metrics)
    print(f"[full] {winner:<22} hit {full_row['hit']:.1%} traded {full_row['traded_hit']:.1%} ({full_row['trades']}) MAE {full_row['mae_pct']:.2f}% ret {full_row['return_pct']:+.2f}% sharpe {full_row['sharpe']:.2f}", flush=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = [
        f"\n## {now} — Improvement campaign{' (6-year history, ~3-year window' + (', order-flow features)' if args.flow else ', candlestick patterns)' if args.patterns else ', on-chain / macro features)' if args.onchain else ')') if args.long else ''}: {len(tune_rows)} experiments on the daily forecast\n",
        "Each experiment scored on the tuning half (older), the top configs re-scored on the validation half (newer), winner re-run over the full window. Score = hit + 0.5 × traded hit + 0.0005 × return%.\n",
        "### Tuning half\n", md_table(ranked), "\n### Validation half (never used for selection)\n", md_table(val_rows),
        f"\n### Winner: `{winner}` — full window\n", md_table([full_row]),
        "\n**Learnings.**\n",
    ]
    best_tune = ranked[0]
    entry.append(f"* Best on the tuning half: `{best_tune['label']}` ({best_tune['hit']:.1%} hit rate); on validation it scored {next((r['hit'] for r in val_rows if r['label']==best_tune['label']), float('nan')):.1%}.")
    entry.append(f"* Baseline validation hit rate {base_val['hit']:.1%} vs winner {best_val['hit']:.1%}; {'adopted' if winner != 'baseline' else 'no candidate generalised, baseline kept'}.")
    h168 = [r for r in ranked if "h168" in r["label"]]
    if h168:
        entry.append(f"* 7-day horizon reaches {max(r['hit'] for r in h168):.1%} on tuning but its days overlap heavily (7× fewer independent samples) — treat with caution.")
    entry.append(f"* Full-window winner hit rate {full_row['hit']:.1%}, MAE {full_row['mae_pct']:.2f}% vs naive {full_row['naive_mae_pct']:.2f}%.")
    entry.append(f"* Campaign runtime {time.perf_counter() - t0:.0f}s.\n")
    text = "\n".join(entry)
    WF.append_learnings(text)
    out = settings.models_dir / ("improvement_campaign_onchain.json" if args.onchain else "improvement_campaign_patterns.json" if args.patterns else "improvement_campaign_flow.json" if args.flow else "improvement_campaign_long.json" if args.long else "improvement_campaign.json")
    out.write_text(json.dumps({"tune": ranked, "validation": val_rows, "full": full_row, "winner": by_label[winner].to_dict()}, indent=2), encoding="utf-8")
    print("\n" + text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
