"""Experiment: predict next-day *volatility* instead of direction, and trade breakouts.

Hypothesis (SYSTEM_LEARNINGS.md): daily direction is ~coin-flip, but whether tomorrow
is a "big" day is far more predictable. If so, a breakout strategy (enter in whichever
direction the market breaks out of the current close ± k × ATR, only on predicted big
days) can profit without forecasting direction.

Walk-forward, zero look-ahead: at the close of day T a LightGBM classifier trained on
hourly rows whose label is already known predicts P(big day T+1). "Big" means the
absolute 24-bar return exceeds the trailing 90-day median of absolute daily returns
(computed causally). Retrained every ``retrain_every_days``.

    venv\\Scripts\\python.exe scripts\\volatility_target.py --long
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest.walk_forward import BARS_PER_DAY, append_learnings, day_close_bars, evaluation_days, WalkForwardConfig  # noqa: E402
from src.data import storage  # noqa: E402
from src.data.processor import add_all_indicators, build_extended_features  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("volatility_target")

CLF_PARAMS = {
    "n_estimators": 200, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "min_child_samples": 50, "reg_lambda": 5.0, "verbose": -1, "n_jobs": -1,
    "random_state": settings.random_state,
}


@dataclass
class DayRow:
    day: str
    close: float
    p_big: float
    predicted_big: bool
    actual_big: bool
    actual_abs_return_pct: float
    naive_big: bool  # yesterday was big
    traded: bool
    side: str
    pnl: float
    return_pct: float
    exit_reason: str
    equity: float


def prepare(ohlcv: pd.DataFrame, horizon: int = BARS_PER_DAY):
    ind = add_all_indicators(ohlcv)
    feats = build_extended_features(ohlcv, ind)
    close = ohlcv["close"]
    fut_abs = (close.shift(-horizon) / close - 1.0).abs()
    # causal threshold: trailing 90-day median of realised |24h return| (known up to bar t - horizon)
    past_abs = (close / close.shift(horizon) - 1.0).abs()
    threshold = past_abs.rolling(90 * BARS_PER_DAY, min_periods=30 * BARS_PER_DAY).median()
    label = pd.Series(np.where(fut_abs > threshold, 1, 0), index=ohlcv.index)
    label[fut_abs.isna() | threshold.isna()] = -1
    daily = ohlcv.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev).abs(), (daily["low"] - prev).abs()], axis=1).max(axis=1)
    datr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().reindex(ohlcv.index, method="ffill")
    return feats, label, fut_abs, threshold, datr


def simulate_breakout(bars: pd.DataFrame, close: float, atr: float, equity: float, k: float, rr: float, fee: float, slip: float, risk_pct: float):
    """OCO breakout: buy-stop at close + k*ATR, sell-stop at close - k*ATR; first touched wins.
    Stop for the trade is the entry level minus/plus k*ATR, take-profit at rr times that."""
    up_lvl, dn_lvl = close + k * atr, close - k * atr
    side, entry, i0 = None, None, None
    for i, (_, b) in enumerate(bars.iterrows()):
        if b["high"] >= up_lvl and b["low"] <= dn_lvl:
            return 0.0, 0.0, "NONE", "AMBIGUOUS"  # both levels in one bar: skip (conservative)
        if b["high"] >= up_lvl:
            side, entry, i0 = "LONG", up_lvl * (1 + slip), i
            break
        if b["low"] <= dn_lvl:
            side, entry, i0 = "SHORT", dn_lvl * (1 - slip), i
            break
    if side is None:
        return 0.0, 0.0, "NONE", "NO_BREAKOUT"
    dist = k * atr
    stop = entry - dist if side == "LONG" else entry + dist
    tp = entry + rr * dist if side == "LONG" else entry - rr * dist
    size = equity * risk_pct / dist
    exit_price, reason = None, "CLOSE"
    for _, b in bars.iloc[i0:].iterrows():
        if side == "LONG":
            if b["low"] <= stop:
                exit_price, reason = stop, "STOP"; break
            if b["high"] >= tp:
                exit_price, reason = tp, "TAKE_PROFIT"; break
        else:
            if b["high"] >= stop:
                exit_price, reason = stop, "STOP"; break
            if b["low"] <= tp:
                exit_price, reason = tp, "TAKE_PROFIT"; break
    if exit_price is None:
        last = float(bars["close"].iloc[-1])
        exit_price = last * (1 - slip) if side == "LONG" else last * (1 + slip)
    gross = (exit_price - entry) * size if side == "LONG" else (entry - exit_price) * size
    pnl = gross - fee * size * (entry + exit_price)
    return float(pnl), float(pnl / equity * 100), side, reason


def run(ohlcv: pd.DataFrame, start_days_ago: int, end_days_ago: int, retrain_every: int, p_threshold: float,
        k: float, rr: float, trade_all_days: bool = False, verbose: bool = True) -> tuple[list[DayRow], dict]:
    feats, label, fut_abs, thr, datr = prepare(ohlcv)
    cfg = WalkForwardConfig(eval_start_days_ago=start_days_ago, eval_end_days_ago=end_days_ago)
    start, end = evaluation_days(ohlcv.index, cfg)
    closes = day_close_bars(ohlcv.index)
    days = [d for d in closes.index if start <= d <= end]
    pos = pd.Series(np.arange(len(ohlcv)), index=ohlcv.index)
    model, last_train, equity, rows = None, None, 10_000.0, []
    fee, slip, risk_pct = settings.backtest_fee_pct, settings.backtest_slippage_pct, 0.01
    prev_big = False
    for day in days:
        ts = closes[day]
        i = int(pos[ts])
        if i + BARS_PER_DAY >= len(ohlcv):
            break
        train_end = ohlcv.index[i - BARS_PER_DAY]
        if model is None or (day - last_train).days >= retrain_every:
            mask = (feats.index <= train_end) & (label >= 0).to_numpy()
            X = feats[mask].dropna()
            y = label.loc[X.index]
            model = lgb.LGBMClassifier(**CLF_PARAMS).fit(X.to_numpy(dtype=np.float32), y.to_numpy())
            cols = list(X.columns)
            last_train = day
            if verbose:
                logger.info("Retrained at %s on %d rows (big-day share %.2f)", day.date(), len(X), y.mean())
        row = feats.iloc[[i]][cols]
        if row.isna().any(axis=None) or not np.isfinite(datr.iloc[i]):
            continue
        p_big = float(model.predict_proba(row.to_numpy(dtype=np.float32))[0][1])
        actual_big = bool(fut_abs.iloc[i] > thr.iloc[i])
        predicted_big = p_big >= p_threshold
        traded = trade_all_days or predicted_big
        pnl = ret = 0.0
        side, reason = "NONE", "NONE"
        if traded:
            bars = ohlcv.iloc[i + 1: i + 1 + BARS_PER_DAY]
            pnl, ret, side, reason = simulate_breakout(bars, float(ohlcv["close"].iloc[i]), float(datr.iloc[i]), equity, k, rr, fee, slip, risk_pct)
            equity += pnl
        rows.append(DayRow(day.isoformat(), float(ohlcv["close"].iloc[i]), p_big, predicted_big, actual_big,
                           float(fut_abs.iloc[i] * 100), prev_big, traded and side != "NONE", side, pnl, ret, reason, equity))
        prev_big = actual_big
    df = pd.DataFrame([asdict(r) for r in rows])
    traded = df[df["traded"]]
    daily_ret = df["return_pct"] / 100
    metrics = {
        "days": int(len(df)),
        "big_day_share": float(df["actual_big"].mean()),
        "accuracy": float((df["predicted_big"] == df["actual_big"]).mean()),
        "auc": float(roc_auc_score(df["actual_big"], df["p_big"])) if df["actual_big"].nunique() == 2 else float("nan"),
        "naive_persistence_accuracy": float((df["naive_big"] == df["actual_big"]).mean()),
        "precision_big": float(df.loc[df["predicted_big"], "actual_big"].mean()) if df["predicted_big"].any() else 0.0,
        "predicted_big_days": int(df["predicted_big"].sum()),
        "trades": int(len(traded)),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "total_return_pct": float((df["equity"].iloc[-1] / 10_000 - 1) * 100),
        "sharpe": float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0.0,
        "max_drawdown_pct": float(-(df["equity"] / df["equity"].cummax() - 1).min() * 100),
        "exit_reasons": {k_: int(v) for k_, v in traded["exit_reason"].value_counts().items()},
        "no_breakout_days": int((df["exit_reason"] == "NO_BREAKOUT").sum()),
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--retrain-every", type=int, default=21)
    parser.add_argument("--k", type=float, default=0.5, help="breakout distance in daily ATR")
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.long:
        from scripts.fetch_long_history import LONG_CACHE
        ohlcv = storage.load_cached(LONG_CACHE)
        start, end = 1_100, 90
    else:
        ohlcv = storage.get_ohlcv()
        start, end = 365, 90
    span = start - end
    mid = end + span // 2

    results = {}
    # Tune the probability threshold on the older half only, validate on the newer half.
    for thr in (0.5, 0.55, 0.6, 0.65):
        _, m = run(ohlcv, start, mid, args.retrain_every, thr, args.k, args.rr, verbose=False)
        results[f"tune_thr{thr}"] = m
        print(f"[tune] thr={thr:.2f} acc {m['accuracy']:.1%} auc {m['auc']:.3f} naive {m['naive_persistence_accuracy']:.1%} precision {m['precision_big']:.1%} trades {m['trades']} ret {m['total_return_pct']:+.2f}% sharpe {m['sharpe']:.2f}", flush=True)
    _, m_all = run(ohlcv, start, mid, args.retrain_every, 0.0, args.k, args.rr, trade_all_days=True, verbose=False)
    results["tune_breakout_every_day"] = m_all
    print(f"[tune] breakout every day: trades {m_all['trades']} ret {m_all['total_return_pct']:+.2f}% sharpe {m_all['sharpe']:.2f}", flush=True)
    best_thr = max((0.5, 0.55, 0.6, 0.65), key=lambda t: results[f"tune_thr{t}"]["sharpe"])

    _, m_val = run(ohlcv, mid, end, args.retrain_every, best_thr, args.k, args.rr, verbose=False)
    _, m_val_all = run(ohlcv, mid, end, args.retrain_every, 0.0, args.k, args.rr, trade_all_days=True, verbose=False)
    results["validation_best"] = {"threshold": best_thr, **m_val}
    results["validation_breakout_every_day"] = m_val_all
    print(f"[val ] thr={best_thr:.2f} acc {m_val['accuracy']:.1%} auc {m_val['auc']:.3f} naive {m_val['naive_persistence_accuracy']:.1%} precision {m_val['precision_big']:.1%} trades {m_val['trades']} ret {m_val['total_return_pct']:+.2f}% sharpe {m_val['sharpe']:.2f} dd {m_val['max_drawdown_pct']:.1f}%", flush=True)
    print(f"[val ] breakout every day: trades {m_val_all['trades']} ret {m_val_all['total_return_pct']:+.2f}% sharpe {m_val_all['sharpe']:.2f}", flush=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"\n## {now} — Volatility target + breakout strategy ({'6-year' if args.long else '2-year'} data, k={args.k} ATR, R:R {args.rr})\n",
        "Predict whether the next 24 h |return| exceeds the trailing 90-day median (\"big day\"); trade an OCO breakout only on predicted big days.\n",
        "| Split | Threshold | Big-day acc. | AUC | Naive (persistence) | Precision | Trades | Win rate | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for thr in (0.5, 0.55, 0.6, 0.65):
        m = results[f"tune_thr{thr}"]
        lines.append(f"| tune | {thr:.2f} | {m['accuracy']:.1%} | {m['auc']:.3f} | {m['naive_persistence_accuracy']:.1%} | {m['precision_big']:.1%} | {m['trades']} | {m['win_rate']:.1%} | {m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.1f}% |")
    lines.append(f"| tune | every day | – | – | – | – | {m_all['trades']} | {m_all['win_rate']:.1%} | {m_all['total_return_pct']:+.2f}% | {m_all['sharpe']:.2f} | {m_all['max_drawdown_pct']:.1f}% |")
    lines.append(f"| **validation** | {best_thr:.2f} | {m_val['accuracy']:.1%} | {m_val['auc']:.3f} | {m_val['naive_persistence_accuracy']:.1%} | {m_val['precision_big']:.1%} | {m_val['trades']} | {m_val['win_rate']:.1%} | {m_val['total_return_pct']:+.2f}% | {m_val['sharpe']:.2f} | {m_val['max_drawdown_pct']:.1f}% |")
    lines.append(f"| validation | every day | – | – | – | – | {m_val_all['trades']} | {m_val_all['win_rate']:.1%} | {m_val_all['total_return_pct']:+.2f}% | {m_val_all['sharpe']:.2f} | {m_val_all['max_drawdown_pct']:.1f}% |")
    lines.append(f"\n* Big-day share {m_val['big_day_share']:.1%}; exits on validation {m_val['exit_reasons']}, days with no breakout {m_val['no_breakout_days']}.")
    lines.append(f"* Runtime {time.perf_counter() - t0:.0f}s.\n")
    text = "\n".join(lines)
    print(text)
    if not args.no_append:
        append_learnings(text)
    out = settings.models_dir / "volatility_target.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
