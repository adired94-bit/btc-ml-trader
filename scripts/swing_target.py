"""Swing-horizon experiment: predict the direction of the next 5 / 10 / 21 trading days
for BTC and MSTR on DAILY bars, with the same tune / validation discipline as everything else.

Data: Yahoo Finance daily closes (BTC-USD, MSTR, ^GSPC, GC=F, DX-Y.NYB) plus the free on-chain /
sentiment frame from ``src.data.onchain`` (MVRV, stable-coin cap, Fear & Greed, active
addresses ...). Every feature at day *t* uses closes up to *t*; on-chain values are lagged one
day. Labels for horizon H are known only at t+H, so the model retrained at day T uses rows
with t + H <= T.

    venv\\Scripts\\python.exe scripts\\swing_target.py                # BTC + MSTR, H = 5, 10, 21
    venv\\Scripts\\python.exe scripts\\swing_target.py --assets MSTR --start 2021-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest.walk_forward import append_learnings  # noqa: E402
from src.data.onchain import daily_onchain_features, load_onchain  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("swing")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console cannot print arrows otherwise

TICKERS = {"BTC": "BTC-USD", "MSTR": "MSTR"}
MACRO = {"^GSPC": "spx", "GC=F": "gold", "DX-Y.NYB": "dxy", "BTC-USD": "btc"}
HORIZONS = (5, 10, 21)
LGB_PARAMS = {"n_estimators": 150, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "subsample": 0.8,
              "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0, "verbose": -1, "n_jobs": -1, "random_state": 42}
FEE_ROUND_TRIP = 0.001


def yahoo_daily(ticker: str) -> pd.DataFrame:
    cache = settings.data_dir / f"yf_{ticker.replace('^', '').replace('=', '').replace('.', '_')}_1d.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 6 * 3600:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_csv(cache)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").normalize()
    df.columns = [c.lower() for c in df.columns]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def price_features(px: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    c, h, l, v = px["close"], px["high"], px["low"], px["volume"]
    f = pd.DataFrame(index=px.index)
    for n in (1, 5, 10, 21, 63, 126, 252):
        f[f"{prefix}ret_{n}"] = c.pct_change(n)
    lr = np.log(c / c.shift(1))
    for n in (10, 21, 63):
        f[f"{prefix}vol_{n}"] = lr.rolling(n).std()
    f[f"{prefix}vol_ratio_10_63"] = f[f"{prefix}vol_10"] / f[f"{prefix}vol_63"].replace(0, np.nan)
    for n in (10, 20, 50, 200):
        e = c.ewm(span=n, adjust=False).mean()
        f[f"{prefix}dist_ema_{n}"] = c / e - 1
        f[f"{prefix}ema_{n}_slope"] = e.pct_change(5)
    f[f"{prefix}rsi_14"] = _rsi(c) / 100
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f[f"{prefix}atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    hi52, lo52 = h.rolling(252, min_periods=100).max(), l.rolling(252, min_periods=100).min()
    f[f"{prefix}range_pos_52w"] = (c - lo52) / (hi52 - lo52).replace(0, np.nan)
    f[f"{prefix}dist_high_52w"] = c / hi52 - 1
    f[f"{prefix}drawdown_252"] = c / c.rolling(252, min_periods=100).max() - 1
    if v.notna().any() and (v > 0).any():
        f[f"{prefix}volume_z_63"] = (v - v.rolling(63).mean()) / v.rolling(63).std().replace(0, np.nan)
    f[f"{prefix}up_days_21"] = (c.pct_change() > 0).astype(float).rolling(21).mean()
    return f


def build_frame(asset: str, with_onchain: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = yahoo_daily(TICKERS[asset])
    feats = price_features(px)
    for tk, name in MACRO.items():
        if TICKERS[asset] == tk:
            continue
        m = yahoo_daily(tk)["close"].reindex(px.index, method="ffill")
        for n in (5, 21, 63):
            feats[f"{name}_ret_{n}"] = m.pct_change(n)
        feats[f"corr_{name}_63"] = px["close"].pct_change().rolling(63, min_periods=30).corr(m.pct_change())
    if asset == "MSTR":
        btc = yahoo_daily("BTC-USD")["close"].reindex(px.index, method="ffill")
        feats["mstr_btc_ret_gap_21"] = px["close"].pct_change(21) - btc.pct_change(21)
        feats["mstr_btc_ret_gap_63"] = px["close"].pct_change(63) - btc.pct_change(63)
        feats["mstr_btc_beta_63"] = px["close"].pct_change().rolling(63, min_periods=30).cov(btc.pct_change()) / btc.pct_change().rolling(63, min_periods=30).var()
    if with_onchain:
        daily = load_onchain()
        if daily is None:
            raise SystemExit("run `python -m src.data.onchain` first")
        oc = daily_onchain_features(daily, yahoo_daily("BTC-USD")["close"])
        oc.index = oc.index + pd.Timedelta(days=1)  # one-day publication lag
        oc = oc.reindex(px.index, method="ffill")
        for col in oc.columns:
            if col.startswith(("spx_", "dxy_", "btc_spx")):
                continue  # already covered by the macro block
            feats[f"oc_{col}"] = oc[col]
    dow = px.index.dayofweek.to_numpy()
    feats["dow_sin"], feats["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    return px, feats.replace([np.inf, -np.inf], np.nan)


def walk_forward(px: pd.DataFrame, feats: pd.DataFrame, horizon: int, start: pd.Timestamp, end: pd.Timestamp,
                 model_kind: str, retrain_every: int = 21, min_rows: int = 250) -> pd.DataFrame:
    close = px["close"]
    fwd = close.shift(-horizon) / close - 1
    label = (fwd > 0).astype(int)
    label[fwd.isna()] = -1
    days = [d for d in px.index if start <= d <= end and not np.isnan(fwd.loc[d])]
    model, last_fit, cols, rows = None, None, None, []
    for d in days:
        i = px.index.get_loc(d)
        train_end = px.index[i - horizon]
        if model is None or (d - last_fit).days >= retrain_every:
            mask = (feats.index <= train_end) & (label >= 0).to_numpy()
            X = feats[mask].dropna(thresh=int(0.7 * feats.shape[1]))
            X = X.fillna(X.median())
            y = label.loc[X.index]
            if len(X) < min_rows or y.nunique() < 2:
                continue
            cols = list(X.columns)
            if model_kind == "lgbm":
                model = lgb.LGBMClassifier(**LGB_PARAMS).fit(X.to_numpy(dtype=np.float32), y.to_numpy())
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(X.to_numpy(dtype=np.float32), y.to_numpy())
            med = X.median()
            last_fit = d
        x = feats.loc[[d], cols].fillna(med)
        if x.isna().any(axis=None):
            continue
        p_up = float(model.predict_proba(x.to_numpy(dtype=np.float32))[0][1])
        mom = float(close.loc[d] / close.iloc[max(0, i - 63)] - 1)
        rows.append({"day": d, "p_up": p_up, "fwd_ret": float(fwd.loc[d]), "actual_up": bool(fwd.loc[d] > 0),
                     "mom_up": mom > 0, "atr_pct": float(feats.loc[d].get("atr_pct", np.nan))})
    return pd.DataFrame(rows).set_index("day")


def evaluate(pred: pd.DataFrame, horizon: int, thr: float, long_only: bool) -> dict:
    df = pred.copy()
    df["pred_up"] = df["p_up"] >= 0.5
    hit_all = float((df["pred_up"] == df["actual_up"]).mean())
    nonov = df.iloc[::horizon]  # non-overlapping cycles
    hit_nonov = float((nonov["pred_up"] == nonov["actual_up"]).mean())
    mom_hit = float((nonov["mom_up"] == nonov["actual_up"]).mean())
    always_up = float(nonov["actual_up"].mean())
    # strategy on non-overlapping cycles: long if p_up >= thr, short if p_up <= 1-thr (unless long_only), else cash
    pos = np.where(nonov["p_up"] >= thr, 1, np.where((nonov["p_up"] <= 1 - thr) & (not long_only), -1, 0))
    strat = pos * nonov["fwd_ret"].to_numpy() - FEE_ROUND_TRIP * (pos != 0)
    equity = np.cumprod(1 + strat)
    bh = float(np.prod(1 + nonov["fwd_ret"]) - 1)
    per_year = 252 / horizon
    sharpe = float(strat.mean() / strat.std() * np.sqrt(per_year)) if strat.std() > 0 else 0.0
    dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    traded = pos != 0
    return {
        "days": int(len(df)), "cycles": int(len(nonov)), "hit_all_days": hit_all, "hit_cycles": hit_nonov,
        "momentum_63d_hit": mom_hit, "always_up_hit": always_up, "trades": int(traded.sum()),
        "trade_hit": float(((pos[traded] > 0) == nonov["actual_up"].to_numpy()[traded]).mean()) if traded.any() else 0.0,
        "strategy_return_pct": float((equity[-1] - 1) * 100), "buy_hold_return_pct": bh * 100,
        "sharpe": sharpe, "max_drawdown_pct": dd * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["BTC", "MSTR"])
    parser.add_argument("--start", default="2021-01-01", help="first evaluation day")
    parser.add_argument("--end-days-ago", type=int, default=30)
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=args.end_days_ago)
    mid = start + (end - start) / 2
    results: dict = {}
    lines = [f"\n## {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — Swing horizons (5 / 10 / 21 trading days) on daily bars: BTC and MSTR\n",
             f"Yahoo daily data + free on-chain/macro frame (1-day lag). Walk-forward, retrain every 21 days, evaluation {start.date()} → {end.date()}, "
             f"tune = first half, validation = second half (never used for selection). Strategy: on non-overlapping cycles go long if P(up) ≥ thr "
             f"(short if P(up) ≤ 1−thr for BTC only), else cash; 0.1 % round-trip cost. Momentum = sign of trailing 63-day return.\n",
             "| Asset | H | Features | Model | Half | Cycles | Hit (cycles) | Momentum | Always-up | Trades | Trade hit | Strategy | Buy&hold | Sharpe | Max DD |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for asset in args.assets:
        long_only = asset == "MSTR"
        for with_oc in (False, True):
            px, feats = build_frame(asset, with_oc)
            for H in HORIZONS:
                for kind in ("lgbm", "logreg"):
                    key = f"{asset}|H{H}|{'oc' if with_oc else 'px'}|{kind}"
                    try:
                        pred_t = walk_forward(px, feats, H, start, mid, kind)
                        pred_v = walk_forward(px, feats, H, mid, end, kind)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("%s failed: %s", key, exc)
                        continue
                    if pred_t.empty or pred_v.empty:
                        continue
                    # choose threshold on tune only
                    best_thr, best = 0.5, None
                    for thr in (0.5, 0.55, 0.6):
                        m = evaluate(pred_t, H, thr, long_only)
                        if best is None or m["sharpe"] > best["sharpe"]:
                            best_thr, best = thr, m
                    mv = evaluate(pred_v, H, best_thr, long_only)
                    results[key] = {"threshold": best_thr, "tune": best, "validation": mv}
                    for half, m in (("tune", best), ("validation", mv)):
                        lines.append(f"| {asset} | {H} | {'px+oc' if with_oc else 'px'} | {kind} | {half} | {m['cycles']} | {m['hit_cycles']:.1%} | {m['momentum_63d_hit']:.1%} | {m['always_up_hit']:.1%} | {m['trades']} | {m['trade_hit']:.1%} | {m['strategy_return_pct']:+.1f}% | {m['buy_hold_return_pct']:+.1f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.1f}% |")
                    print(f"{key:<26} thr {best_thr:.2f} | tune hit {best['hit_cycles']:.1%} (mom {best['momentum_63d_hit']:.1%}) strat {best['strategy_return_pct']:+.1f}% vs B&H {best['buy_hold_return_pct']:+.1f}% | val hit {mv['hit_cycles']:.1%} (mom {mv['momentum_63d_hit']:.1%}) strat {mv['strategy_return_pct']:+.1f}% vs B&H {mv['buy_hold_return_pct']:+.1f}% sharpe {mv['sharpe']:.2f}", flush=True)
    lines.append(f"\n* Runtime {time.perf_counter() - t0:.0f}s.\n")
    text = "\n".join(lines)
    print(text)
    if not args.no_append:
        append_learnings(text)
    out = settings.models_dir / "experiments" / "swing_target.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
