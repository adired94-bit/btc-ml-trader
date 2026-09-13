"""Idea 03 - cross-asset context (ETH/USDT, optionally SOL/USDT) for the daily BTC direction forecast.

Downloads ~2 years of hourly ETH/USDT (and SOL/USDT when quick) from the configured exchange, caches
them under ``data/<SYMBOL>_1h.csv`` and adds strictly causal cross-asset features to the extended
feature set:

* ETH 1h / 24h / 72h returns
* ETH/BTC ratio: 24h and 168h change, distance to its 30-day mean
* ETH-minus-BTC 24h return (relative strength)
* rolling 72h (and 336h) correlation of hourly ETH and BTC log returns
* ETH volume ratio (24h), ETH/BTC 24h realised-volatility ratio, 7-day beta of ETH on BTC

Evaluation follows scripts/improve.py: tune on the older half of the 365->90 window (365->227),
validate on the newer half (227->90), same trade simulation, same metrics. Baseline = plain
extended-features XGB+LGBM ensemble with identical (light) hyper-parameters.

    venv\\Scripts\\python.exe scripts\\experiments\\idea03_cross_asset.py
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
from src.data import fetcher, storage  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger("idea03")

LIGHT = {
    "xgb_params": {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2},
    "lgbm_params": {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2},
    "reg_params": {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2},
}
REGIME_W = {"trend_up": 1.5, "high_volatility": 1.5}
CROSS_PREFIXES = ("eth_", "sol_")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def load_alt_symbol(symbol: str, limit: int, force: bool = False) -> pd.DataFrame | None:
    """Return hourly OHLCV for ``symbol`` from data/<slug>_1h.csv, downloading it if missing."""
    slug = symbol.replace("/", "_")
    path = settings.data_dir / f"{slug}_{settings.timeframe}.csv"
    if not force:
        cached = storage.load_cached(path)
        if cached is not None and len(cached) >= limit * 0.95:
            return cached
    try:
        handle = fetcher.connect(symbol)
        df = fetcher.fetch_ohlcv_history(limit=limit, handle=handle)
    except fetcher.MarketDataError as exc:
        logger.error("Could not download %s: %s", symbol, exc)
        return None
    storage.save_cache(df, path)
    return df


def align(alt: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align an alt-asset frame to the BTC hourly index by bar open time (ffill gaps <= 3h)."""
    return alt.reindex(index).ffill(limit=3)


def cross_asset_features(btc: pd.DataFrame, alt: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Strictly causal cross-asset features. Bar t only uses alt/btc bars <= t (same open time)."""
    a = align(alt, btc.index)
    ac, bc = a["close"], btc["close"]
    feats = pd.DataFrame(index=btc.index)
    for lag in (1, 24, 72):
        feats[f"{prefix}ret_{lag}"] = ac.pct_change(lag)
    ratio = ac / bc
    feats[f"{prefix}btc_ratio_chg_24"] = ratio.pct_change(24)
    feats[f"{prefix}btc_ratio_chg_168"] = ratio.pct_change(168)
    feats[f"{prefix}btc_ratio_dist_720"] = ratio / ratio.rolling(720, min_periods=240).mean() - 1.0
    feats[f"{prefix}minus_btc_24"] = ac.pct_change(24) - bc.pct_change(24)
    la, lb = np.log(ac / ac.shift(1)), np.log(bc / bc.shift(1))
    feats[f"{prefix}btc_corr_72"] = la.rolling(72, min_periods=48).corr(lb)
    feats[f"{prefix}btc_corr_336"] = la.rolling(336, min_periods=168).corr(lb)
    vol_ma = a["volume"].rolling(24).mean().replace(0.0, np.nan)
    feats[f"{prefix}volume_ratio_24"] = a["volume"] / vol_ma
    feats[f"{prefix}volume_ratio_24_168"] = (
        a["volume"].rolling(24).mean() / a["volume"].rolling(168).mean().replace(0.0, np.nan)
    )
    feats[f"{prefix}vol_ratio_24"] = la.rolling(24).std() / lb.rolling(24).std().replace(0.0, np.nan)
    cov = la.rolling(168, min_periods=100).cov(lb)  # 7-day beta of alt on BTC
    feats[f"{prefix}beta_168"] = cov / lb.rolling(168, min_periods=100).var().replace(0.0, np.nan)
    return feats.replace([np.inf, -np.inf], np.nan)


CORE_KEYS = ("ret_", "ratio_chg", "minus_btc", "corr_72", "volume_ratio_24")


def build_prepared(btc: pd.DataFrame, alts: dict[str, pd.DataFrame], horizon: int, subset: str) -> WF.PreparedData:
    """PreparedData whose feature frame = extended features (+ cross-asset block depending on ``subset``)."""
    base = WF.prepare_data(btc, horizon, "extended")
    feats = base.features
    if subset != "none":
        blocks = [feats]
        for sym, df in alts.items():
            block = cross_asset_features(btc, df, sym.split("/")[0].lower() + "_")
            if subset == "core":  # the compact set named in the idea brief
                keep = [c for c in block.columns if any(k in c for k in CORE_KEYS) and not c.endswith("ratio_24_168")]
                block = block[keep]
            blocks.append(block)
        feats = pd.concat(blocks, axis=1)
    return WF.PreparedData(base.ohlcv, base.indicators, feats, base.future_return, base.direction, base.regimes, base.daily_atr)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------


def row(label: str, half: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label, "half": half, "days": m["days"], "hit": round(m["directional_accuracy"], 4),
        "momentum_hit": round(m["momentum_30d_accuracy"], 4), "always_up": round(m["always_up_accuracy"], 4),
        "traded_days": m["traded_days"], "traded_hit": round(m["traded_directional_accuracy"], 4),
        "return_pct": round(m["total_return_pct"], 2), "sharpe": round(m["sharpe_ratio"], 2),
        "max_dd": round(m["max_drawdown_pct"], 2), "mae_pct": round(m["mae_pct"], 3),
    }


def md_table(rows: list[dict[str, Any]]) -> str:
    head = "| Config | Half | Days | Hit | Mom30 | Traded | Traded hit | Return | Sharpe | MaxDD |\n|---|---|---|---|---|---|---|---|---|---|"
    body = [
        f"| {r['label']} | {r['half']} | {r['days']} | {r['hit']:.1%} | {r['momentum_hit']:.1%} | {r['traded_days']} | "
        f"{r['traded_hit']:.1%} | {r['return_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% |"
        for r in rows
    ]
    return "\n".join([head, *body])


def feature_importance(data: WF.PreparedData, cfg: WF.WalkForwardConfig, train_end: pd.Timestamp) -> dict[str, float]:
    """Gain importance of the LightGBM leg fitted once at ``train_end`` (diagnostic only)."""
    m = WF.DailyModels(cfg).fit(data, train_end)
    gain = m.direction.lgbm_model.booster_.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    return {name: round(float(g) / total, 4) for name, g in sorted(zip(m.feature_names, gain), key=lambda t: -t[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sol", action="store_true", help="skip SOL/USDT")
    parser.add_argument("--refresh", action="store_true", help="re-download alt data")
    args = parser.parse_args()
    t0 = time.perf_counter()

    try:
        btc = storage.get_ohlcv()
    except Exception as exc:  # noqa: BLE001 - offline: fall back to the cache
        logger.warning("get_ohlcv failed (%s); using cached BTC data", exc)
        btc = storage.load_cached()
    assert btc is not None
    limit = settings.history_candles
    alts: dict[str, pd.DataFrame] = {}
    eth = load_alt_symbol("ETH/USDT", limit, args.refresh)
    if eth is None:
        raise SystemExit("ETH/USDT data unavailable")
    alts["ETH/USDT"] = eth
    if not args.no_sol:
        sol = load_alt_symbol("SOL/USDT", limit, args.refresh)
        if sol is not None:
            alts["SOL/USDT"] = sol
    for sym, df in alts.items():
        aligned = align(df, btc.index)
        logger.info(
            "%s: %d bars %s -> %s; %d BTC bars without alt data",
            sym, len(df), df.index[0], df.index[-1], int(aligned["close"].isna().sum()),
        )

    base = WF.WalkForwardConfig(feature_set="extended", retrain_every_days=14, **LIGHT)
    span = base.eval_start_days_ago - base.eval_end_days_ago
    mid = base.eval_end_days_ago + span // 2  # 227
    halves = {"tune": dict(eval_end_days_ago=mid), "validation": dict(eval_start_days_ago=mid)}

    eth_only = {"ETH/USDT": alts["ETH/USDT"]}
    datasets = {
        "none": build_prepared(btc, {}, base.horizon_bars, "none"),
        "eth_core": build_prepared(btc, eth_only, base.horizon_bars, "core"),
        "eth_full": build_prepared(btc, eth_only, base.horizon_bars, "full"),
    }
    if len(alts) > 1:
        datasets["eth_sol_full"] = build_prepared(btc, alts, base.horizon_bars, "full")
    for k, d in datasets.items():
        logger.info("dataset %s: %d features", k, d.features.shape[1])

    experiments: list[tuple[str, str, WF.WalkForwardConfig]] = [
        ("baseline ext+ens", "none", base.copy(label="baseline")),
        ("baseline ext+ens+regime_w", "none", base.copy(label="baseline+regime_w", regime_weights=REGIME_W)),
        ("baseline ext+logreg", "none", base.copy(label="baseline+logreg", model="logreg", logreg_c=0.05)),
        ("eth_core+ens", "eth_core", base.copy(label="eth_core+ens")),
        ("eth_full+ens", "eth_full", base.copy(label="eth_full+ens")),
        ("eth_full+ens+regime_w", "eth_full", base.copy(label="eth_full+ens+regime_w", regime_weights=REGIME_W)),
        ("eth_core+logreg", "eth_core", base.copy(label="eth_core+logreg", model="logreg", logreg_c=0.05)),
    ]
    if "eth_sol_full" in datasets:
        experiments.append(("eth_sol_full+ens", "eth_sol_full", base.copy(label="eth_sol_full+ens")))

    results: dict[str, list[dict[str, Any]]] = {"tune": [], "validation": []}
    for label, ds, cfg in experiments:
        for half, override in halves.items():
            res = WF.run_walk_forward(btc, cfg.copy(**override), datasets[ds], verbose=False)
            r = row(label, half, res.metrics)
            results[half].append(r)
            print(
                f"[{half:<10}] {label:<28} hit {r['hit']:.1%} traded {r['traded_hit']:.1%} ({r['traded_days']}) "
                f"ret {r['return_pct']:+.2f}% sharpe {r['sharpe']:.2f} dd {r['max_dd']:.2f}% ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

    # Diagnostic: gain share of the cross-asset features in the ETH-full ensemble fitted at the end of the tune half.
    idx = btc.index
    tune_end_day = (idx[-1] - pd.Timedelta(days=mid)).floor("D")
    train_end = idx[idx <= tune_end_day][-1 - base.horizon_bars]
    imp = feature_importance(datasets["eth_full"], base, train_end)
    cross_share = sum(v for k, v in imp.items() if k.startswith(CROSS_PREFIXES))
    top_cross = {k: v for k, v in list(imp.items())[:40] if k.startswith(CROSS_PREFIXES)}
    print(f"\nLGBM gain share of cross-asset features at {train_end.date()}: {cross_share:.1%}; top-40 cross features: {top_cross}")

    report = "\n### Tune half (365->227)\n" + md_table(results["tune"]) + "\n\n### Validation half (227->90)\n" + md_table(results["validation"])
    print(report)
    out = settings.models_dir / "idea03_cross_asset.json"
    out.write_text(json.dumps({
        "idea": "cross-asset context (ETH/USDT, SOL/USDT) features on top of the extended set",
        "window": {"tune": [base.eval_start_days_ago, mid], "validation": [mid, base.eval_end_days_ago]},
        "light_params": LIGHT, "alts": {k: [str(v.index[0]), str(v.index[-1]), len(v)] for k, v in alts.items()},
        "n_features": {k: int(d.features.shape[1]) for k, d in datasets.items()},
        "tune": results["tune"], "validation": results["validation"],
        "lgbm_gain_importance_eth_full_at_tune_end": imp, "cross_asset_gain_share": cross_share,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"saved {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
