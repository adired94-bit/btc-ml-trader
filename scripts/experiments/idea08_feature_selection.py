"""Idea 08 - "Less is more": feature selection for the daily direction forecast.

Ranks the ~55 extended features by (a) permutation importance and (b) split gain,
both computed **only** on hourly bars whose 24-bar label is known before the first
evaluation day (zero look-ahead), and then re-runs the strict day-by-day walk-forward
with the top-8 / top-15 / top-25 features, without calendar features and without
volume-profile features. Every variant and the all-features baseline are scored the
same way on the tuning half (365 -> 227 days ago) and the validation half
(227 -> 90 days ago).

    venv\\Scripts\\python.exe scripts\\experiments\\idea08_feature_selection.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from src.backtest import walk_forward as WF  # noqa: E402
from src.data import storage  # noqa: E402
from src.data.processor import UP  # noqa: E402
from src.logging_config import get_logger  # noqa: E402
from src.models.ensemble import DirectionEnsemble  # noqa: E402

logger = get_logger("idea08")

# Fast, CPU-friendly settings shared by every run (rules: n_estimators <= 80, n_jobs = 2).
FAST_XGB = {**WF.WF_XGB_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_LGBM = {**WF.WF_LGBM_PARAMS, "n_estimators": 80, "n_jobs": 2}
FAST_REG = {**WF.WF_REG_PARAMS, "n_estimators": 80, "n_jobs": 2}
BASE_CFG = WF.WalkForwardConfig(
    feature_set="extended", retrain_every_days=14, xgb_params=FAST_XGB, lgbm_params=FAST_LGBM, reg_params=FAST_REG,
    label="ext+ens(all)",
)
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos")
VOLUME_PROFILE = ("dist_poc_pct", "dist_va_low_pct", "dist_va_high_pct")
PERM_REPEATS = 3
METRIC_KEYS = (
    "days", "directional_accuracy", "traded_days", "traded_directional_accuracy", "total_return_pct", "sharpe_ratio",
    "max_drawdown_pct",
)


# ----------------------------------------------------------------------
# Importance (pre-evaluation data only)
# ----------------------------------------------------------------------


def importance_cutoff(ohlcv: pd.DataFrame, cfg: WF.WalkForwardConfig) -> pd.Timestamp:
    """Last bar whose label is known at the close of the first evaluation day - identical to the
    ``train_end`` the walk-forward itself uses for its first retrain."""
    idx = ohlcv.index
    start, _ = WF.evaluation_days(idx, cfg)
    closes = WF.day_close_bars(idx)
    first_day = next(d for d in closes.index if d >= start)
    i = int(np.searchsorted(idx, closes[first_day]))
    return idx[i - cfg.horizon_bars]


def pre_window_training_set(data: WF.PreparedData, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    mask = (data.features.index <= cutoff) & (data.direction >= 0).to_numpy()
    X = data.features[mask].dropna()
    y = data.direction.loc[X.index]
    return X, y


def fit_fast_ensemble(X: pd.DataFrame, y: pd.Series) -> DirectionEnsemble:
    return DirectionEnsemble(xgb_params=FAST_XGB, lgbm_params=FAST_LGBM).fit(X, y)


def gain_importance(model: DirectionEnsemble, columns: list[str]) -> pd.Series:
    """Normalised total-gain importance averaged over the two boosters."""
    xgb_scores = model.xgb_model.get_booster().get_score(importance_type="total_gain")
    xgb_gain = np.array([xgb_scores.get(f"f{i}", 0.0) for i in range(len(columns))], dtype=float)
    lgb_gain = np.asarray(model.lgbm_model.booster_.feature_importance(importance_type="gain"), dtype=float)
    xgb_gain = xgb_gain / xgb_gain.sum() if xgb_gain.sum() > 0 else xgb_gain
    lgb_gain = lgb_gain / lgb_gain.sum() if lgb_gain.sum() > 0 else lgb_gain
    return pd.Series(0.5 * xgb_gain + 0.5 * lgb_gain, index=columns).sort_values(ascending=False)


def _log_loss(p_up: np.ndarray, y_up: np.ndarray) -> float:
    p = np.clip(p_up, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_up * np.log(p) + (1 - y_up) * np.log(1 - p)))


def permutation_importance(
    X: pd.DataFrame, y: pd.Series, horizon_bars: int, holdout_frac: float = 0.25, repeats: int = PERM_REPEATS,
    seed: int = 42,
) -> tuple[pd.Series, dict[str, Any]]:
    """Time-ordered permutation importance: fit on the first (1 - holdout_frac) of the pre-window rows,
    leave a ``horizon_bars`` gap, and measure the log-loss increase when each feature is shuffled in
    the trailing hold-out block. Uses the walk-forward booster settings."""
    n = len(X)
    split = int(n * (1 - holdout_frac))
    X_fit, y_fit = X.iloc[:split], y.iloc[:split]
    X_hold, y_hold = X.iloc[split + horizon_bars:], y.iloc[split + horizon_bars:]
    model = fit_fast_ensemble(X_fit, y_fit)
    y_up = (y_hold.to_numpy() == UP).astype(float)
    base_p = model.predict_proba(X_hold)[:, UP]
    base_loss = _log_loss(base_p, y_up)
    base_acc = float(((base_p >= 0.5) == (y_up == 1)).mean())
    rng = np.random.default_rng(seed)
    rows: dict[str, float] = {}
    for col in X.columns:
        deltas = []
        for _ in range(repeats):
            Xp = X_hold.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(_log_loss(model.predict_proba(Xp)[:, UP], y_up) - base_loss)
        rows[col] = float(np.mean(deltas))
    info = {
        "fit_rows": int(len(X_fit)), "holdout_rows": int(len(X_hold)), "holdout_start": str(X_hold.index[0]),
        "holdout_end": str(X_hold.index[-1]), "base_logloss": base_loss, "base_accuracy": base_acc, "repeats": repeats,
    }
    return pd.Series(rows).sort_values(ascending=False), info


# ----------------------------------------------------------------------
# Walk-forward helpers
# ----------------------------------------------------------------------


def restrict(data: WF.PreparedData, columns: list[str]) -> WF.PreparedData:
    return replace(data, features=data.features[list(columns)])


def run_half(ohlcv: pd.DataFrame, data: WF.PreparedData, cfg: WF.WalkForwardConfig, half: str, mid: int) -> dict[str, Any]:
    window = {"eval_end_days_ago": mid} if half == "tune" else {"eval_start_days_ago": mid}
    res = WF.run_walk_forward(ohlcv, cfg.copy(**window), data, verbose=False)
    m = {k: res.metrics[k] for k in METRIC_KEYS}
    m.update({"momentum_30d_accuracy": res.metrics["momentum_30d_accuracy"], "always_up_accuracy": res.metrics["always_up_accuracy"],
              "runtime_seconds": round(res.runtime_seconds, 1)})
    return m


def fmt(label: str, half: str, m: dict[str, Any]) -> str:
    return (
        f"[{half:<4}] {label:<22} days {m['days']:>3} hit {m['directional_accuracy']:.1%} traded {m['traded_days']:>3} "
        f"@ {m['traded_directional_accuracy']:.1%} ret {m['total_return_pct']:+.2f}% sharpe {m['sharpe_ratio']:.2f} "
        f"dd {m['max_drawdown_pct']:.2f}% ({m['runtime_seconds']:.0f}s)"
    )


def md_table(results: dict[str, dict[str, dict[str, Any]]], n_feats: dict[str, int]) -> str:
    head = "| Config | #feat | Half | Days | Hit | Traded | Traded hit | Return | Sharpe | Max DD |\n|---|---|---|---|---|---|---|---|---|---|"
    body = []
    for label, halves in results.items():
        for half, m in halves.items():
            body.append(
                f"| {label} | {n_feats[label]} | {half} | {m['days']} | {m['directional_accuracy']:.1%} | {m['traded_days']} | "
                f"{m['traded_directional_accuracy']:.1%} | {m['total_return_pct']:+.2f}% | {m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.2f}% |"
            )
    return "\n".join([head, *body])


# ----------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    try:
        ohlcv = storage.get_ohlcv()
    except Exception as exc:  # noqa: BLE001 - offline is fine, the cache is what we evaluate on
        logger.warning("get_ohlcv failed (%s); using the cached CSV", exc)
        ohlcv = storage.load_cached()
    cfg = BASE_CFG
    span = cfg.eval_start_days_ago - cfg.eval_end_days_ago
    mid = cfg.eval_end_days_ago + span // 2
    data = WF.prepare_data(ohlcv, cfg.horizon_bars, cfg.feature_set)
    all_cols = list(data.features.columns)
    print(f"data {ohlcv.index[0]} -> {ohlcv.index[-1]} ({len(ohlcv)} bars); {len(all_cols)} extended features; mid={mid}")

    # --- 1. importance on pre-window data only ---------------------------------------------
    cutoff = importance_cutoff(ohlcv, cfg.copy(eval_end_days_ago=mid))
    X_pre, y_pre = pre_window_training_set(data, cutoff)
    print(f"importance data: {len(X_pre)} rows, {X_pre.index[0]} -> {X_pre.index[-1]} (cutoff {cutoff})")
    t1 = time.perf_counter()
    perm, perm_info = permutation_importance(X_pre, y_pre, cfg.horizon_bars)
    print(f"permutation importance done in {time.perf_counter() - t1:.0f}s; holdout base logloss {perm_info['base_logloss']:.4f} acc {perm_info['base_accuracy']:.1%}")
    t1 = time.perf_counter()
    gain = gain_importance(fit_fast_ensemble(X_pre, y_pre), all_cols)
    print(f"gain importance done in {time.perf_counter() - t1:.0f}s")

    ranks = pd.DataFrame({
        "perm_delta_logloss": perm, "gain_share": gain,
        "perm_rank": perm.rank(ascending=False), "gain_rank": gain.rank(ascending=False),
    })
    ranks["mean_rank"] = (ranks["perm_rank"] + ranks["gain_rank"]) / 2
    ranks = ranks.sort_values(["mean_rank", "perm_rank"])
    print("\nTop 25 by mean rank (perm + gain):")
    print(ranks.head(25).round(4).to_string())
    print(f"\nfeatures with NEGATIVE permutation importance (shuffling them helps): "
          f"{[c for c in all_cols if perm[c] < 0]}")

    # --- 2. feature subsets ----------------------------------------------------------------
    ordered = list(ranks.index)
    subsets: dict[str, list[str]] = {
        "ext+ens(all)": all_cols,
        "top8": ordered[:8],
        "top15": ordered[:15],
        "top25": ordered[:25],
        "top15_perm_only": list(perm.index[:15]),
        "no_calendar": [c for c in all_cols if c not in CALENDAR],
        "no_volume_profile": [c for c in all_cols if c not in VOLUME_PROFILE],
    }
    n_feats = {k: len(v) for k, v in subsets.items()}

    # --- 3. walk-forward, tune then validation, same procedure for every subset -------------
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for label, cols in subsets.items():
        sub = restrict(data, cols)
        results[label] = {}
        for half in ("tune", "validation"):
            m = run_half(ohlcv, sub, cfg.copy(label=label), half, mid)
            results[label][half] = m
            print(fmt(label, half, m), flush=True)

    # --- 4. save + report ---------------------------------------------------------------------
    table = md_table(results, n_feats)
    total = time.perf_counter() - t0
    payload = {
        "idea": "08 feature selection (less is more)",
        "config": cfg.to_dict(), "mid_days_ago": mid, "importance_cutoff": str(cutoff), "importance_rows": int(len(X_pre)),
        "permutation_info": perm_info,
        "ranking": ranks.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "subsets": subsets, "results": results, "markdown_table": table, "runtime_seconds": round(total, 1),
    }
    out = settings.models_dir / "idea08_feature_selection.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("\n" + table)
    print(f"\nsaved {out} ({total:.0f}s total)")


if __name__ == "__main__":
    main()
