"""Out-of-sample parameter sweep for the strategy layer.

Fits the ensemble once per walk-forward fold (exactly like ``src.backtest.engine``),
keeps the out-of-sample probabilities of both boosters, and then evaluates many
risk / threshold configurations on those *fixed* predictions. No model is retrained
per configuration, so the sweep is cheap and every row is fully out-of-sample.

    venv\\Scripts\\python.exe scripts\\param_sweep.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.backtest.engine import compute_metrics, simulate_trades  # noqa: E402
from src.data import storage  # noqa: E402
from src.data.processor import build_dataset  # noqa: E402
from src.models.ensemble import DirectionEnsemble  # noqa: E402
from src.risk.management import RiskManager  # noqa: E402

TRAIN_WINDOW, TEST_WINDOW = 6_000, 1_000


def oos_probabilities(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    gap = settings.prediction_horizon
    parts = []
    start = 0
    while start + TRAIN_WINDOW + gap < len(X):
        train_end = start + TRAIN_WINDOW
        test_start, test_end = train_end + gap, min(train_end + gap + TEST_WINDOW, len(X))
        if test_end - test_start < 10:
            break
        model = DirectionEnsemble().fit(X.iloc[start:train_end], y["direction"].iloc[start:train_end])
        test_X = X.iloc[test_start:test_end]
        ens = model.predict_proba(test_X)
        comp = model.predict_proba_components(test_X)
        frame = pd.DataFrame(ens, index=test_X.index, columns=["p_down", "p_flat", "p_up"])
        frame["xgb_arg"] = comp["xgboost"].argmax(axis=1)
        frame["lgb_arg"] = comp["lightgbm"].argmax(axis=1)
        parts.append(frame)
        print(f"fold {len(parts)}: {test_X.index[0]} -> {test_X.index[-1]}", flush=True)
        start += TEST_WINDOW
    return pd.concat(parts)


def main() -> None:
    df = storage.get_ohlcv()
    X, y, ind = build_dataset(df)
    probs = oos_probabilities(X, y)
    rows = []
    for thr, atr_mult, rr, hold, agree in itertools.product([0.50, 0.55, 0.60, 0.65], [1.5, 2.0, 3.0], [2.0, 3.0], [24, 48], [False, True]):
        p = probs.copy()
        if agree:  # require both boosters to agree on the traded side
            disagree = p["xgb_arg"] != p["lgb_arg"]
            p.loc[disagree, ["p_up", "p_down"]] = 0.0
        risk = RiskManager(equity=10_000, atr_multiplier=atr_mult, reward_risk_ratios=[rr])
        trades, curve = simulate_trades(df, p[["p_down", "p_flat", "p_up"]], ind["atr_14"], risk, thr, rr, hold,
                                        settings.backtest_fee_pct, settings.backtest_slippage_pct, True)
        m = compute_metrics(trades, curve, df, settings.timeframe, 10_000)
        rows.append({"threshold": thr, "atr_mult": atr_mult, "rr": rr, "hold": hold, "agree": agree, "trades": m.total_trades,
                     "win_rate": round(m.win_rate, 3), "pf": round(m.profit_factor, 2), "return_pct": round(m.total_return_pct, 2),
                     "sharpe": round(m.sharpe_ratio, 2), "max_dd": round(m.max_drawdown_pct, 2)})
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\nOOS window: {probs.index[0]} -> {probs.index[-1]} ({len(probs)} bars)")
    print(table.head(15).to_string(index=False))
    print("\n... worst 5:")
    print(table.tail(5).to_string(index=False))
    out = settings.models_dir / "param_sweep.json"
    out.write_text(json.dumps({"window": [probs.index[0].isoformat(), probs.index[-1].isoformat()], "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
