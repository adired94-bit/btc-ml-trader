"""Streamlit dashboard for the BTC/USDT ML signal platform.

Run with::

    streamlit run app.py --server.port 8501

The dashboard is a thin client over the FastAPI server (``src/api/main.py``).
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from config import settings

st.set_page_config(page_title="BTC/USDT ML Signals", page_icon="₿", layout="wide")



# ----------------------------------------------------------------------
# Transport: HTTP to the FastAPI server when reachable, otherwise in-process
# (single-process deployments such as Streamlit Community Cloud).
# ----------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _embedded_client():
    from src.api.embedded import get_client

    return get_client()


def _http_alive(base: str) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _detail(exc: Exception) -> str:
    return getattr(exc, "detail", None) or str(exc)

ACTION_COLORS = {"LONG": "#35C98D", "SHORT": "#E5615E", "WAIT": "#E8A33D"}


# ----------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------


def api_get(base: str, path: str, params: dict[str, Any] | None = None, timeout: int = 30, silent: bool = False) -> dict[str, Any] | None:
    if base == "embedded":
        try:
            return _embedded_client().get(path, params)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user like an HTTP error
            if not silent:
                st.warning(f"{path}: {_detail(exc)}")
            return None
    try:
        resp = requests.get(f"{base}{path}", params=params, timeout=timeout)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            if not silent:
                st.warning(f"API {path} -> {resp.status_code}: {detail}")
            return None
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {base}{path}: {exc}")
        return None


def api_post(base: str, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any] | None:
    if base == "embedded":
        try:
            return _embedded_client().post(path, payload)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{path}: {_detail(exc)}")
            return None
    try:
        resp = requests.post(f"{base}{path}", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            st.warning(f"API {path} -> {resp.status_code}: {detail}")
            return None
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {base}{path}: {exc}")
        return None


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.title("₿ Control panel")
    # A widget-keyed session_state entry may only be written before the widget is instantiated,
    # so an unreachable URL is swapped for "embedded" here, on the rerun requested further down.
    unreachable = st.session_state.pop("api_base_unreachable", None)
    if unreachable:
        st.session_state["api_base"] = "embedded"
    if "api_base" not in st.session_state:
        st.session_state["api_base"] = settings.api_url if _http_alive(settings.api_url) else "embedded"
    api_base = st.text_input("API URL (or 'embedded')", key="api_base").rstrip("/")
    if unreachable:
        st.warning(f"API at {unreachable} is unreachable - switched to in-process mode.")
    if api_base == "embedded":
        st.caption("Running in-process: no separate API server needed.")
    st.subheader("Signal")
    threshold = st.slider("Probability threshold", 0.34, 0.90, float(settings.signal_probability_threshold), 0.01)
    st.subheader("Risk")
    equity = st.number_input("Account equity (USDT)", min_value=100.0, value=float(settings.account_equity), step=500.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, float(settings.risk_per_trade_pct * 100), 0.1) / 100
    atr_mult = st.slider("ATR stop multiplier", 0.5, 4.0, float(settings.atr_stop_multiplier), 0.1)
    st.subheader("Chart")
    n_candles = st.slider("Candles shown", 100, 1500, 300, 50)
    st.subheader("Refresh")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.slider("Interval (s)", 15, 300, 60, 5)
    if st.button("Refresh now"):
        st.rerun()

health = api_get(api_base, "/health", timeout=10, silent=True)
if health is None and api_base != "embedded":
    st.session_state["api_base_unreachable"] = api_base
    st.rerun()
if health is None:
    st.error("Neither the API server nor the in-process engine could start. Check the logs.")
    st.stop()

# ----------------------------------------------------------------------
# Header: ticker + signal
# ----------------------------------------------------------------------

with st.spinner("Loading live market data and model signal..."):
    ticker = api_get(api_base, "/market/ticker", timeout=15)
    signal = api_get(api_base, "/signal/latest", {"threshold": threshold, "equity": equity, "risk_per_trade_pct": risk_pct, "atr_multiplier": atr_mult}, timeout=90)

st.title(f"{settings.symbol} · {settings.timeframe} · ML trading signals")
c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1.6])
if ticker:
    c1.metric("Last price", f"{ticker['last']:,.2f}", f"{ticker['change_24h_pct']:+.2f}% (24h)")
    c2.metric("24h high", f"{ticker['high_24h']:,.0f}")
    c3.metric("24h low", f"{ticker['low_24h']:,.0f}")
    c4.metric("24h volume (BTC)", f"{ticker['volume_24h']:,.0f}")
else:
    c1.metric("Last price", "n/a")

with c5:
    if signal:
        action = signal["action"]
        color = ACTION_COLORS.get(action, "#8494AC")
        st.markdown(
            f"<div style='border:2px solid {color};border-radius:12px;padding:10px 14px;text-align:center'>"
            f"<div style='font-size:12px;color:#8494AC'>SIGNAL · {signal['prediction']['timestamp'][:16].replace('T', ' ')} UTC</div>"
            f"<div style='font-size:32px;font-weight:800;color:{color}'>{action}</div>"
            f"<div style='font-size:13px'>confidence {signal['probability']:.1%} · threshold {signal['threshold']:.0%}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif not health.get("models_trained"):
        st.warning("Models are not trained yet. Use the *Model* tab to start training.")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

VIEWS = ["🧭 Swing", "📈 Chart", "🎯 Signal & risk", "🧪 Backtest", "🧠 Model"]
# A radio persisted in session_state keeps the selected view across auto-refresh reruns (st.tabs resets).
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")

if view == "🧭 Swing":
    st.subheader("מצב שוק לסווינג: BTC ו-MSTR")
    st.caption("לא איתות קנייה/מכירה. תשובה לשאלות של מי שמחזיק שבועות: איפה אנחנו במחזור, כמה מסוכן השבוע והחודש הקרובים, אילו רמות חשובות, וכמה פוזיציה מתאימה לסיכון שאתה מוכן לספוג.")
    sz1, sz2 = st.columns(2)
    portfolio = sz1.number_input("גודל התיק (USD)", min_value=100.0, value=10_000.0, step=500.0, key="swing_portfolio")
    max_loss = sz2.slider("הפסד חודשי מקסימלי שאתה מוכן לספוג (% מהתיק)", 1.0, 30.0, 10.0, 0.5, key="swing_max_loss")
    REGIME_HE = {"BULL": ("שורי", "#35C98D"), "BEAR": ("דובי", "#E5615E"), "NEUTRAL": ("ניטרלי / מעורב", "#E8A33D")}
    TREND_HE = {"UP": "עולה", "DOWN": "יורדת", "MIXED": "מעורבת"}
    cols = st.columns(2)
    for col, asset in zip(cols, ("BTC", "MSTR")):
        with col:
            with st.spinner(f"טוען {asset}..."):
                snap = api_get(api_base, f"/swing/{asset}", {"history_days": 365}, timeout=120)
            if not snap:
                st.warning(f"אין נתונים ל-{asset} כרגע.")
                continue
            reg = snap["regime"]
            label, colour = REGIME_HE[reg["label"]]
            st.markdown(
                f"<div style='border:2px solid {colour};border-radius:12px;padding:12px 16px'>"
                f"<div style='font-size:13px;color:#8494AC'>{asset} · {snap['symbol']} · נכון ל-{snap['as_of']}</div>"
                f"<div style='font-size:34px;font-weight:800'>{snap['price']:,.2f} <span style='font-size:16px;color:{'#35C98D' if snap['changes_pct']['1d'] >= 0 else '#E5615E'}'>{snap['changes_pct']['1d']:+.2f}% היום</span></div>"
                f"<div style='font-size:22px;font-weight:800;color:{colour};margin-top:4px'>משטר: {label} <span style='font-size:13px;color:#8494AC'>(ציון {reg['score']:+d} מתוך ±{reg['max_score']})</span></div>"
                "</div>", unsafe_allow_html=True)
            for r in reg["reasons"]:
                st.markdown(f"• {r}")
            ch = snap["changes_pct"]
            m1, m2 = st.columns(2)
            m1.metric("שבוע", f"{ch['1w']:+.1f}%")
            m2.metric("חודש", f"{ch['1m']:+.1f}%")
            m3, m4 = st.columns(2)
            m3.metric("3 חודשים", f"{ch['3m']:+.1f}%")
            m4.metric("שנה", f"{ch['1y']:+.1f}%")
            risk = snap["risk"]
            st.markdown("**סיכון (מהתנודתיות של 21 הימים האחרונים)**")
            r1, r2 = st.columns(2)
            r1.metric("תזוזה צפויה לשבוע (±1σ)", f"±{risk['expected_move_1w_pct']:.1f}%")
            r2.metric("תזוזה צפויה לחודש (±1σ)", f"±{risk['expected_move_1m_pct']:.1f}%")
            st.metric("תנודתיות שנתית", f"{risk['vol_21d_annualised_pct']:.0f}%", f"אחוזון {risk['vol_percentile_1y']:.0f} בשנה האחרונה")
            st.caption(f"טווח סביר לחודש: {risk['range_1m'][0]:,.0f} – {risk['range_1m'][1]:,.0f} · במקרה קיצון (2σ): {risk['range_1m_2sigma'][0]:,.0f} – {risk['range_1m_2sigma'][1]:,.0f} · ירידה מהשיא: {risk['drawdown_from_ath_pct']:.1f}% · הנפילה הגדולה בשנה: {risk['max_drawdown_1y_pct']:.1f}%"
                       + (f" · בטא לביטקוין {risk['beta_to_btc_63d']:.2f}" if "beta_to_btc_63d" in risk else ""))
            two_sigma = 2 * risk["expected_move_1m_pct"]
            frac = min(1.0, max_loss / two_sigma) if two_sigma > 0 else 1.0
            st.markdown("**גודל פוזיציה מתאים**")
            st.info(f"כדי שירידה חודשית קיצונית (2σ = {two_sigma:.0f}%) תפסיד לכל היותר {max_loss:.0f}% מהתיק: עד **{frac * 100:.0f}% מהתיק** = **{portfolio * frac:,.0f} USD** ב-{asset}.")
            lv = snap["levels"]
            st.markdown("**רמות מחיר**")
            st.dataframe(pd.DataFrame({
                "רמה": ["שיא כל הזמנים", "שיא 52 שבועות", "שיא 20 יום", "ממוצע 50 יום", "ממוצע 200 יום", "מחיר הכי נסחר (120 יום)", "שפל 20 יום", "שפל 52 שבועות"],
                "מחיר": [lv["all_time_high"], lv["high_52w"], lv["high_20d"], lv["ema50"], lv["ema200"], lv["poc_120d"], lv["low_20d"], lv["low_52w"]],
                "מרחק": [f"{(snap['price'] / v - 1) * 100:+.1f}%" for v in (lv["all_time_high"], lv["high_52w"], lv["high_20d"], lv["ema50"], lv["ema200"], lv["poc_120d"], lv["low_20d"], lv["low_52w"])],
            }).style.format({"מחיר": "{:,.2f}"}), hide_index=True, width="stretch")
            hist = pd.DataFrame(snap["history"])
            hist["date"] = pd.to_datetime(hist["date"])
            rows_n = 2 if asset == "BTC" and "cm_mvrv" in hist else 1
            figs = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if rows_n == 2 else [1.0], vertical_spacing=0.04)
            figs.add_trace(go.Candlestick(x=hist["date"], open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"], name=asset,
                                          increasing_line_color="#35C98D", decreasing_line_color="#E5615E"), row=1, col=1)
            figs.add_trace(go.Scatter(x=hist["date"], y=hist["ema50"], name="EMA 50", line=dict(color="#4EA8DE", width=1.2)), row=1, col=1)
            figs.add_trace(go.Scatter(x=hist["date"], y=hist["ema200"], name="EMA 200", line=dict(color="#C77DFF", width=1.6)), row=1, col=1)
            figs.add_hline(y=lv["high_52w"], line=dict(color="rgba(245,197,66,.5)", dash="dot"), row=1, col=1)
            figs.add_hline(y=lv["low_52w"], line=dict(color="rgba(245,197,66,.5)", dash="dot"), row=1, col=1)
            if rows_n == 2:
                figs.add_trace(go.Scatter(x=hist["date"], y=hist["cm_mvrv"], name="MVRV", line=dict(color="#F5C542")), row=2, col=1)
                figs.add_hline(y=1.0, line=dict(color="#35C98D", dash="dot", width=1), row=2, col=1)
                figs.add_hline(y=3.0, line=dict(color="#E5615E", dash="dot", width=1), row=2, col=1)
            figs.update_layout(height=520 if rows_n == 2 else 400, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figs, width="stretch")
    st.caption("מקורות: Yahoo Finance (מחירים יומיים), Coin Metrics (MVRV, סטייבלקוינים), Alternative.me (פחד/חמדנות). המדדים מתעדכנים פעם ביום. זה כלי מצב וסיכון, לא תחזית כיוון: בבדיקות על 6 שנים, חיזוי כיוון לא ניצח החזקה פשוטה.")

if view == "📈 Chart":
    ind = api_get(api_base, "/indicators", {"limit": n_candles})
    vp = api_get(api_base, "/indicators/volume-profile", {"lookback": 240, "bins": 30})
    if ind and ind["rows"]:
        df = pd.DataFrame(ind["rows"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig = make_subplots(
            rows=3, cols=2, shared_xaxes=True, column_widths=[0.82, 0.18], row_heights=[0.62, 0.2, 0.18],
            vertical_spacing=0.03, horizontal_spacing=0.02,
            specs=[[{}, {"rowspan": 3}], [{}, None], [{}, None]],
        )
        fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC",
                                     increasing_line_color="#35C98D", decreasing_line_color="#E5615E"), row=1, col=1)
        for col, colour, width in (("ema_20", "#F5C542", 1.2), ("ema_50", "#4EA8DE", 1.2), ("ema_200", "#C77DFF", 1.6), ("vwap", "#FFFFFF", 1.0)):
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=col.upper(), line=dict(color=colour, width=width, dash="dot" if col == "vwap" else None)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bb_upper"], name="BB upper", line=dict(color="rgba(132,148,172,0.5)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bb_lower"], name="BB lower", line=dict(color="rgba(132,148,172,0.5)", width=1),
                                 fill="tonexty", fillcolor="rgba(132,148,172,0.08)"), row=1, col=1)
        if signal and signal.get("trade_plan"):
            plan = signal["trade_plan"]
            x0, x1 = df["timestamp"].iloc[max(0, len(df) - 60)], df["timestamp"].iloc[-1] + pd.Timedelta(hours=8)
            fig.add_shape(type="line", x0=x0, x1=x1, y0=plan["entry_price"], y1=plan["entry_price"], line=dict(color="#FFFFFF", width=1, dash="dash"), row=1, col=1)
            fig.add_shape(type="line", x0=x0, x1=x1, y0=plan["stop_loss"], y1=plan["stop_loss"], line=dict(color="#E5615E", width=1.5), row=1, col=1)
            for tp in plan["take_profits"]:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=tp["price"], y1=tp["price"], line=dict(color="#35C98D", width=1.5), row=1, col=1)
                fig.add_annotation(x=x1, y=tp["price"], text=f"TP 1:{tp['reward_risk']:.0f}", showarrow=False, xanchor="left", font=dict(color="#35C98D", size=10), row=1, col=1)
            fig.add_annotation(x=x1, y=plan["stop_loss"], text="SL", showarrow=False, xanchor="left", font=dict(color="#E5615E", size=10), row=1, col=1)
        if signal:
            pred = signal["prediction"]
            last_ts = df["timestamp"].iloc[-1]
            fig.add_trace(go.Scatter(x=[last_ts, last_ts + pd.Timedelta(hours=pred["horizon_bars"])], y=[pred["expected_high"]] * 2, name="Expected high",
                                     line=dict(color="#35C98D", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=[last_ts, last_ts + pd.Timedelta(hours=pred["horizon_bars"])], y=[pred["expected_low"]] * 2, name="Expected low",
                                     line=dict(color="#E5615E", width=1, dash="dot")), row=1, col=1)
        # Volume profile (horizontal)
        if vp:
            levels = pd.DataFrame(vp["levels"])
            fig.add_trace(go.Bar(x=levels["volume"], y=levels["price"], orientation="h", name="Volume profile", marker_color="rgba(78,168,222,0.55)"), row=1, col=2)
            fig.add_hline(y=vp["poc"], line=dict(color="#F5C542", width=2), row=1, col=2)
            fig.add_hrect(y0=vp["value_area_low"], y1=vp["value_area_high"], fillcolor="rgba(245,197,66,0.10)", line_width=0, row=1, col=2)
        # RSI
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rsi_14"], name="RSI 14", line=dict(color="#4EA8DE")), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#E5615E", dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#35C98D", dash="dot", width=1), row=2, col=1)
        # Volume
        colours = ["#35C98D" if c >= o else "#E5615E" for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colours, opacity=0.7), row=3, col=1)
        fig.update_layout(height=780, template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0),
                          margin=dict(l=10, r=10, t=30, b=10))
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Vol", row=3, col=1)
        fig.update_yaxes(matches="y", showticklabels=False, row=1, col=2)
        st.plotly_chart(fig, width="stretch")

        latest = df.iloc[-1]
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("RSI 14", f"{latest['rsi_14']:.1f}")
        k2.metric("ATR 14", f"{latest['atr_14']:,.0f} ({latest['atr_pct'] * 100:.2f}%)")
        k3.metric("EMA 20 / 50", f"{latest['ema_20']:,.0f} / {latest['ema_50']:,.0f}")
        k4.metric("EMA 200", f"{latest['ema_200']:,.0f}")
        k5.metric("VWAP (day)", f"{latest['vwap']:,.0f}")
        k6.metric("POC (240h)", f"{vp['poc']:,.0f}" if vp else "n/a")
    else:
        st.info("No indicator data available yet.")

if view == "🎯 Signal & risk":
    if signal:
        pred = signal["prediction"]
        left, right = st.columns([1, 1.2])
        with left:
            st.subheader("Direction probabilities")
            probs = pd.DataFrame(
                {"class": ["DOWN", "FLAT", "UP"], "ensemble": [pred["prob_down"], pred["prob_flat"], pred["prob_up"]]}
            )
            for name, comp in pred.get("components", {}).items():
                probs[name] = [comp["DOWN"], comp["FLAT"], comp["UP"]]
            bar = go.Figure()
            for col in probs.columns[1:]:
                bar.add_trace(go.Bar(x=probs["class"], y=probs[col], name=col))
            bar.add_hline(y=signal["threshold"], line=dict(color="#E8A33D", dash="dash"))
            bar.update_layout(template="plotly_dark", height=320, barmode="group", yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(bar, width="stretch")
            st.caption(signal["reason"])
            st.write(
                f"Expected range over the next {pred['horizon_bars']} bars: "
                f"**{pred['expected_low']:,.0f}** ({pred['expected_max_down_pct']:+.2f}%) → "
                f"**{pred['expected_high']:,.0f}** ({pred['expected_max_up_pct']:+.2f}%)"
            )
        with right:
            st.subheader("Trade plan")
            plan = signal.get("trade_plan")
            if plan:
                r1, r2, r3 = st.columns(3)
                r1.metric("Entry", f"{plan['entry_price']:,.2f}")
                r2.metric("Stop-loss", f"{plan['stop_loss']:,.2f}", f"-{plan['stop_distance_pct'] * 100:.2f}%")
                r3.metric("Position", f"{plan['position_size']:.4f} BTC", f"{plan['notional']:,.0f} USDT")
                r4, r5, r6 = st.columns(3)
                r4.metric("Risk amount", f"{plan['risk_amount']:,.2f} USDT")
                r5.metric("Leverage needed", f"{plan['leverage_required']:.2f}x", "capped" if plan["capped_by_leverage"] else None)
                r6.metric("ATR used", f"{plan['atr']:,.0f}")
                tps = pd.DataFrame(plan["take_profits"])
                tps["reward_risk"] = tps["reward_risk"].map(lambda r: f"1:{r:.0f}")
                tps["distance_pct"] = (tps["distance_pct"] * 100).map(lambda v: f"{v:.2f}%")
                tps["price"] = tps["price"].map(lambda v: f"{v:,.2f}")
                tps["profit_usdt"] = [f"{plan['risk_amount'] * t['reward_risk']:,.2f}" for t in plan["take_profits"]]
                st.table(tps.rename(columns={"reward_risk": "R:R", "price": "Take-profit", "distance_pct": "Distance", "profit_usdt": "Profit at TP"}))
            else:
                st.info("No trade: the model is not confident enough. Waiting for the next candle.")
            st.subheader("Key indicators")
            ind_row = signal["indicators"]
            st.dataframe(pd.DataFrame({"indicator": list(ind_row), "value": [round(v, 4) for v in ind_row.values()]}), hide_index=True, width="stretch", height=330)
    else:
        st.info("Signal unavailable - train the model first.")

if view == "🧪 Backtest":
    st.subheader("Walk-forward backtest (out-of-sample)")
    latest_bt = api_get(api_base, "/backtest/latest", {"mode": "walk_forward"}, silent=True)
    if latest_bt is None:
        latest_bt = api_get(api_base, "/backtest/latest", {"mode": "holdout"}, silent=True)
    colA, colB = st.columns([1, 3])
    with colA:
        bt_mode = st.selectbox("Mode", ["holdout", "walk_forward"], index=0)
        bt_thr = st.slider("Threshold", 0.34, 0.90, threshold, 0.01, key="bt_thr")
        bt_rr = st.selectbox("Take-profit R:R", [2.0, 3.0], index=0)
        bt_atr = st.slider("ATR multiplier", 0.5, 4.0, atr_mult, 0.1, key="bt_atr")
        bt_short = st.checkbox("Allow shorts", True)
        if st.button("Run backtest", type="primary"):
            payload = {"mode": bt_mode, "threshold": bt_thr, "take_profit_rr": bt_rr, "atr_multiplier": bt_atr,
                       "risk_per_trade_pct": risk_pct, "initial_equity": equity, "allow_short": bt_short}
            started = api_post(api_base, "/backtest/run", payload, timeout=60)
            if started:
                key = started["key"]
                result = started.get("result")
                with st.spinner("Running backtest (re-training on the in-sample window)..."):
                    waited = 0
                    while result is None and waited < 600:
                        time.sleep(3)
                        waited += 3
                        poll = api_get(api_base, "/backtest/result", {"key": key})
                        if poll is None:
                            break
                        if poll.get("status") == "ready":
                            result = poll["result"]
                if result:
                    st.session_state["bt_result"] = result
    result = st.session_state.get("bt_result") or latest_bt
    with colB:
        if result:
            m = result["metrics"]
            st.caption(f"{result['mode']} · {result['start'][:10]} → {result['end'][:10]} · {result['bars']} bars · {len(result.get('folds', []))} fold(s)")
            # Two rows of three: six metrics side by side get ellipsised inside the 3/4-width panel.
            g1, g2, g3 = st.columns(3)
            g4, g5, g6 = st.columns(3)
            g1.metric("Trades", m["total_trades"])
            g2.metric("Win rate", f"{m['win_rate']:.1%}")
            g3.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
            g4.metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
            g5.metric("Return", f"{m['total_return_pct']:+.2f}%", f"B&H {m['buy_and_hold_return_pct']:+.1f}%")
            g6.metric("Profit factor", f"{m['profit_factor']:.2f}")
            curve = pd.DataFrame(result["equity_curve"])
            curve["timestamp"] = pd.to_datetime(curve["timestamp"])
            eq = go.Figure(go.Scatter(x=curve["timestamp"], y=curve["equity"], name="Equity", line=dict(color="#35C98D")))
            eq.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Equity (USDT)")
            st.plotly_chart(eq, width="stretch")
            trades = pd.DataFrame(result["trades"])
            if not trades.empty:
                st.dataframe(trades.tail(200).iloc[::-1], hide_index=True, width="stretch", height=300)
        else:
            st.info("No backtest yet - run one from the left panel or via `python -m src.backtest.engine`.")

if view == "🧠 Model":
    info = api_get(api_base, "/model/info")
    if info:
        meta = info.get("metadata") or {}
        colM1, colM2 = st.columns([1, 1])
        with colM1:
            st.subheader("Training status")
            st.write(f"Trained: **{info['trained']}** · running: **{info['training_running']}**")
            if info.get("training_error"):
                st.error(info["training_error"])
            if meta:
                dm = meta["direction_metrics"]
                st.write(f"Trained at: `{meta['trained_at'][:19]}` · horizon {meta['horizon']} bars · {meta['n_train']} train / {meta['n_test']} test rows")
                st.write(f"Out-of-sample test period: `{meta['test_start'][:10]}` → `{meta['test_end'][:10]}`")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Accuracy", f"{dm['accuracy']:.3f}")
                t2.metric("Balanced acc.", f"{dm['balanced_accuracy']:.3f}")
                t3.metric("Log-loss", f"{dm['log_loss']:.4f}")
                t4.metric("Signal accuracy", f"{dm['signal_directional_accuracy']:.1%}", f"{dm['n_signals']} signals")
                st.json({"range_metrics": meta["range_metrics"], "tuned": meta["tuned"], "training_seconds": meta["training_seconds"]}, expanded=False)
            tune = st.checkbox("Hyper-parameter search (slower)")
            if st.button("Re-train model"):
                started = api_post(api_base, "/model/train", {"tune": tune, "n_iter": 12})
                if started:
                    st.success("Training started in the background; refresh in a minute.")
        with colM2:
            if meta:
                st.subheader("Top features")
                feats = pd.Series(meta["top_features"]).sort_values()
                fi = go.Figure(go.Bar(x=feats.values, y=feats.index, orientation="h", marker_color="#4EA8DE"))
                fi.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fi, width="stretch")

cache_age = health.get("market_cache_age_s")  # None until the in-process market cache is first warmed
st.caption(
    f"API {api_base} · cache age {'n/a' if cache_age is None else f'{cache_age}s'} · "
    f"exchange {health.get('exchange')} · v{health.get('version')}"
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
