"""
Streamlit entry point for the Gold (XAU/USD) Institutional Prediction Engine.

Features THREE composite horizons (Short / Medium / Long) so at least one
is usually active instead of stuck in neutral.  All composites, forecasts,
and trade logic automatically recompute when you switch the data interval.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Robust imports
# ---------------------------------------------------------------------------
def _try_import():
    try:
        from config import Config, MARKET_LABELS
        from data import DataSourceFactory
        from scoring import (
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from backtest import BacktestConfig, run_backtest
        from trades import run_trade_engine, TradeConfig as EngineTradeConfig
        from trade_ui import render_trade_dashboard
        return {
            "Config": Config, "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score, "forecasts": forecasts,
            "forecast_classify": forecast_classify,
            "market_regime": market_regime, "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig, "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "flat",
        }
    except ImportError:
        from engine.config import Config, MARKET_LABELS
        from engine.data import DataSourceFactory
        from engine.scoring import (
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from engine.backtest import BacktestConfig, run_backtest
        from engine.trades import run_trade_engine, TradeConfig as EngineTradeConfig
        from engine.trade_ui import render_trade_dashboard
        return {
            "Config": Config, "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score, "forecasts": forecasts,
            "forecast_classify": forecast_classify,
            "market_regime": market_regime, "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig, "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "engine",
        }

_IMPORTS = _try_import()
Config                  = _IMPORTS["Config"]
MARKET_LABELS           = _IMPORTS["MARKET_LABELS"]
DataSourceFactory       = _IMPORTS["DataSourceFactory"]
composite_score         = _IMPORTS["composite_score"]
forecasts               = _IMPORTS["forecasts"]
forecast_classify       = _IMPORTS["forecast_classify"]
market_regime           = _IMPORTS["market_regime"]
flow_meter              = _IMPORTS["flow_meter"]
BacktestConfig          = _IMPORTS["BacktestConfig"]
run_backtest            = _IMPORTS["run_backtest"]
run_trade_engine        = _IMPORTS["run_trade_engine"]
EngineTradeConfig       = _IMPORTS["EngineTradeConfig"]
render_trade_dashboard  = _IMPORTS["render_trade_dashboard"]


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Institutional Engine",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .stMetric > div { padding: 0.5rem 0.75rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
SIGNAL_COLOURS = {
    "Strong Bullish":  "#006400", "Bullish": "#2E8B57", "Neutral": "#DAA520",
    "Bearish": "#B22222", "Strong Bearish": "#8B0000",
}

REGIME_COLOURS = {
    "Gold Bull": "#DAA520", "Gold Bear": "#B22222", "Transition": "#888888",
}

COMPOSITE_COLOURS = {
    "short":  "#FF8C00",
    "medium": "#1f77b4",
    "long":   "#9467bd",
}


@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def _fetch(source_name: str, api_key: str, interval: str, lookback_days: int):
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    src = DataSourceFactory.create(cfg)
    return src.fetch_all(lookback_days=lookback_days, interval=interval)


def _score_alignment_check(data):
    missing = [m for m in MARKET_LABELS if m not in data or len(data[m]) == 0]
    return missing


def _gauge(value, title, min_val=0, max_val=100, suffix="", bar_color="#DAA520"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 28}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar":  {"color": bar_color},
            "steps": [
                {"range": [min_val, 40], "color": "#f8d7da"},
                {"range": [40, 60],      "color": "#fff3cd"},
                {"range": [60, max_val], "color": "#d4edda"},
            ],
        },
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=0))
    return fig


def _signal_card(label, score, conf, color, title):
    return f"""
    <div style="background:{color}15;border-left:5px solid {color};
                padding:12px 14px;border-radius:8px;margin-bottom:8px">
      <div style="font-size:0.78rem;color:#666">{title}</div>
      <div style="font-size:1.25rem;font-weight:700;color:{color}">{label}</div>
      <div style="font-size:0.85rem;color:#444">
        Score: <b>{score:+.1f}</b> · Conf: <b>{conf:.1f}%</b>
      </div>
    </div>
    """


def _humanize(bars: int, interval: str) -> str:
    if interval == "1d":
        return f"~{bars} days"
    if interval == "1h":
        return f"~{bars/24:.1f} days" if bars >= 24 else f"~{bars} hours"
    hours = bars * {"30m": 0.5, "15m": 0.25, "5m": 5/60, "1m": 1/60}.get(interval, 1.0)
    if hours < 1:
        return f"~{int(hours*60)} min"
    if hours < 24:
        return f"~{hours:.1f} hours"
    return f"~{hours/24:.1f} days"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
config = Config.from_sidebar()

# Backward-compat shim: if config only has the old single `periods` dict,
# auto-create the three horizon sets so the app doesn't crash.
if not hasattr(config, "periods_short"):
    # Old config loaded — synthesise the three sets from the single periods dict
    base = getattr(config, "periods", getattr(config, "periods_medium", {}))
    if not base:
        base = {"ema_fast": 20, "ema_slow": 50, "rsi_len": 14, "roc_len": 20}
    config.periods_short  = {k: max(2, int(v * 0.4)) for k, v in base.items()}
    config.periods_medium = dict(base)
    config.periods_long   = {k: int(v * 2.5) for k, v in base.items()}
    st.toast("⚠️ Using legacy config — three composites auto-generated from default periods. Update config.py for full control.", icon="⚠️")


# -----------------------------------------------------------------------------
# Pull data
# -----------------------------------------------------------------------------
try:
    from engine.data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG
except ImportError:
    from data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG

INTERVAL_LABELS = {
    "1d": "Daily (1d) — swing", "1h": "Hourly (1h) — intraday",
    "30m": "30-min — intraday", "15m": "15-min — intraday / scalping",
    "5m": "5-min — scalping", "1m": "1-min — scalping (limited history)",
}

with st.spinner(f"Loading {config.interval} market data…"):
    try:
        lb = interval_lookback_days(config.interval)
        data = _fetch(config.data_source, config.twelvedata_api_key, config.interval, lb)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

missing = _score_alignment_check(data)
if missing:
    st.warning(f"Missing data for: {', '.join(missing)}. Try switching data source.")
    if "gold" not in data or len(data["gold"]) == 0:
        st.error("Gold is required. Cannot continue.")
        st.stop()


# -----------------------------------------------------------------------------
# Compute THREE composites + forecasts + regime + flow
# -----------------------------------------------------------------------------
with st.spinner("Computing multi-timeframe composites…"):
    score_short  = composite_score(data, config, periods=config.periods_short)
    score_medium = composite_score(data, config, periods=config.periods_medium)
    score_long   = composite_score(data, config, periods=config.periods_long)

    fcasts = forecasts(data, config)
    f_labels = {}
    f_conf = {}
    for name, s in fcasts.items():
        lbl, conf = forecast_classify(s)
        f_labels[name] = lbl
        f_conf[name] = conf

    regime_s = market_regime(score_short.per_market)
    regime_m = market_regime(score_medium.per_market)
    regime_l = market_regime(score_long.per_market)

    flow_s = flow_meter(
        score_short.composite,
        data["dxy"]["Close"] if "dxy" in data else pd.Series(100.0, index=score_short.composite.index),
        data["vix"]["Close"] if "vix" in data else pd.Series(15.0, index=score_short.composite.index),
        data["ief"]["Close"] if "ief" in data else pd.Series(100.0, index=score_short.composite.index),
    )
    flow_m = flow_meter(
        score_medium.composite,
        data["dxy"]["Close"] if "dxy" in data else pd.Series(100.0, index=score_medium.composite.index),
        data["vix"]["Close"] if "vix" in data else pd.Series(15.0, index=score_medium.composite.index),
        data["ief"]["Close"] if "ief" in data else pd.Series(100.0, index=score_medium.composite.index),
    )
    flow_l = flow_meter(
        score_long.composite,
        data["dxy"]["Close"] if "dxy" in data else pd.Series(100.0, index=score_long.composite.index),
        data["vix"]["Close"] if "vix" in data else pd.Series(15.0, index=score_long.composite.index),
        data["ief"]["Close"] if "ief" in data else pd.Series(100.0, index=score_long.composite.index),
    )


SCORES = {
    "short":  {"score": score_short,  "regime": regime_s, "flow": flow_s,
               "periods": config.periods_short,  "label": "Short (fast)"},
    "medium": {"score": score_medium, "regime": regime_m, "flow": flow_m,
               "periods": config.periods_medium, "label": "Medium (balanced)"},
    "long":   {"score": score_long,   "regime": regime_l, "flow": flow_l,
               "periods": config.periods_long,   "label": "Long (structural)"},
}

latest = score_medium.composite.index[-1]


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🥇 Gold (XAU/USD) Institutional Prediction Engine")
st.caption(
    f"Interval: **{INTERVAL_LABELS.get(config.interval, config.interval)}** · "
    f"Data: **{config.data_source}** · "
    f"Last bar: **{data['gold'].index[-1].strftime('%Y-%m-%d %H:%M')}** · "
    f"Markets: **{len(data)}/7**"
)

_s = config.periods_short; _m = config.periods_medium; _l = config.periods_long
with st.expander("ℹ️  What do the composite & forecast horizons mean?", expanded=False):
    st.markdown(f"""
All horizons are computed on **{config.interval} bars**. Their wall-clock meaning changes with the interval.

| Horizon | EMA fast/slow | RSI | Approx. duration |
|---|---|---|---|
| **Composite Short**  | EMA{_s['ema_fast']}/{_s['ema_slow']} | RSI{_s['rsi_len']} | {_humanize(_s['ema_slow'], config.interval)} |
| **Composite Medium** | EMA{_m['ema_fast']}/{_m['ema_slow']} | RSI{_m['rsi_len']} | {_humanize(_m['ema_slow'], config.interval)} |
| **Composite Long**   | EMA{_l['ema_fast']}/{_l['ema_slow']} | RSI{_l['rsi_len']} | {_humanize(_l['ema_slow'], config.interval)} |

A shorter composite is more responsive but noisier. A longer composite is smoother but lags.
Switching from **1d** to **15m** makes every horizon faster in wall-clock terms.
    """)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_live, tab_backtest, tab_trades, tab_about = st.tabs(
    ["📊  Live", "🧪  Backtest", "🎯  Trades", "ℹ️  About"]
)


# =============================================================================
# LIVE TAB
# =============================================================================
with tab_live:

    # ----- Row 1: Three composite signal cards -----------------------------
    st.subheader("Multi-timeframe composite signals")
    c1, c2, c3 = st.columns(3)
    for col, key, emoji in [(c1, "short", "⚡"), (c2, "medium", "⚖️"), (c3, "long", "🏛️")]:
        sc = SCORES[key]["score"]
        lbl = str(sc.label.loc[latest])
        scr = float(sc.composite.loc[latest])
        cnf = float(sc.confidence.loc[latest])
        clr = SIGNAL_COLOURS[lbl]
        with col:
            st.markdown(
                _signal_card(lbl, scr, cnf, clr, f"{emoji} {SCORES[key]['label'].upper()} COMPOSITE"),
                unsafe_allow_html=True,
            )

    labels_now = {k: str(SCORES[k]["score"].label.loc[latest]) for k in SCORES}
    bulls = sum(1 for v in labels_now.values() if "Bull" in v)
    bears = sum(1 for v in labels_now.values() if "Bear" in v)
    consensus = "Bullish" if bulls >= 2 else "Bearish" if bears >= 2 else "Mixed"
    consensus_color = "#2E8B57" if bulls >= 2 else "#B22222" if bears >= 2 else "#DAA520"
    st.markdown(
        f"<div style='text-align:center;font-size:0.9rem;color:#666'>"
        f"Consensus: <b style='color:{consensus_color}'>{consensus}</b> "
        f"({bulls} bull · {3-bulls-bears} neutral · {bears} bear)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ----- Row 2: Trading composite selector --------------------------------
    st.subheader("Trading setup")
    tc_left, tc_right = st.columns([1, 3])
    with tc_left:
        trade_horizon = st.radio(
            "Trade on which composite?",
            options=["short", "medium", "long"],
            index=1,
            format_func=lambda k: {
                "short":  "⚡ Short — quick swings",
                "medium": "⚖️ Medium — balanced",
                "long":   "🏛️ Long — structural",
            }[k],
            key="trade_horizon_select",
            help="The selected composite drives the live entry setup, backtest, and trade engine.",
        )
    with tc_right:
        sel = SCORES[trade_horizon]
        sel_score = sel["score"]
        sel_lbl = str(sel_score.label.loc[latest])
        sel_scr = float(sel_score.composite.loc[latest])
        sel_cnf = float(sel_score.confidence.loc[latest])
        sel_clr = SIGNAL_COLOURS[sel_lbl]
        sel_flow = float(sel["flow"].loc[latest])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal", sel_lbl)
        m2.metric("Score", f"{sel_scr:+.1f}")
        m3.metric("Confidence", f"{sel_cnf:.1f}%")
        m4.metric("Flow", f"{sel_flow:.0f}/100", delta=f"{sel_flow-50:+.0f} vs neutral")

    st.session_state["active_composite_key"] = trade_horizon
    st.session_state["active_score_result"] = sel_score

    st.markdown("")

    # ----- Row 3: Market breakdown ------------------------------------------
    st.subheader(f"Market breakdown — {SCORES[trade_horizon]['label']}")
    contribs = sel_score.contributions
    weights = config.normalized_weights()
    rows = []
    for mkt, label in MARKET_LABELS.items():
        if mkt not in sel_score.per_market:
            continue
        s = float(sel_score.per_market[mkt].loc[latest])
        w = weights[mkt] * 100
        c = float(contribs[mkt].loc[latest])
        direction = "▼" if c < 0 else "▲"
        rows.append({
            "Market": label, "Asset Score": f"{s:+.1f}",
            "Weight": f"{w:.1f}%", "Contribution": f"{c:+.1f}", "Direction": direction,
        })
    df_mkts = pd.DataFrame(rows).set_index("Market")

    def _color_contrib(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 20:   return "background-color: #2E8B5730"
        if v < -20:  return "background-color: #B2222230"
        return "background-color: #DAA52030"

    st.dataframe(
        df_mkts.style.map(_color_contrib, subset=["Contribution"]),
        use_container_width=True, height=290,
    )

    # ----- Row 4: Live entry / SL / TP -------------------------------------
    try:
        from indicators import atr as _atr, ema as _ema
    except ImportError:
        try:
            from engine.indicators import atr as _atr, ema as _ema
        except ImportError:
            _atr = _ema = None

    if _atr is not None and "gold" in data and len(data["gold"]) > 30:
        gold_now = data["gold"]
        atr_series = _atr(gold_now["High"], gold_now["Low"], gold_now["Close"], 14)
        price_now = float(gold_now["Close"].iloc[-1])
        atr_now = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

        f_label_now = str(sel_score.label.loc[latest])
        f_conf_now = sel_cnf
        f_score_now = sel_scr
        f_color_now = sel_clr
        p = sel["periods"]
        f_ef, f_es = p["ema_fast"], p["ema_slow"]

        if f_label_now == "Bullish":
            side = 1
        elif f_label_now == "Bearish":
            side = -1
        else:
            side = 0

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(
                f"""<div style="background:{f_color_now}1a;border-left:5px solid {f_color_now};
                            padding:8px 12px;border-radius:6px">
                      <div style="font-size:0.75rem;color:#666">COMPOSITE SIGNAL</div>
                      <div style="font-size:1.05rem;font-weight:700;color:{f_color_now}">{f_label_now}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with h2:
            st.metric("Score", f"{f_score_now:+.1f}")
        with h3:
            st.metric("Confidence", f"{f_conf_now:.1f}%")
        with h4:
            st.metric("EMA bias", f"{f_ef}/{f_es}")

        st.subheader(f"🎯 Live entry setup — {trade_horizon} composite")

        if side == 0 or atr_now <= 0:
            st.info(
                f"ℹ️ No entry on the **{trade_horizon}** composite right now "
                f"(label = **{f_label_now}**). Wait for Bullish / Bearish."
            )
        else:
            _horizon_params = {
                "short":  {"k_sl": 1.5, "k_tp": 2.5, "pb": 0.3},
                "medium": {"k_sl": 1.5, "k_tp": 3.0, "pb": 0.5},
                "long":   {"k_sl": 2.0, "k_tp": 5.0, "pb": 0.8},
            }
            params = _horizon_params[trade_horizon]
            k_sl, k_tp, pb_frac = params["k_sl"], params["k_tp"], params["pb"]

            e_fast = float(_ema(gold_now["Close"], f_ef).iloc[-1])
            e_slow = float(_ema(gold_now["Close"], f_es).iloc[-1])
            entry_limit = price_now - side * pb_frac * atr_now
            structural_pullback = e_slow
            if side == 1 and entry_limit < structural_pullback:
                entry_limit = max(entry_limit, structural_pullback)
            if side == -1 and entry_limit > structural_pullback:
                entry_limit = min(entry_limit, structural_pullback)

            stop_dist = k_sl * atr_now
            target_dist = k_tp * atr_now
            stop_price = entry_limit - side * stop_dist
            target_price = entry_limit + side * target_dist
            rr = target_dist / stop_dist

            equity = 10_000.0
            risk_dollars = equity * 0.01
            size = risk_dollars / stop_dist
            t_target = _humanize(int(f_es), config.interval)

            side_label = "LONG 🟢" if side == 1 else "SHORT 🔴"

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.metric("Action", side_label, delta=f"{trade_horizon} composite")
            ec2.metric("Entry (limit)", f"${entry_limit:,.2f}",
                       delta=f"${abs(price_now - entry_limit):.1f} from close")
            ec3.metric("Stop", f"${stop_price:,.2f}",
                       delta=f"${stop_dist:.1f} risk", delta_color="inverse")
            ec4.metric("Target", f"${target_price:,.2f}",
                       delta=f"+${target_dist:.1f} · ~{t_target}")
            ec5.metric("Size (1% risk)", f"{size:.3f}",
                       help=f"Contracts at 1% of $10k. Risk = ${risk_dollars:.0f} per trade.")

            st.caption(
                f"**{trade_horizon.upper()}** composite: **{f_label_now}** "
                f"(score {f_score_now:+.1f}, conf {f_conf_now:.1f}%) · "
                f"EMA bias: fast=${e_fast:,.0f} vs slow=${e_slow:,.0f} · "
                f"R:R **{rr:.2f}**"
            )

            lookback = min(60, len(gold_now))
            chart_df = gold_now.tail(lookback)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
                low=chart_df["Low"], close=chart_df["Close"],
                name="Gold", increasing_line_color="#2E8B57",
                decreasing_line_color="#B22222",
            ))
            ema_slow_full = _ema(gold_now["Close"], f_es).tail(lookback)
            fig.add_trace(go.Scatter(
                x=ema_slow_full.index, y=ema_slow_full.values,
                name=f"EMA {f_es}", line=dict(color="#FFA500", width=1, dash="dot"),
            ))
            for level, color, name in [
                (entry_limit,  "#DAA520", f"Entry ${entry_limit:,.0f}"),
                (stop_price,   "#B22222", f"Stop ${stop_price:,.0f}"),
                (target_price, "#006400", f"Target ${target_price:,.0f}"),
            ]:
                fig.add_hline(y=level, line_color=color, line_width=2,
                              line_dash="dash", annotation_text=name,
                              annotation_position="right",
                              annotation_font_color=color)
            fig.update_layout(
                height=400, template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Gold (USD)", xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Live entry setup unavailable (ATR not warmed up or imports failed).")

    st.markdown("")

    # ----- Row 5: Price-based forecasts -------------------------------------
    st.subheader("Price-based forecast horizons (Gold only)")
    f1, f2, f3 = st.columns(3)
    for col, name, lbl in [(f1, "short", "Short (technical)"),
                           (f2, "medium", "Medium (technical)"),
                           (f3, "long", "Long (technical)")]:
        with col:
            v = f_labels[name].loc[latest]
            c = f_conf[name].loc[latest]
            color = SIGNAL_COLOURS[v]
            st.markdown(
                f"""
                <div style="background:{color}1a;border-left:5px solid {color};
                            padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">{lbl}</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{color}">{v}</div>
                  <div style="font-size:0.78rem;color:#444">Confidence: {c:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- Row 6: Regime & flow ---------------------------------------------
    st.subheader("Regime & flow")
    r1, r2 = st.columns([1, 1])
    with r1:
        reg_now = sel["regime"].loc[latest]
        st.markdown(
            f"""<div style="background:{REGIME_COLOURS[reg_now]}1a;border-left:5px solid {REGIME_COLOURS[reg_now]};
                        padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">Market Regime</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{REGIME_COLOURS[reg_now]}">{reg_now}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Gold Bull: DXY<−20 ∧ IEF>20 · Gold Bear: DXY>20 ∧ IEF<−20")
    with r2:
        st.plotly_chart(
            _gauge(sel_flow, "Institutional Flow Meter", bar_color=COMPOSITE_COLOURS[trade_horizon]),
            use_container_width=True,
        )

    # ----- Row 7: All three composites overlaid -----------------------------
    st.subheader("Composite history — all timeframes")
    window = st.slider("Lookback (bars)", 30, 500, 180, key="live_lb")

    fig = go.Figure()
    for key, color, name in [
        ("short",  COMPOSITE_COLOURS["short"],  "Short (fast)"),
        ("medium", COMPOSITE_COLOURS["medium"], "Medium (balanced)"),
        ("long",   COMPOSITE_COLOURS["long"],   "Long (structural)"),
    ]:
        hist = SCORES[key]["score"].composite.tail(window)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values, name=name,
            line=dict(color=color, width=2),
        ))

    for level, color, name in [
        (70, "#006400", "Strong Bull"), (40, "#2E8B57", "Bull"),
        (-40, "#B22222", "Bear"), (-70, "#8B0000", "Strong Bear"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.4,
                      annotation_text=name, annotation_position="right")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Composite Score", xaxis_title=None,
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "ℹ️ When all three composites agree, conviction is highest. "
        "When they diverge, the market is in transition."
    )

    # ----- Row 8: Mini price charts -----------------------------------------
    st.subheader("Underlying price action")
    chart_cols = st.columns(2)
    for i, (mkt, label) in enumerate(MARKET_LABELS.items()):
        if mkt not in data:
            continue
        close = data[mkt]["Close"].tail(120)
        with chart_cols[i % 2]:
            fig = go.Figure(go.Scatter(x=close.index, y=close.values, name=label,
                                       line=dict(color="#DAA520", width=1.5)))
            fig.update_layout(height=160, margin=dict(l=10, r=10, t=20, b=0),
                              title=label, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_backtest:
    st.subheader("🧪 Historical backtest")

    bt_comp_key = st.session_state.get("active_composite_key", "medium")
    bt_score = SCORES[bt_comp_key]["score"]

    st.caption(
        f"Strategy: long Gold when the **{bt_comp_key}** composite says Bullish / Strong Bullish, "
        f"cash otherwise. Returns on **{config.interval} bars**. No look-ahead."
    )

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        start = st.date_input("Start", value=datetime(2018, 1, 1),
                              min_value=datetime(2000, 1, 1), max_value=datetime.now(), key="bt_start")
    with bc2:
        end = st.date_input("End", value=datetime.now(),
                            min_value=datetime(2000, 1, 1), max_value=datetime.now() + timedelta(days=1), key="bt_end")
    with bc3:
        _max_hold = {"1d": 250, "1h": 500, "30m": 500, "15m": 500, "5m": 1000, "1m": 1000}.get(config.interval, 500)
        _default_hold = {"1d": 5, "1h": 12, "30m": 24, "15m": 32, "5m": 96, "1m": 240}.get(config.interval, 5)
        hold = st.number_input(f"Holding ({_blabel}s)", 1, _max_hold, _default_hold, key="bt_hold")
    with bc4:
        bt_comp_key = st.selectbox(
            "Backtest composite",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(bt_comp_key),
            format_func=lambda k: {"short": "⚡ Short", "medium": "⚖️ Medium", "long": "🏛️ Long"}[k],
            key="bt_comp",
        )
        bt_score = SCORES[bt_comp_key]["score"]

    bt_cfg = BacktestConfig(start=start.isoformat(), end=end.isoformat(), holding_period=int(hold))

    # Temporarily override config.periods so backtest uses the selected composite
    original_periods = config.periods_medium
    if bt_comp_key == "short":
        config.periods_medium = config.periods_short
    elif bt_comp_key == "long":
        config.periods_medium = config.periods_long

    with st.spinner("Running backtest…"):
        try:
            result = run_backtest(data, config, bt_cfg)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()
        finally:
            config.periods_medium = original_periods

    st.markdown("##### Performance vs buy & hold")
    m = result.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Strategy CAGR", f"{m['Strategy CAGR %']:.2f}%",
               delta=f"{m['Strategy CAGR %'] - m['Benchmark CAGR %']:.2f}% vs BH")
    mc2.metric("Strategy Sharpe", f"{m['Strategy Sharpe']:.2f}",
               delta=f"{m['Strategy Sharpe'] - m['Benchmark Sharpe']:.2f}")
    mc3.metric("Strategy Max DD", f"{m['Strategy Max DD %']:.2f}%",
               delta=f"{m['Strategy Max DD %'] - m['Benchmark Max DD %']:.2f}% vs BH", delta_color="inverse")
    mc4.metric("Strategy Total Ret", f"{m['Strategy Total Ret %']:.2f}%",
               delta=f"{m['Strategy Total Ret %'] - m['Benchmark Total Ret %']:.2f}% vs BH")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Strategy", line=dict(color="#DAA520", width=2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                             name="Buy & Hold Gold", line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=0),
                      yaxis_title="Equity ($)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Per-signal statistics (forward return, %)")
    st.dataframe(result.summary, use_container_width=True)

    st.markdown("##### Last 30 signals")
    st.dataframe(
        result.signals.tail(30).iloc[::-1]
            .assign(composite=lambda d: d["composite"].round(1),
                    fwd_return=lambda d: (d["fwd_return"] * 100).round(2))
            .rename(columns={"fwd_return": "fwd_ret_%"}),
        use_container_width=True, height=380,
    )

    st.markdown("##### Score distribution & forward return")
    fig = px.scatter(
        result.signals.dropna(subset=["fwd_return"]).reset_index(),
        x="composite", y="fwd_return", color="signal",
        color_discrete_map=SIGNAL_COLOURS,
        labels={"composite": "Composite Score", "fwd_return": "Forward Return"},
        opacity=0.6,
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TRADES TAB
# =============================================================================
with tab_trades:
    st.subheader("🎯 Trade Engine — R-based execution")

    tr_comp_key = st.session_state.get("active_composite_key", "medium")
    tr_score = SCORES[tr_comp_key]["score"]

    st.caption(
        f"Trading on the **{tr_comp_key}** composite signals. "
        "Pullback entries · ATR stops · breakeven at +1R · risk-based sizing."
    )

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        risk_pct = st.number_input("Risk per trade (% equity)", 0.1, 5.0, 1.0, step=0.1, key="trade_risk") / 100.0
    with tc2:
        k_sl = st.number_input("Stop (× ATR)", 0.5, 5.0, 1.5, step=0.1, key="trade_k_sl")
    with tc3:
        k_tp = st.number_input("Target (× ATR)", 1.0, 10.0, 3.0, step=0.1, key="trade_k_tp")
    with tc4:
        min_rr = st.number_input("Min R:R filter", 1.0, 5.0, 1.5, step=0.1, key="trade_min_rr")

    tc5, tc6, tc7, tc8 = st.columns(4)
    with tc5:
        use_be = st.checkbox("Breakeven at +1R", value=True, key="trade_be")
    with tc6:
        swing_lb = st.number_input("Swing lookback (bars)", 0, 50, 10, step=1, key="trade_swing")
    with tc7:
        pb_frac = st.number_input("Pullback depth (× ATR)", 0.0, 2.0, 0.5, step=0.1, key="trade_pb")
    with tc8:
        entry_win = st.number_input("Entry window (bars)", 1, 20, 3, step=1, key="trade_ew")

    trade_cfg = EngineTradeConfig(
        risk_per_trade=float(risk_pct), k_sl=float(k_sl), k_tp=float(k_tp),
        min_rr=float(min_rr), use_breakeven_be=bool(use_be),
        swing_lookback=int(swing_lb), pullback_atr_frac=float(pb_frac),
        entry_window=int(entry_win),
    )

    with st.spinner("Running trade engine…"):
        try:
            trade_log = run_trade_engine(
                de40=data["gold"],
                composite=tr_score.composite,
                labels=tr_score.label,
                confidence=tr_score.confidence,
                cfg=trade_cfg,
                initial_equity=10_000.0,
            )
            render_trade_dashboard(trade_log, data["gold"], config)
        except Exception as e:
            st.error(f"Trade engine failed: {e}")
            st.exception(e)


# =============================================================================
# ABOUT TAB
# =============================================================================
with tab_about:
    st.markdown(
        """
        ### What this is
        A multi-market, weighted macro engine for forecasting the next
        directional bias of **Gold (XAU/USD)**.

        ### Key feature: three composite horizons
        Unlike a single composite that often sits neutral, this engine computes
        **three independent composites** on every bar:

        | Horizon | EMA fast/slow | RSI | Best for |
        |---|---|---|---|
        | **Short** | 8 / 21 | 7 | Scalping, quick reversals |
        | **Medium** | 20 / 50 | 14 | Intraday swing |
        | **Long** | 50 / 200 | 21 | Position, macro trends |

        All three are shown side-by-side. When 2+ agree, conviction is high.
        When they diverge, the market is in transition.

        ### Gold macro drivers
        | Market | Weight | Role | Sign |
        |---|---|---|---|
        | DXY | 25% | Dollar strength | Inverted |
        | IEF | 20% | Treasury bond prices (yield proxy) | Positive |
        | Silver | 15% | Precious metals beta | Positive |
        | S&P 500 | 15% | Risk-on substitution | Inverted |
        | EUR/USD | 10% | Dollar proxy | Positive |
        | VIX | 10% | Fear gauge | Positive |
        | Gold | 5% | Self-momentum | Positive |

        ### Deploy
        1. Push to GitHub.
        2. Go to **share.streamlit.io** → connect repo → entry point `app.py`.
        3. Open on your phone — mobile responsive.

        ### Limits
        - yfinance is delayed ~15 min and can be rate-limited.
        - Twelve Data free tier = 800 req/day, 8/min.
        - Backtest assumes zero commissions / slippage.
        """
    )
