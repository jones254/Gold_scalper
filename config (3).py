"""
Configuration container for the Gold (XAU/USD) Engine.

Now supports THREE composite horizons (Short / Medium / Long) so the
signal is never stuck in "neutral" — at least one timeframe is usually
active.  The periods are expressed in bars, so their wall-clock meaning
changes automatically when you switch the data interval in the sidebar.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict


# ----------------------------------------------------------------------------- 
# Default constants - tuned for gold macro drivers
# -----------------------------------------------------------------------------
DEFAULT_DATA_SOURCE = "yfinance"
DEFAULT_TWELVE_DATA_KEY = ""

DEFAULT_WEIGHTS: Dict[str, float] = {
    "dxy":    25.0,
    "ief":    20.0,
    "silver": 15.0,
    "sp500":  15.0,
    "eurusd": 10.0,
    "vix":    10.0,
    "gold":    5.0,
}

# Short  = fast, responsive (scalping / quick swings)
DEFAULT_PERIODS_SHORT: Dict[str, int] = {
    "ema_fast": 8,
    "ema_slow": 21,
    "rsi_len":  7,
    "roc_len":  10,
}

# Medium = balanced (intraday / swing)
DEFAULT_PERIODS_MEDIUM: Dict[str, int] = {
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_len":  14,
    "roc_len":  20,
}

# Long   = slow, structural (position / macro)
DEFAULT_PERIODS_LONG: Dict[str, int] = {
    "ema_fast": 50,
    "ema_slow": 200,
    "rsi_len":  21,
    "roc_len":  50,
}

DEFAULT_FORECASTS = {
    "short":  {"ema_fast": 10, "ema_slow": 20, "rsi": 7},
    "medium": {"ema_fast": 20, "ema_slow": 50, "rsi": 14},
    "long":   {"ema_fast": 50, "ema_slow": 200, "rsi": 21},
}

MARKET_LABELS = {
    "dxy":    "DXY",
    "ief":    "US Treasuries (IEF)",
    "silver": "Silver",
    "sp500":  "S&P 500",
    "eurusd": "EUR/USD",
    "vix":    "VIX",
    "gold":   "Gold",
}

NEGATIVE_CORRELATIONS = {"dxy", "sp500"}


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class Config:
    data_source: str = DEFAULT_DATA_SOURCE
    twelvedata_api_key: str = DEFAULT_TWELVE_DATA_KEY
    interval: str = "1d"
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # Three composite period sets
    periods_short:  Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS_SHORT))
    periods_medium: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS_MEDIUM))
    periods_long:   Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PERIODS_LONG))

    # Back-compat alias — returns the medium (default) set
    @property
    def periods(self) -> Dict[str, int]:
        return self.periods_medium

    forecasts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        k: dict(v) for k, v in DEFAULT_FORECASTS.items()
    })

    # --- helpers -------------------------------------------------------------
    def weight_sum(self) -> float:
        return sum(max(0.0, v) for v in self.weights.values()) or 1.0

    def normalized_weights(self) -> Dict[str, float]:
        s = self.weight_sum()
        return {k: max(0.0, v) / s for k, v in self.weights.items()}

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_sidebar(cls) -> "Config":
        import streamlit as st
        cfg = cls()

        with st.sidebar:
            st.subheader("Data Source")
            src = st.selectbox(
                "Provider",
                ["yfinance", "twelvedata"],
                index=0 if cfg.data_source == "yfinance" else 1,
                help=(
                    "yfinance is free, no key, covers all 7 markets.  "
                    "Twelve Data needs a free API key from twelvedata.com."
                ),
            )
            cfg.data_source = src
            if src == "twelvedata":
                cfg.twelvedata_api_key = st.text_input(
                    "Twelve Data API key", value=cfg.twelvedata_api_key, type="password"
                )

            try:
                from .data import INTERVAL_CONFIG
            except ImportError:
                from data import INTERVAL_CONFIG
            interval_opts = list(INTERVAL_CONFIG.keys())
            current_idx = interval_opts.index(cfg.interval) if cfg.interval in interval_opts else 0
            cfg.interval = st.selectbox(
                "Data interval (bar size)",
                interval_opts,
                index=current_idx,
                help=(
                    "1d = swing trading (years of history).  "
                    "1h = intraday (60-day lookback).  "
                    "15m/5m/1m = scalping (7-30 day lookback).  "
                    "All 3 composite horizons + 3 forecast horizons are computed on this interval."
                ),
                key="interval_select",
            )
            cfg._interval_lookback = INTERVAL_CONFIG[cfg.interval]["lookback_days"]

            st.subheader("Weights (% — auto-normalized)")
            new_w = {}
            for k, label in MARKET_LABELS.items():
                new_w[k] = st.slider(
                    label, 0, 100, int(cfg.weights[k]), step=1, key=f"w_{k}"
                )
            cfg.weights = new_w

            with st.expander("Composite horizons", expanded=False):
                st.caption("Short — fast & responsive (scalping / quick swings)")
                cfg.periods_short["ema_fast"] = st.number_input("S EMA fast", 2, 100, cfg.periods_short["ema_fast"], key="cs_ema_fast")
                cfg.periods_short["ema_slow"] = st.number_input("S EMA slow", 5, 200, cfg.periods_short["ema_slow"], key="cs_ema_slow")
                cfg.periods_short["rsi_len"]  = st.number_input("S RSI",      2, 50,  cfg.periods_short["rsi_len"],  key="cs_rsi")
                cfg.periods_short["roc_len"]  = st.number_input("S ROC",      1, 100, cfg.periods_short["roc_len"],  key="cs_roc")

                st.caption("Medium — balanced (intraday / swing)")
                cfg.periods_medium["ema_fast"] = st.number_input("M EMA fast", 2, 200, cfg.periods_medium["ema_fast"], key="cm_ema_fast")
                cfg.periods_medium["ema_slow"] = st.number_input("M EMA slow", 5, 400, cfg.periods_medium["ema_slow"], key="cm_ema_slow")
                cfg.periods_medium["rsi_len"]  = st.number_input("M RSI",      2, 100, cfg.periods_medium["rsi_len"],  key="cm_rsi")
                cfg.periods_medium["roc_len"]  = st.number_input("M ROC",      1, 200, cfg.periods_medium["roc_len"],  key="cm_roc")

                st.caption("Long — slow & structural (position / macro)")
                cfg.periods_long["ema_fast"] = st.number_input("L EMA fast", 5, 400, cfg.periods_long["ema_fast"], key="cl_ema_fast")
                cfg.periods_long["ema_slow"] = st.number_input("L EMA slow", 10, 800, cfg.periods_long["ema_slow"], key="cl_ema_slow")
                cfg.periods_long["rsi_len"]  = st.number_input("L RSI",      2, 200, cfg.periods_long["rsi_len"],  key="cl_rsi")
                cfg.periods_long["roc_len"]  = st.number_input("L ROC",      1, 400, cfg.periods_long["roc_len"],  key="cl_roc")

            with st.expander("Forecast horizons (price-based)", expanded=False):
                st.caption("Short")
                cfg.forecasts["short"]["ema_fast"]  = st.number_input("FS EMA fast", 2, 50, cfg.forecasts["short"]["ema_fast"], key="fs_ema_fast")
                cfg.forecasts["short"]["ema_slow"]  = st.number_input("FS EMA slow", 5, 100, cfg.forecasts["short"]["ema_slow"], key="fs_ema_slow")
                cfg.forecasts["short"]["rsi"]       = st.number_input("FS RSI",      2, 50,  cfg.forecasts["short"]["rsi"], key="fs_rsi")
                st.caption("Medium")
                cfg.forecasts["medium"]["ema_fast"] = st.number_input("FM EMA fast", 2, 100, cfg.forecasts["medium"]["ema_fast"], key="fm_ema_fast")
                cfg.forecasts["medium"]["ema_slow"] = st.number_input("FM EMA slow", 5, 200, cfg.forecasts["medium"]["ema_slow"], key="fm_ema_slow")
                cfg.forecasts["medium"]["rsi"]      = st.number_input("FM RSI",      2, 50,  cfg.forecasts["medium"]["rsi"], key="fm_rsi")
                st.caption("Long")
                cfg.forecasts["long"]["ema_fast"]   = st.number_input("FL EMA fast", 5, 200, cfg.forecasts["long"]["ema_fast"], key="fl_ema_fast")
                cfg.forecasts["long"]["ema_slow"]   = st.number_input("FL EMA slow", 10, 400, cfg.forecasts["long"]["ema_slow"], key="fl_ema_slow")
                cfg.forecasts["long"]["rsi"]        = st.number_input("FL RSI",      2, 100, cfg.forecasts["long"]["rsi"], key="fl_rsi")

            if st.button("Reset to defaults"):
                st.rerun()

        return cfg
