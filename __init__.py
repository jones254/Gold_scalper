"""
Gold (XAU/USD) Institutional Prediction Engine
================================================

A self-contained, multi-market macro prediction engine for Gold.

Architecture:
    data.py       - yfinance / Twelve Data abstraction
    indicators.py - EMA, RSI, ROC, ATR (Pine-equivalent implementations)
    scoring.py    - 7-market composite + forecasts (gold macro drivers)
    backtest.py   - historical backtest engine
    config.py     - weights, periods, data-source selection
"""

__version__ = "1.0.0"
