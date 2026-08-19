"""
Composite scoring engine for the Gold (XAU/USD) Prediction Engine.

Pipeline:
    1. Compute Trend / Momentum / Strength for each market
    2. Blend with the 40/35/25 weights -> asset score
    3. Invert the sign of negatively-correlated assets (DXY, SP500)
    4. Weighted sum, normalised -> composite in [-100, +100]
    5. Classify into 5 buckets (Strong Bull -> Strong Bear)
    6. Compute three forecast horizons on Gold
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from .indicators import ema, rsi, roc
    from .config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS
except ImportError:
    from indicators import ema, rsi, roc
    from config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS


# -----------------------------------------------------------------------------
# Per-market asset score
# -----------------------------------------------------------------------------
def asset_score(close: pd.Series, periods: Dict[str, int]) -> pd.Series:
    """
    40% Trend + 35% Momentum + 25% Strength, clamped to [-100, +100].
    """
    e1 = close.ewm(span=periods["ema_fast"], adjust=False, min_periods=periods["ema_fast"]).mean()
    e2 = close.ewm(span=periods["ema_slow"], adjust=False, min_periods=periods["ema_slow"]).mean()
    t_score = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))

    r = rsi(close, periods["rsi_len"])
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_score = np.where(r > 60, pos, np.where(r < 40, neg, soft))

    rc = roc(close, periods["roc_len"])
    s_score = (rc * 20.0).clip(-100.0, 100.0)

    raw = t_score * 0.40 + m_score * 0.35 + s_score * 0.25
    return pd.Series(raw, index=close.index).clip(-100.0, 100.0).fillna(0.0)


# -----------------------------------------------------------------------------
# Composite + classification
# -----------------------------------------------------------------------------
@dataclass
class ScoreResult:
    composite: pd.Series
    per_market: Dict[str, pd.Series]
    contributions: pd.DataFrame
    label: pd.Series
    confidence: pd.Series


def composite_score(
    data: Dict[str, pd.DataFrame],
    config: Config,
    periods: Optional[Dict[str, int]] = None,
) -> ScoreResult:
    """
    Compute the Gold prediction composite for every bar in `data['gold']`.

    Parameters
    ----------
    data : dict of DataFrames
        Aligned OHLCV for all markets.
    config : Config
        Active engine configuration (weights, correlations).
    periods : dict, optional
        Override the period set used for this composite.  If None, uses
        config.periods (medium / default).
    """
    if periods is None:
        periods = config.periods

    weights = config.normalized_weights()
    per_market: Dict[str, pd.Series] = {}
    contributions: Dict[str, pd.Series] = {}

    for mkt, df in data.items():
        if mkt not in weights:
            continue
        close = df["Close"]
        score = asset_score(close, periods)
        if mkt in NEGATIVE_CORRELATIONS:
            score = -score
        per_market[mkt] = score
        contributions[mkt] = score * weights[mkt]

    contrib_df = pd.DataFrame(contributions)
    composite = contrib_df.sum(axis=1).clip(-100.0, 100.0).rename("composite")

    label, confidence = _classify(composite)
    return ScoreResult(
        composite=composite,
        per_market=per_market,
        contributions=contrib_df,
        label=label,
        confidence=confidence,
    )


def _classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """5-bucket classification with confidence."""
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(62.5, index=score.index, dtype=float)

    label = label.mask(score >  70, "Strong Bullish")
    label = label.mask((score >  40) & (score <=  70), "Bullish")
    label = label.mask((score >= -40) & (score <=  40), "Neutral")
    label = label.mask((score >= -70) & (score <  -40), "Bearish")
    label = label.mask(score <  -70, "Strong Bearish")

    conf = conf.mask(score >  70, 95.0)
    conf = conf.mask((score >  40) & (score <=  70), 82.5)
    conf = conf.mask((score >= -40) & (score <=  40), 62.5)
    conf = conf.mask((score >= -70) & (score <  -40), 82.5)
    conf = conf.mask(score <  -70, 95.0)
    return label, conf


# -----------------------------------------------------------------------------
# Forecast horizons (price-based, on Gold only)
# -----------------------------------------------------------------------------
def forecast_score(close: pd.Series, ema_fast: int, ema_slow: int, rsi_len: int) -> pd.Series:
    """
    Single horizon score: 70% trend (EMA fast/slow) + 30% momentum (RSI),
    clamped to [-100, +100].
    """
    e1 = ema(close, ema_fast)
    e2 = ema(close, ema_slow)
    t = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))
    t_series = pd.Series(t, index=close.index).fillna(0.0)

    r = rsi(close, rsi_len)
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_series = pd.Series(np.where(r > 60, pos, np.where(r < 40, neg, soft)),
                         index=close.index).fillna(0.0)

    return (t_series * 0.70 + m_series * 0.30).clip(-100.0, 100.0)


def forecasts(data: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.Series]:
    """Compute the three forecast horizon scores for Gold."""
    close = data["gold"]["Close"]
    out: Dict[str, pd.Series] = {}
    for name, cfg in config.forecasts.items():
        out[name] = forecast_score(
            close,
            ema_fast=cfg["ema_fast"],
            ema_slow=cfg["ema_slow"],
            rsi_len=cfg["rsi"],
        )
    return out


def forecast_classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Per-horizon label + confidence in [50, 95]%."""
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(60.0, index=score.index, dtype=float)

    bull = score >  40
    bear = score < -40

    label = label.mask(bull, "Bullish")
    label = label.mask(bear, "Bearish")

    conf = conf.mask(
        bull,
        75.0 + ((score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0,
    )
    conf = conf.mask(
        bear,
        75.0 + ((-score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0,
    )
    neutral = ~bull & ~bear
    conf = conf.mask(
        neutral,
        50.0 + (1.0 - score.abs() / 40.0).clip(0.0, 1.0) * 25.0,
    )
    return label, conf


# -----------------------------------------------------------------------------
# Market regime (Gold Bull / Gold Bear / Transition)
# -----------------------------------------------------------------------------
def market_regime(per_market: Dict[str, pd.Series]) -> pd.Series:
    """
    Gold Bull : DXY bearish (< -20) AND IEF bullish (> 20)
    Gold Bear : DXY bullish (> 20) AND IEF bearish (< -20)
    Else      : Transition
    """
    dxy = per_market["dxy"]
    ief = per_market["ief"]

    bull = (dxy < -20) & (ief > 20)
    bear = (dxy > 20) & (ief < -20)

    regime = pd.Series("Transition", index=dxy.index, dtype=object)
    regime = regime.mask(bull, "Gold Bull")
    regime = regime.mask(bear, "Gold Bear")
    return regime


# -----------------------------------------------------------------------------
# Institutional flow meter  (0-100)
# -----------------------------------------------------------------------------
def flow_meter(
    composite: pd.Series,
    dxy_close: pd.Series,
    vix_close: pd.Series,
    ief_close: pd.Series,
) -> pd.Series:
    """
    0-100 gauge blending:
        50% composite,
        20% inverted DXY (lower dollar = better for gold),
        15% VIX (higher fear = better for gold),
        15% IEF (higher bond prices / lower yields = better for gold).
    """
    composite_n = (composite + 100.0) / 2.0

    dxy_v = dxy_close.clip(lower=0.0).fillna(100.0)
    dxy_n = ((1.0 - (dxy_v - 90.0) / 20.0).clip(0.0, 1.0)) * 100.0

    vix_v = vix_close.clip(lower=0.0).fillna(15.0)
    vix_n = (vix_v / 40.0).clip(0.0, 1.0) * 100.0

    ief_v = ief_close.ffill()
    ief_ma = ief_v.rolling(200, min_periods=1).mean()
    ief_pct = ((ief_v - ief_ma) / ief_ma * 100.0).fillna(0.0)
    ief_n = (ief_pct + 50.0).clip(0.0, 100.0)

    raw = composite_n * 0.50 + dxy_n * 0.20 + vix_n * 0.15 + ief_n * 0.15
    return raw.clip(0.0, 100.0).fillna(50.0)
