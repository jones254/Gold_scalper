# Gold (XAU/USD) Institutional Prediction Engine

A self-contained, mobile-friendly web app for forecasting the next directional
bias of **Gold (XAU/USD)** using a weighted multi-market macro model.

This is a **Gold-adapted fork** of the Germany 40 engine — same math, same
buckets, same forecast horizons, but tuned for gold's unique macro drivers.

## What's in the box

| Module | What it does |
|---|---|
| `app.py` | Streamlit UI (Live + Backtest + Trades tabs) |
| `config.py` | Weights, periods, data-source selection (gold-tuned) |
| `data.py` | `yfinance` (default) + `twelvedata` swap |
| `indicators.py` | EMA / RSI / ROC (TradingView-equivalent) |
| `scoring.py` | 7-market composite, 5-bucket classification, forecasts |
| `backtest.py` | Long-flat backtest with equity curve + metrics |
| `trades.py` | R-based trade engine (pullback entries, ATR stops, BE) |
| `trade_ui.py` | Trade dashboard (metrics, equity, log) |

## Gold Macro Drivers

| Market | Weight | Role | Sign |
|---|---|---|---|
| **DXY** | 25% | Dollar strength — #1 driver | Inverted |
| **IEF** | 20% | US Treasury bond prices (yield proxy) | Positive |
| **Silver** | 15% | High-beta precious metals confirmation | Positive |
| **S&P 500** | 15% | Risk-on / safe-haven substitution | Inverted |
| **EUR/USD** | 10% | Dollar proxy (ECB divergence) | Positive |
| **VIX** | 10% | Fear gauge / flight-to-safety | Positive |
| **Gold** | 5% | Self-referential momentum | Positive |

**Why IEF instead of ^TNX?**  
IEF is a bond *price* ETF. When yields fall, IEF rises — same direction as gold.
This keeps the engine purely price-based and avoids yfinance data quirks.
If you prefer raw yields, swap `IEF` for `^TNX` and **invert** it.

## Run locally (5 min)

```bash
# 1. Clone / unzip into a folder
cd gold_engine

# 2. Create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser. To view it on
your phone on the same Wi-Fi, find your laptop's local IP (e.g. `192.168.1.42`)
and visit `http://192.168.1.42:8501` from your phone.

## Deploy to the public internet

### Option A — Streamlit Cloud (free, easiest)

1. Push the folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → pick the repo → entry point `app.py`.
4. Wait ~1 min.  You'll get a URL like
   `https://your-app.streamlit.app` that you can open from anywhere.

### Option B — Render / Railway / Fly.io

Each of these reads a `requirements.txt` automatically.  Start command:

```
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Option C — VPS (Hetzner / DigitalOcean)

```bash
# On the server
git clone <your repo>
cd gold_engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nohup streamlit run app.py --server.port=80 --server.address=0.0.0.0 &
```

## Data source — yfinance vs Twelve Data

The app ships with **yfinance** (free, no key).  It covers all 7 markets:

| Market | yfinance ticker | Twelve Data ticker |
|---|---|---|
| DXY | `DX-Y.NYB` | `DXY` |
| IEF | `IEF` | `IEF` |
| Silver | `SI=F` | `XAG/USD` |
| S&P 500 | `SPY` | `SPY` |
| EUR/USD | `EURUSD=X` | `EUR/USD` |
| VIX | `^VIX` | `VIX` |
| Gold | `GC=F` | `XAU/USD` |

**Switching to Twelve Data** is one dropdown + one API key in the sidebar:
1. Get a free key at [twelvedata.com](https://twelvedata.com)
   (800 requests/day, 8/min).
2. In the app sidebar → **Data Source → twelvedata** → paste key.

## Data interval — swing, intraday, scalping

A **Data interval** dropdown in the sidebar picks the bar size the engine
runs on. All 7 markets + the composite + all 3 forecast horizons are
recomputed on the selected interval.

| Interval | Bars/day | Max lookback | Best for |
|---|---|---|---|
| `1d`  | 1 | 3 years | **Swing trading** (the default) |
| `1h`  | 24 | 60 days | **Intraday** (1-2 day holds) |
| `30m` | 32 | 30 days | Intraday / end-of-day |
| `15m` | 64 | 30 days | **Scalping / day trading** |
| `5m`  | 192 | 30 days | Scalping |
| `1m`  | 960 | 5 days | Ultra-short scalping |

**Important:** the 3 forecast horizons (Short / Medium / Long) are bar-count
horizons, not wall-clock horizons. The header shows a tooltip explaining exactly
what each horizon means in the chosen interval.

The backtest holding-period input is also in **bars**, not days.

## Backtest

The **🧪 Backtest** tab runs a long-flat strategy on historical data:

- **Long** Gold when the model says `Bullish` or `Strong Bullish`
- **Cash** otherwise
- Default 5-bar forward window (configurable)

You get:
- Strategy vs buy-and-hold equity curve
- CAGR, Sharpe, max drawdown, total return
- Per-signal hit rate and average forward return
- Score vs return scatter
- Last 30 signals table

## Trade Engine

The **🎯 Trades** tab runs the R-based execution engine:

- Pullback limit entries (not market orders)
- ATR-based stops + structural swing-stop blend
- Automatic breakeven at +1R
- Risk-based position sizing
- Both long and short directions

Tweak risk %, stop/target multipliers, and min R:R in the sidebar.

## Phone / PWA notes

The app is mobile-responsive out of the box.  For a more "app-like" feel:

1. Open the deployed URL in your phone's browser.
2. **iOS Safari** → Share → *Add to Home Screen*.
3. **Android Chrome** → menu → *Add to Home Screen*.

It launches fullscreen, no browser chrome.

## Performance

- First data pull: 5–15 s (yfinance downloads 7 tickers).
- Cached for 5 min in Streamlit (`@st.cache_data(ttl=300)`).
- Composite + backtest: <1 s for 3 years of daily data.
- Suitable for free-tier hosting.

## Caveats

- yfinance is unofficial and can break when Yahoo changes its API.
  The Twelve Data swap is the recommended fallback.
- Backtest assumes zero commissions and zero slippage.  Use realistic
  numbers in your own broker before sizing real capital.
- Intraday intervals (15m/5m/1m) have limited lookback (5-30 days) on
  yfinance.  This is a yfinance limitation, not a bug.
- Twelve Data free tier (8 calls/min) is too tight for frequent
  intraday refreshes.  Upgrade to a paid tier or increase the cache TTL.
- The DXY-Gold correlation can break down during regime shifts (e.g.
  when both are bid as safe havens). The composite naturally dampens
  this by blending 7 markets.

## License

MIT — do whatever you want.
