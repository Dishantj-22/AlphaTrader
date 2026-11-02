# AlphaTrader — Indian Stock Market Prediction System

> AI-powered NSE/BSE trading signal engine for short-term trading.
> Combines technical analysis, candlestick patterns, volume analytics,
> and news sentiment into a weighted ensemble that generates
> Buy / Sell / Hold signals with confidence scores and risk levels.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Quick Start — No Setup Required](#2-quick-start--no-setup-required)
3. [Installation](#3-installation)
4. [predict.py — Complete Command Reference](#4-predictpy--complete-command-reference)
5. [Data Providers](#5-data-providers)
6. [Signal Types Explained](#6-signal-types-explained)
7. [Reading the Output](#7-reading-the-output)
8. [Top 5 Picks Feature](#8-top-5-picks-feature)
9. [Risk Management](#9-risk-management)
10. [How the Prediction Engine Works](#10-how-the-prediction-engine-works)
11. [Technical Indicators Used](#11-technical-indicators-used)
12. [Candlestick Patterns Detected](#12-candlestick-patterns-detected)
13. [Ensemble Scoring System](#13-ensemble-scoring-system)
14. [Full Microservices Architecture](#14-full-microservices-architecture)
15. [Project Structure](#15-project-structure)
16. [Environment Variables](#16-environment-variables)
17. [Docker Setup](#17-docker-setup)
18. [Kubernetes Deployment](#18-kubernetes-deployment)
19. [Model Training](#19-model-training)
20. [Running Tests](#20-running-tests)
21. [Accuracy Expectations](#21-accuracy-expectations)
22. [Disclaimer](#22-disclaimer)

---

## 1. What This System Does

AlphaTrader scans NSE stocks and generates probabilistic trading signals.

For each stock it:
- Fetches historical OHLCV data (Yahoo Finance by default — free, no key)
- Computes 25+ technical indicators (RSI, MACD, EMA, ADX, Bollinger Bands, ATR, etc.)
- Detects candlestick patterns (Engulfing, Hammer, Shooting Star, Marubozu, etc.)
- Analyses volume pressure (OBV, CMF, MFI, volume ratio)
- Combines all signals into a weighted ensemble confidence score
- Applies ATR-based stoploss and target calculation
- Outputs a ranked signal: STRONG BUY / BUY / WEAK BUY / HOLD / WEAK SELL / SELL / STRONG SELL

It supports three trading styles:
- **Intraday** — same day entry and exit
- **BTST** — Buy Today Sell Tomorrow (1-day hold)
- **Swing** — 2-5 day positional trades

---

## 2. Quick Start — No Setup Required

```bash
# Install minimal dependencies
pip install yfinance pandas numpy pandas-ta

# Predict a single stock
python predict.py --symbol RELIANCE

# Top 5 short-term picks from full Nifty 50
python predict.py --top5 --style btst

# Scan multiple stocks
python predict.py --symbol TCS INFY HDFCBANK SBIN --table
```

That is it. No API key. No broker account. No database.

---

## 3. Installation

### Minimal (predict.py only)

```bash
pip install -r requirements/predict.txt
```

Contents of `requirements/predict.txt`:
```
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
pandas-ta==0.3.14b
```

### Full System (microservices + ML training)

```bash
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
pip install -r requirements/api.txt
pip install -r requirements/nlp.txt
```

### Optional Broker Providers

```bash
# Zerodha
pip install kiteconnect

# Upstox
pip install upstox-python-sdk

# Angel One
pip install smartapi-python pyotp

# Groww (uses requests — already installed via yfinance)
# No extra package needed
```

### Python Version

Tested on Python 3.11 and 3.14.
On Python 3.14 a numba stub is automatically injected — no manual action needed.

---

## 4. predict.py — Complete Command Reference

### Basic Syntax

```bash
python predict.py [OPTIONS]
```

### All Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--symbol` | string(s) | `RELIANCE` | One or more NSE symbols, or `all` for Nifty 50 |
| `--style` | choice | `swing` | Trading style: `intraday`, `btst`, `swing` |
| `--days` | integer | `365` | Historical days to analyse (min 400 fetched internally) |
| `--provider` | choice | auto | Data source: `yfinance`, `zerodha`, `upstox`, `angel`, `groww` |
| `--table` | flag | off | Show compact summary table instead of full cards |
| `--top5` | flag | off | Show Top 5 trade opportunities after predictions |

---

### Single Stock — Full Card

```bash
python predict.py --symbol RELIANCE
python predict.py --symbol TCS
python predict.py --symbol HDFCBANK
```

Output:
```
+---------------------------------------------------+
| [ +] TCS         Signal: BUY                     |
+---------------------------------------------------+
|   Confidence  [##############......] 70.2%        |
|   Entry       Rs.   2,221.00                      |
|   Target      Rs.   2,443.00                      |
|   Stop Loss   Rs.   2,110.00                      |
|   R:R Ratio   2.00x                               |
|   Holding     3 day(s)                            |
+---------------------------------------------------+
|   RSI    62.4        ADX       28.5               |
|   ATR%    1.80%      Vol Ratio  1.42x             |
+---------------------------------------------------+
|   Score Breakdown                                 |
|     Technical   :  72.3%                         |
|     Volume      :  64.1%                         |
|     Candlestick :  75.0%                         |
|     Data points : 249                            |
+---------------------------------------------------+
```

---

### Multiple Stocks — Cards

```bash
python predict.py --symbol RELIANCE TCS INFY
python predict.py --symbol HDFCBANK ICICIBANK AXISBANK KOTAKBANK SBIN
```

---

### Multiple Stocks — Compact Table

```bash
python predict.py --symbol RELIANCE TCS INFY HDFCBANK --table
python predict.py --symbol RELIANCE TCS INFY HDFCBANK SBIN WIPRO BAJFINANCE ITC --table
```

Output:
```
======================================================================
  ALPHATRADER SIGNAL SUMMARY  (8 stocks scanned)
======================================================================
  SYMBOL         SIGNAL          CONF      ENTRY     TARGET         SL   R:R
  ------------------------------------------------------------------
  [ +] BAJFINANCE   BUY            68.9%  Rs.  1,069  Rs.  1,140  Rs.  1,034  2.0x
  [ ~] SBIN         WEAK BUY       55.8%  Rs.  1,044  Rs.  1,104  Rs.  1,015  2.0x
  ...
======================================================================
```

---

### Full Nifty 50 Scan

```bash
# All 50 stocks — compact table
python predict.py --symbol all --table

# All 50 stocks — full cards (verbose)
python predict.py --symbol all
```

---

### Trading Style

```bash
# Intraday (same day)
python predict.py --symbol RELIANCE --style intraday

# BTST — Buy Today Sell Tomorrow
python predict.py --symbol RELIANCE --style btst
python predict.py --symbol all --style btst --table

# Swing trading (2-5 days, default)
python predict.py --symbol RELIANCE --style swing
```

---

### Adjust Historical Window

```bash
# Use 2 years of data for more robust signals
python predict.py --symbol RELIANCE --days 730

# Use 1 year (default)
python predict.py --symbol TCS --days 365
```

Note: Regardless of `--days`, the system always fetches at least 400 days
internally so that EMA-200 and ADX have enough warmup bars.

---

### Top 5 Picks

```bash
# Top 5 from full Nifty 50 (default when --top5 used alone)
python predict.py --top5

# Top 5 BTST picks
python predict.py --top5 --style btst

# Top 5 from your own watchlist
python predict.py --symbol RELIANCE TCS INFY HDFCBANK SBIN WIPRO BAJFINANCE ITC --top5

# Top 5 + full table together
python predict.py --symbol all --table --top5

# Top 5 with a specific provider
python predict.py --top5 --provider zerodha --style btst
```

---

### Sector Watchlists (manual)

```bash
# Banking sector
python predict.py --symbol HDFCBANK ICICIBANK AXISBANK KOTAKBANK SBIN INDUSINDBK --table

# IT sector
python predict.py --symbol TCS INFY WIPRO HCLTECH TECHM --table

# Pharma sector
python predict.py --symbol SUNPHARMA DRREDDY CIPLA DIVISLAB APOLLOHOSP --table

# Auto sector
python predict.py --symbol MARUTI M&M TATAMOTORS BAJAJFINSV EICHERMOT --table

# FMCG sector
python predict.py --symbol HINDUNILVR ITC NESTLEIND BRITANNIA TATACONSUM --table

# Energy sector
python predict.py --symbol RELIANCE ONGC BPCL COALINDIA NTPC POWERGRID --table

# Metals and infra
python predict.py --symbol TATASTEEL JSWSTEEL HINDALCO LT GRASIM ULTRACEMCO --table
```

---

### Provider Selection

```bash
# Default — Yahoo Finance (free, no key)
python predict.py --symbol RELIANCE

# Zerodha (requires ZERODHA_API_KEY + ZERODHA_ACCESS_TOKEN in .env)
python predict.py --symbol RELIANCE --provider zerodha

# Upstox (requires UPSTOX_ACCESS_TOKEN in .env)
python predict.py --symbol TCS --provider upstox

# Angel One (requires ANGEL_API_KEY + ANGEL_CLIENT_ID + ANGEL_PASSWORD + ANGEL_TOTP_SECRET)
python predict.py --symbol INFY --provider angel

# Groww (requires GROWW_AUTH_TOKEN in .env)
python predict.py --symbol HDFCBANK --provider groww
```

---

### Combined Examples

```bash
# Best daily routine: full Nifty 50 top 5 BTST scan
python predict.py --top5 --style btst

# Pre-market swing scan with table
python predict.py --symbol all --table --style swing

# Deep analysis on a single stock
python predict.py --symbol RELIANCE --days 730 --style swing

# Banking sector BTST picks
python predict.py --symbol HDFCBANK ICICIBANK AXISBANK KOTAKBANK SBIN INDUSINDBK --top5 --style btst

# IT sector with Zerodha data
python predict.py --symbol TCS INFY WIPRO HCLTECH TECHM --table --provider zerodha
```

---

## 5. Data Providers

| Provider | Cost | Key Required | Data Quality | How to Get Key |
|---|---|---|---|---|
| **yfinance** | Free | None | End-of-day only | Nothing needed |
| **Zerodha** | Broker account | Yes | Real-time + historical | [kite.trade](https://kite.trade/docs/connect/v3/) |
| **Upstox** | Broker account | Yes | Real-time + historical | [upstox.com/developer](https://upstox.com/developer/api-documentation/) |
| **Angel One** | Broker account | Yes | Real-time + historical | [smartapi.angelbroking.com](https://smartapi.angelbroking.com/docs) |
| **Groww** | Broker account | Unofficial token | Historical | Browser DevTools (see below) |

### Auto-Selection Logic

When `--provider` is not specified, the system picks the best available:

```
Zerodha → Upstox → Angel One → Groww → yfinance
```

### How to Get Groww Token

1. Open `https://groww.in` in Chrome
2. Log in to your Groww account
3. Press `F12` to open DevTools
4. Go to **Network** tab
5. Refresh the page
6. Click any request to `groww.in`
7. Find the **Authorization** header
8. Copy the value after `Bearer ` (just the token part)
9. Paste into `.env` as `GROWW_AUTH_TOKEN=your_token_here`

Note: Groww token expires periodically. Refresh it from DevTools when it stops working.

---

## 6. Signal Types Explained

| Signal | Icon | Confidence | Meaning |
|---|---|---|---|
| `STRONG BUY` | `[++]` | >= 80% | Very high conviction bullish setup |
| `BUY` | `[ +]` | >= 65% | Clear bullish setup with confirmation |
| `WEAK BUY` | `[ ~]` | >= 55% | Mild bullish bias, use smaller size |
| `HOLD` | `[ =]` | 45–55% | No clear edge, stay out |
| `WEAK SELL` | `[ v]` | 35–45% | Mild bearish bias |
| `SELL` | `[ -]` | <= 35% | Clear bearish setup |
| `STRONG SELL` | `[--]` | <= 20% | Very high conviction bearish setup |

### ADX Override Rule

When ADX < 20 (weak / no trend), all signals are capped at BUY / SELL.
STRONG BUY and STRONG SELL are never generated in a sideways market.

---

## 7. Reading the Output

### Confidence Bar

```
Confidence  [##############......] 70.2%
```

- Each `#` = 5% confidence
- 20 blocks total = 100%
- Target is >= 65% for actionable signals

### Entry / Target / Stoploss

Calculated using ATR (Average True Range):
- **Stoploss** = Entry - (ATR x 1.5) for BUY signals
- **Target**   = Entry + (ATR x 3.0) for BUY signals
- **R:R Ratio** is always 2:1 (risk 1 to make 2)

### Score Breakdown

| Score | What It Measures |
|---|---|
| Technical | EMA alignment, RSI, MACD, ADX, Supertrend, Bollinger Bands |
| Volume | Volume ratio, CMF, MFI — institutional money flow |
| Candlestick | Last candle pattern — Engulfing, Hammer, Doji, etc. |

### Data Points

Number of daily candles used after warmup period.
More data points = more reliable signal.

---

## 8. Top 5 Picks Feature

The `--top5` flag surfaces the highest-conviction setups after scanning.

- **Top 3 BUY picks**: ranked by confidence descending
- **Top 2 SELL picks**: ranked by lowest confidence (strongest bearish)

Each pick card shows:
```
#1  [ +] ULTRACEMCO  BUY    Confidence: 68.9%
    Entry  Rs. 12,094.00   Target Rs. 12,743.34  (+5.4%)
    SL     Rs. 11,769.33   R:R    2.00x   Hold: 1d
    Why  : strong trend, RSI overbought 65
```

### Why / Rationale Labels

| Label | Condition |
|---|---|
| `strong trend` | Technical score >= 65% |
| `mild trend` | Technical score 50–65% |
| `weak trend` | Technical score < 50% |
| `high volume` | Volume score >= 65% |
| `ADX {value}` | ADX >= 25 (strong directional move) |
| `RSI oversold {value}` | RSI < 40 (potential bounce) |
| `RSI overbought {value}` | RSI > 60 (momentum running) |
| `bullish candle` | Candlestick score >= 70% |
| `bearish candle` | Candlestick score <= 30% |

### Automatic Symbol Expansion

When `--top5` is used without `--symbol`, it automatically scans all Nifty 50 stocks:

```bash
# These two are equivalent
python predict.py --top5
python predict.py --symbol all --top5
```

---

## 9. Risk Management

### Position Sizing Rule

The system uses fixed fractional position sizing:
```
Max risk per trade = 1% of portfolio
Position size = (Portfolio x 1%) / Risk per share
Risk per share = Entry - Stoploss
```

### ATR-Based Levels

```
Stoploss = Entry - (ATR x 1.5)   [for BUY]
Target   = Entry + (ATR x 3.0)   [for BUY]

Stoploss = Entry + (ATR x 1.5)   [for SELL]
Target   = Entry - (ATR x 3.0)   [for SELL]
```

ATR is the 14-period Average True Range — it adapts to each stock's actual volatility.
A volatile stock like ADANIENT gets wider stops than a stable one like NESTLEIND.

### Risk Validation Rules

Signals are rejected automatically if:
- R:R ratio < 1.5
- Sector exposure would exceed 25%
- Portfolio exposure would exceed 80%

### Recommended Capital Allocation

| Portfolio Size | Max per Trade | Max Open Positions |
|---|---|---|
| Rs. 1 lakh | Rs. 1,000 | 5 |
| Rs. 5 lakh | Rs. 5,000 | 10 |
| Rs. 10 lakh | Rs. 10,000 | 15 |

---

## 10. How the Prediction Engine Works

```
Raw OHLCV Data (yfinance / broker API)
        |
        v
Feature Engineering
  - 25+ Technical Indicators
  - Candlestick Pattern Score
  - Volume Pressure Score
        |
        v
Weighted Ensemble
  Technical   40%
  Sentiment   25%  (static 0.5 when no live news feed)
  Candlestick 20%
  Volume      15%
        |
        v
Confidence Score (0 - 100%)
        |
        v
ADX Trend Filter
  ADX < 20 --> cap signal to BUY / SELL (no STRONG)
        |
        v
Signal Classification
  >= 80% --> STRONG BUY
  >= 65% --> BUY
  >= 55% --> WEAK BUY
  45-55% --> HOLD
  35-45% --> WEAK SELL
  <= 35% --> SELL
  <= 20% --> STRONG SELL
        |
        v
ATR Risk Calculator
  Stoploss, Target, R:R, Position Size
        |
        v
Output Card / Table / Top 5
```

---

## 11. Technical Indicators Used

### Trend

| Indicator | Period | Usage |
|---|---|---|
| EMA | 9, 20, 50, 200 | Trend direction and alignment |
| EMA Cross | 9 vs 20 | Short-term momentum shift |
| Price vs EMA 50 | — | Medium-term trend confirmation |
| Price vs EMA 200 | — | Long-term trend confirmation |
| Supertrend | 10, 3.0 | Dynamic trend with direction flag |
| ADX | 14 | Trend strength (>25 = strong) |
| DI+ / DI- | 14 | Directional movement |

### Momentum

| Indicator | Period | Usage |
|---|---|---|
| RSI | 14 | Overbought (>70) / Oversold (<30) |
| MACD | 12, 26, 9 | Momentum and signal line cross |
| MACD Histogram | — | Momentum acceleration |
| ROC | 10 | Rate of change |
| CCI | 20 | Cyclical momentum |

### Volatility

| Indicator | Period | Usage |
|---|---|---|
| ATR | 14 | Volatility for position sizing |
| ATR % | 14 | ATR as % of price |
| Bollinger Bands | 20, 2.0 | Price channel and squeeze |
| BB % | — | Position within bands |
| BB Width | — | Volatility expansion / contraction |

### Volume

| Indicator | Period | Usage |
|---|---|---|
| OBV | — | Cumulative volume direction |
| CMF | 20 | Chaikin Money Flow — institutional buying |
| MFI | 14 | Money Flow Index |
| Volume Ratio | 20 | Current vs 20-day average volume |
| Volume Spike | — | Volume > 2x average |

### Market Structure

| Metric | Usage |
|---|---|
| 52W High | Distance from yearly high |
| 52W Low | Distance from yearly low |
| Gap Up / Gap Down | Opening gap detection |
| Daily Return | Previous day return |
| HL Range | High-Low as % of close |

---

## 12. Candlestick Patterns Detected

| Pattern | Score | Bias |
|---|---|---|
| Bullish Engulfing | +0.75 | Bullish reversal |
| Marubozu Bullish | +0.65 | Strong bullish continuation |
| Hammer | +0.60 | Bullish reversal at support |
| Bullish Harami | +0.40 | Mild bullish inside bar |
| Doji | 0.00 | Indecision / neutral |
| Bearish Harami | -0.40 | Mild bearish inside bar |
| Shooting Star | -0.60 | Bearish reversal at resistance |
| Marubozu Bearish | -0.65 | Strong bearish continuation |
| Bearish Engulfing | -0.75 | Bearish reversal |

Score range: -1.0 (very bearish) to +1.0 (very bullish)
Score is normalised to 0-100% for display in the Score Breakdown.

---

## 13. Ensemble Scoring System

### Weights

```
Technical Score   40%   (EMA, RSI, MACD, ADX, BB, Supertrend, Volume Ratio, CCI)
News Sentiment    25%   (static 0.5 = neutral when no live news feed connected)
Candlestick       20%   (pattern recognition on last 3 candles)
Volume Pressure   15%   (CMF, MFI, volume ratio combined)
```

### Technical Score Formula

Each indicator casts a vote (0 or 1) with a weight:

```python
close > ema_9          weight 0.05
ema_9 > ema_20         weight 0.07
ema_20 > ema_50        weight 0.06
close > ema_50         weight 0.06
close > ema_200        weight 0.08
supertrend == UP       weight 0.08
DI+ > DI-              weight 0.05
RSI between 30-70      weight 0.04
RSI > 50               weight 0.05
MACD > 0               weight 0.05
MACD histogram > 0     weight 0.05
ROC > 0                weight 0.04
CCI > 0                weight 0.03
BB% between 0.2-0.8    weight 0.04
BB% > 0.5              weight 0.04
Volume ratio > 1.0     weight 0.05
CMF > 0                weight 0.06
MFI > 50               weight 0.04
% from 52w high > -10  weight 0.04
% from 52w low > 10    weight 0.04
ADX > 20               weight 0.04
```

### Volume Score Formula

```python
volume_score = (volume_ratio/5) * 0.40
             + ((cmf + 1) / 2)  * 0.35
             + (mfi / 100)      * 0.25
```

### Final Confidence Formula

```python
candle_norm = (candle_score + 1) / 2      # -1..+1  →  0..1
confidence  = technical * 0.40
            + candle_norm * 0.20
            + volume * 0.15
            + sentiment * 0.25
```

---

## 14. Full Microservices Architecture

```
NSE / BSE Market Data
        |
        v
+-----------------------------------------------+
|  Ingestion Service                            |
|  yfinance / Zerodha / Upstox / Angel / Groww  |
|  Publishes to Kafka: market.tick, market.ohlcv|
+-----------------------------------------------+
        |
        +-------------------+-------------------+
        v                   v                   v
+---------------+   +---------------+   +-----------+
| Feature Engine|   | News Sentiment|   | (future)  |
| 25+ indicators|   | FinBERT NLP   |   | Social    |
| Candlestick   |   | NSE/RSS scrape|   | Intel     |
| Kafka:        |   | Kafka:        |   |           |
| features.*    |   | news.sentiment|   |           |
+---------------+   +---------------+   +-----------+
        |                   |
        +-------------------+
                |
                v
+-----------------------------------------------+
|  ML Engine                                    |
|  XGBoost + LightGBM (tabular features)       |
|  LSTM + Attention (sequence model)            |
|  MLflow model registry                        |
+-----------------------------------------------+
                |
                v
+-----------------------------------------------+
|  Signal Engine                                |
|  Weighted Ensemble (40/25/20/15)              |
|  ATR Risk Calculator                          |
|  Redis dedup (5 min cooldown per symbol)      |
|  Kafka: signals.generated                     |
+-----------------------------------------------+
                |
                v
+-----------------------------------------------+
|  API Gateway (FastAPI)                        |
|  REST endpoints + JWT auth                    |
|  WebSocket hub (live signal push)             |
|  Prometheus metrics                           |
+-----------------------------------------------+
                |
        +-------+-------+
        v               v
  Dashboard App    Mobile App
  (React +         (Flutter /
  TradingView)     React Native)
```

### Kafka Topics

| Topic | Producer | Consumer | Contents |
|---|---|---|---|
| `market.tick` | Ingestion | — | Live tick data |
| `market.ohlcv` | Ingestion | Feature Engine | OHLCV candles |
| `news.raw` | News Scraper | Sentiment Analyzer | Raw articles |
| `news.sentiment` | Sentiment | Signal Engine | Scored articles |
| `features.computed` | Feature Engine | Signal Engine | Feature vectors |
| `signals.generated` | Signal Engine | API Gateway | Trading signals |

---

## 15. Project Structure

```
AlphaTrader/
|
|-- predict.py                      # Standalone prediction CLI (start here)
|
|-- shared/
|   |-- config/
|   |   `-- settings.py             # Centralised config via pydantic-settings
|   |-- schemas/
|   |   |-- market.py               # OHLCVCandle, TickData, InstrumentMeta
|   |   |-- signal.py               # TradingSignal, SignalType, ModelContribution
|   |   `-- news.py                 # RawNewsArticle, SentimentResult
|   |-- database/
|   |   `-- session.py              # Async SQLAlchemy session factory
|   `-- messaging/
|       `-- kafka_client.py         # KafkaProducerClient, KafkaConsumerClient
|
|-- services/
|   |-- ingestion/
|   |   |-- adapters/
|   |   |   |-- provider_registry.py   # Pluggable data provider registry
|   |   |   |-- yfinance_adapter.py    # Free Yahoo Finance adapter
|   |   |   |-- zerodha_adapter.py     # Zerodha Kite Connect WebSocket
|   |   |   `-- upstox_adapter.py      # Upstox v2 polling adapter
|   |   |-- producers/
|   |   |   `-- ohlcv_producer.py      # OHLCV candle Kafka publisher
|   |   `-- main.py                    # Ingestion service entrypoint
|   |
|   |-- feature_engine/
|   |   |-- indicators/
|   |   |   `-- technical.py           # 25+ technical indicators engine
|   |   |-- patterns/
|   |   |   `-- candlestick.py         # Rule-based candlestick engine
|   |   |-- pipeline.py                # Streaming feature pipeline (Kafka)
|   |   `-- main.py                    # Feature engine service entrypoint
|   |
|   |-- news_sentiment/
|   |   |-- scrapers/
|   |   |   `-- news_scraper.py        # NSE + RSS news scraper
|   |   |-- nlp/
|   |   |   `-- sentiment_analyzer.py  # FinBERT sentiment pipeline
|   |   `-- main.py                    # News sentiment service entrypoint
|   |
|   |-- ml_engine/
|   |   |-- models/
|   |   |   |-- gradient_boosting.py   # XGBoost + LightGBM with walk-forward CV
|   |   |   `-- lstm_model.py          # LSTM + Attention sequence model
|   |   |-- inference/
|   |   |   `-- engine.py              # Model inference gateway (MLflow)
|   |   `-- training/
|   |
|   |-- signal_engine/
|   |   |-- ensemble/
|   |   |   `-- engine.py              # Weighted ensemble decision engine
|   |   |-- generator/
|   |   |   `-- signal_generator.py    # End-to-end signal pipeline runner
|   |   `-- main.py                    # Signal engine service entrypoint
|   |
|   |-- risk_engine/
|   |   `-- calculator/
|   |       `-- risk_engine.py         # ATR sizing, R:R validation, exposure checks
|   |
|   `-- api_gateway/
|       |-- main.py                    # FastAPI app with lifespan + middleware
|       |-- routers/
|       |   |-- signals.py             # GET /api/v1/signals/
|       |   |-- market.py              # GET /api/v1/market/quote/{symbol}
|       |   |-- health.py              # GET /internal/health
|       |   `-- auth.py                # POST /api/v1/auth/token
|       |-- auth/
|       |   |-- jwt_handler.py         # JWT encode / decode
|       |   `-- dependencies.py        # FastAPI OAuth2 dependency
|       `-- websocket/
|           `-- hub.py                 # WebSocket connection manager
|
|-- ml/
|   `-- training/
|       `-- train_all.py               # Offline model training script
|
|-- tests/
|   |-- conftest.py                    # Pytest fixtures
|   |-- test_technical_indicators.py   # Indicator engine unit tests
|   |-- test_candlestick_patterns.py   # Pattern engine unit tests
|   |-- test_ensemble_engine.py        # Ensemble decision tests
|   |-- test_risk_engine.py            # Risk calculator tests
|   `-- test_api_gateway.py            # FastAPI integration tests
|
|-- infrastructure/
|   |-- kubernetes/
|   |   |-- base/
|   |   |   |-- namespace.yaml         # alphatrader namespace + configmap
|   |   |   `-- ingress.yaml           # Nginx ingress with WebSocket support
|   |   `-- services/
|   |       |-- api-gateway.yaml       # Deployment + Service + HPA
|   |       `-- workers.yaml           # Signal engine, feature engine, NLP worker
|   `-- monitoring/
|       |-- prometheus.yml             # Prometheus scrape config
|       `-- grafana-datasources.yml    # Grafana datasource provisioning
|
|-- requirements/
|   |-- predict.txt     # Minimal — just for predict.py
|   |-- base.txt        # Shared: pydantic, kafka, redis, db, aiohttp
|   |-- api.txt         # FastAPI, uvicorn, websockets, prometheus
|   |-- ml.txt          # PyTorch, XGBoost, LightGBM, MLflow
|   |-- nlp.txt         # Transformers, torch, sentencepiece
|   |-- ingestion.txt   # kiteconnect, upstox-python-sdk, pyotp
|   `-- dev.txt         # pytest, ruff, mypy, pre-commit
|
|-- .github/
|   `-- workflows/
|       `-- ci-cd.yml           # Lint → Test → Build → Deploy to AKS
|
|-- docker-compose.yml          # Full local stack (Kafka, TimescaleDB, Redis, MLflow, Grafana)
|-- Dockerfile                  # Multi-stage build for all 5 services
|-- pyproject.toml              # Ruff, mypy, pytest, coverage config
|-- .env.example                # Template for all environment variables
|-- .pre-commit-config.yaml     # Ruff + trailing whitespace hooks
|-- .gitignore
`-- README.md
```

---

## 16. Environment Variables

Copy `.env.example` to `.env` and fill in only what you need.

```bash
cp .env.example .env
```

### Minimal (predict.py only)

Nothing. `yfinance` requires no configuration.

### Application

```env
ENVIRONMENT=development          # development | production
LOG_LEVEL=INFO
SECRET_KEY=your-long-random-secret-here
```

### Databases (microservices mode)

```env
TIMESCALE_URL=postgresql+asyncpg://trader:trader@localhost:5432/alphatrader
POSTGRES_URL=postgresql+asyncpg://trader:trader@localhost:5432/alphatrader
MONGO_URI=mongodb://localhost:27017/alphatrader
REDIS_URL=redis://localhost:6379/0
```

### Kafka (microservices mode)

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### MLflow (model training)

```env
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Zerodha Kite Connect

```env
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_daily_access_token
```

Access token must be refreshed daily.
Get it via: `https://kite.trade/connect/login?api_key=YOUR_KEY`

### Upstox v2

```env
UPSTOX_ACCESS_TOKEN=your_access_token
```

### Angel One Smart API

```env
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret_key
```

TOTP secret is the base32 key from Google Authenticator setup.

### Groww (unofficial)

```env
GROWW_AUTH_TOKEN=your_bearer_token
```

From browser DevTools → Network → any groww.in request → Authorization header.

### Azure (optional cloud services)

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

---

## 17. Docker Setup

### Start Full Local Stack

```bash
# Start all infrastructure services
docker compose up -d zookeeper kafka timescaledb mongo redis mlflow prometheus grafana

# Start all application services
docker compose up -d ingestion feature_engine news_sentiment signal_engine api_gateway

# Start everything at once
docker compose up -d
```

### Check Status

```bash
docker compose ps
docker compose logs api_gateway -f
docker compose logs signal_engine -f
```

### Useful URLs After Starting

| Service | URL | Credentials |
|---|---|---|
| API Gateway | http://localhost:8000 | — |
| API Docs (Swagger) | http://localhost:8000/docs | — |
| Kafka UI | http://localhost:8080 | — |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### Stop Everything

```bash
docker compose down
docker compose down -v    # also removes volumes (deletes all data)
```

---

## 18. Kubernetes Deployment

### Prerequisites

- Azure CLI installed
- kubectl installed
- AKS cluster running

### Create Namespace

```bash
kubectl apply -f infrastructure/kubernetes/base/namespace.yaml
```

### Create Secrets

```bash
kubectl create secret generic alphatrader-secrets \
  --from-literal=SECRET_KEY=your-secret \
  --from-literal=TIMESCALE_URL=your-db-url \
  --from-literal=REDIS_URL=your-redis-url \
  -n alphatrader
```

### Deploy Services

```bash
kubectl apply -f infrastructure/kubernetes/base/
kubectl apply -f infrastructure/kubernetes/services/
```

### Check Rollout

```bash
kubectl rollout status deployment/api-gateway -n alphatrader
kubectl get pods -n alphatrader
kubectl logs -l app=signal-engine -n alphatrader -f
```

### Scale Manually

```bash
# Scale API gateway to 5 replicas
kubectl scale deployment api-gateway --replicas=5 -n alphatrader

# HPA handles auto-scaling based on CPU (60%) and Memory (75%)
kubectl get hpa -n alphatrader
```

---

## 19. Model Training

Trains XGBoost + LightGBM and LSTM models on historical data.
Requires Zerodha credentials (for intraday data) or can use yfinance for daily.

```bash
# Install ML dependencies
pip install -r requirements/ml.txt

# Train a single symbol
python ml/training/train_all.py --symbol RELIANCE

# Train with 2 years of data
python ml/training/train_all.py --symbol TCS --days 730

# Train all Nifty 50 symbols
python ml/training/train_all.py --symbol all --days 365
```

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser
# http://localhost:5000
```

Each training run logs:
- Cross-validation accuracy (walk-forward splits)
- Feature importance
- Model artifacts (xgboost, lightgbm, lstm)

### Promote Model to Production

```bash
# Via MLflow UI: Experiments → Select run → Register Model → Set stage to Production
# Or via CLI:
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage('xgboost_lgbm_ensemble', version=1, stage='Production')
"
```

---

## 20. Running Tests

```bash
# Install test dependencies
pip install -r requirements/dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=services --cov=shared --cov-report=term-missing

# Run specific test file
pytest tests/test_ensemble_engine.py -v
pytest tests/test_risk_engine.py -v
pytest tests/test_technical_indicators.py -v
pytest tests/test_candlestick_patterns.py -v
pytest tests/test_api_gateway.py -v

# Run with minimum coverage gate (70%)
pytest tests/ --cov=services --cov=shared --cov-fail-under=70
```

### Test Coverage Map

| Test File | Covers |
|---|---|
| `test_technical_indicators.py` | EMA, RSI, BB bounds, volume ratio, no-NaN after compute |
| `test_candlestick_patterns.py` | Engulfing, hammer, harami detection, score bounds |
| `test_ensemble_engine.py` | Bullish/bearish/mixed inputs, ADX downgrade, confidence bounds |
| `test_risk_engine.py` | Stoploss direction, R:R minimum, 1% risk cap, sector exposure |
| `test_api_gateway.py` | Login, invalid credentials, auth-required endpoints |

---

## 21. Accuracy Expectations

| Configuration | Expected Win Rate |
|---|---|
| Technical indicators only | 55–60% |
| Technical + Volume | 60–65% |
| Technical + Candlestick | 62–67% |
| Technical + Volume + Candlestick | 65–70% |
| Full Ensemble + Live News Sentiment | 70–80% |

These are realistic expectations based on similar systems in live Indian markets.
No system can guarantee profitability.

### What Improves Accuracy

- Live broker data instead of yfinance (eliminates end-of-day delay)
- Live news sentiment (FinBERT) connected to the ensemble
- Trained ML models (XGBoost/LSTM) replacing heuristic scoring
- Larger historical dataset for training (2+ years)
- Market regime detection (trending vs sideways markets)

### What Hurts Accuracy

- Gap openings and sudden news events (unpredictable)
- Low liquidity stocks (spreads eat into R:R)
- SEBI circuit breakers (force exits at bad prices)
- Earnings week (high volatility, patterns break down)

---

## 22. Disclaimer

```
This software is for educational and research purposes only.

AlphaTrader does NOT provide financial advice.
All signals are probabilistic estimates, not guaranteed predictions.
Past performance of any algorithm does not guarantee future results.

Trading in equities, futures, and options involves substantial risk.
You can lose more than your initial investment.

Always:
  - Apply your own judgement before entering any trade
  - Use strict stoploss discipline
  - Never risk more than you can afford to lose
  - Consult a SEBI-registered investment advisor for financial advice

The authors of this software are not responsible for any trading
losses incurred by using this system.
```

---

## Support and Contributions

- Report bugs via GitHub Issues
- Feature requests welcome via Pull Requests
- For questions, open a Discussion

---

*Built for Indian markets. NSE and BSE listed stocks only.*
*Data via Yahoo Finance (free) or broker APIs (optional).*
