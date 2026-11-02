"""
Offline training script.
Fetches historical OHLCV data, computes features, trains all models,
and registers them to MLflow.

Usage:
    python ml/training/train_all.py --symbol RELIANCE --days 730
    python ml/training/train_all.py --symbol all --days 365
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config.settings import get_settings
from services.feature_engine.indicators.technical import TechnicalIndicatorEngine, IndicatorConfig
from services.feature_engine.patterns.candlestick import CandlestickPatternEngine
from services.ml_engine.models.gradient_boosting import GradientBoostingModel
from services.ml_engine.models.lstm_model import LSTMModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)
settings = get_settings()

TRAINING_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
]


def fetch_historical(symbol: str, days: int) -> pd.DataFrame:
    from datetime import datetime, timedelta
    kite = KiteConnect(api_key=settings.broker_api.zerodha_api_key)
    kite.set_access_token(settings.broker_api.zerodha_access_token)

    instruments = kite.instruments("NSE")
    token_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
    token = token_map.get(symbol)
    if not token:
        raise ValueError(f"Symbol not found: {symbol}")

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    raw = kite.historical_data(token, from_dt, to_dt, "day")
    df = pd.DataFrame(raw)
    df.rename(columns={"date": "timestamp"}, inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    indicator_engine = TechnicalIndicatorEngine(IndicatorConfig())
    pattern_engine = CandlestickPatternEngine()
    df = indicator_engine.compute(df)
    df = pattern_engine.detect_all(df)
    return df


def train_symbol(symbol: str, days: int) -> None:
    log.info("=== Training %s | days=%d ===", symbol, days)

    df = fetch_historical(symbol, days)
    log.info("Fetched %d candles for %s", len(df), symbol)

    df = build_features(df)
    log.info("Features computed: %d rows, %d columns", len(df), len(df.columns))

    # Gradient Boosting
    gb = GradientBoostingModel(model_name=f"xgboost_lgbm_{symbol.lower()}")
    gb_metrics = gb.train(df)
    log.info("GB training complete: %s", gb_metrics)

    # LSTM
    lstm = LSTMModelTrainer()
    lstm_metrics = lstm.train(df, epochs=30)
    log.info("LSTM training complete: %s", lstm_metrics)


def main():
    parser = argparse.ArgumentParser(description="AlphaTrader model trainer")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol or 'all'")
    parser.add_argument("--days", type=int, default=730, help="Historical days to fetch")
    args = parser.parse_args()

    symbols = TRAINING_SYMBOLS if args.symbol == "all" else [args.symbol.upper()]
    for sym in symbols:
        try:
            train_symbol(sym, args.days)
        except Exception as exc:
            log.exception("Training failed for %s: %s", sym, exc)


if __name__ == "__main__":
    main()
