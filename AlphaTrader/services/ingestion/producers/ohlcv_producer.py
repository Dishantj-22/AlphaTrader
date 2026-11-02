"""
Free OHLCV producer using yfinance.
No broker API key needed.
"""

import logging
import asyncio
import pandas as pd
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaProducerClient
from shared.schemas.market import OHLCVCandle, Exchange
from .yfinance_adapter import fetch_ohlcv

log = logging.getLogger(__name__)
settings = get_settings()


class OHLCVProducer:
    def __init__(self, producer: KafkaProducerClient):
        self._producer = producer

    async def backfill(self, symbol: str, interval: str = "day", days: int = 365) -> None:
        log.info("Backfilling %s | interval=%s days=%d", symbol, interval, days)
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(None, lambda: fetch_ohlcv(symbol, interval, days))
            for ts, row in df.iterrows():
                candle = OHLCVCandle(
                    symbol=symbol,
                    exchange=Exchange.NSE,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                self._producer.publish(settings.kafka.topic_ohlcv, candle, key=f"{symbol}_{interval}")
            self._producer.flush()
            log.info("Backfilled %d candles for %s", len(df), symbol)
        except Exception as exc:
            log.exception("Backfill failed for %s: %s", symbol, exc)

    async def stream_live_candles(self, symbol: str, interval: str = "5minute") -> None:
        """Polls yfinance every 5 minutes for the latest candle."""
        log.info("Starting live candle stream | symbol=%s interval=%s", symbol, interval)
        while True:
            try:
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(None, lambda: fetch_ohlcv(symbol, interval, days=3))
                if not df.empty:
                    row = df.iloc[-1]
                    candle = OHLCVCandle(
                        symbol=symbol,
                        exchange=Exchange.NSE,
                        timestamp=df.index[-1],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    )
                    self._producer.publish(settings.kafka.topic_ohlcv, candle, key=f"{symbol}_{interval}")
            except Exception as exc:
                log.exception("Live candle error for %s: %s", symbol, exc)
            await asyncio.sleep(300)  # poll every 5 minutes
