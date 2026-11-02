"""
Feature pipeline: consumes OHLCV Kafka events, computes all features,
and publishes enriched feature vectors back to Kafka.
"""

import logging
import asyncio
import pandas as pd
from collections import defaultdict
from datetime import datetime

from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaConsumerClient, KafkaProducerClient
from shared.schemas.market import OHLCVCandle
from .indicators.technical import TechnicalIndicatorEngine, IndicatorConfig
from .patterns.candlestick import CandlestickPatternEngine

log = logging.getLogger(__name__)
settings = get_settings()

LOOKBACK = settings.ml.feature_lookback_candles


class FeaturePipeline:
    """
    Stateful streaming pipeline.
    Maintains a rolling candle buffer per symbol and emits feature rows
    whenever a new candle arrives and the buffer is full.
    """

    def __init__(self):
        self._consumer = KafkaConsumerClient(
            topics=[settings.kafka.topic_ohlcv],
            group_id=settings.kafka.consumer_group_features,
            schema=OHLCVCandle,
        )
        self._producer = KafkaProducerClient()
        self._indicator_engine = TechnicalIndicatorEngine(IndicatorConfig())
        self._pattern_engine = CandlestickPatternEngine()
        self._buffers: dict[str, list[OHLCVCandle]] = defaultdict(list)

    async def run(self) -> None:
        log.info("FeaturePipeline started")
        loop = asyncio.get_event_loop()
        while True:
            candle = await loop.run_in_executor(None, lambda: self._consumer.poll(timeout=1.0))
            if candle:
                await self._process_candle(candle)

    async def _process_candle(self, candle: OHLCVCandle) -> None:
        key = f"{candle.symbol}_{candle.exchange.value}"
        self._buffers[key].append(candle)

        if len(self._buffers[key]) > LOOKBACK + 50:
            self._buffers[key] = self._buffers[key][-(LOOKBACK + 50):]

        if len(self._buffers[key]) < LOOKBACK:
            return

        df = self._to_dataframe(self._buffers[key])
        df = self._indicator_engine.compute(df)
        df = self._pattern_engine.detect_all(df)

        feature_row = df.iloc[-1].to_dict()
        feature_row["symbol"] = candle.symbol
        feature_row["exchange"] = candle.exchange.value
        feature_row["computed_at"] = datetime.utcnow().isoformat()

        self._producer.publish(
            settings.kafka.topic_features,
            _FeatureEvent(data=feature_row),
            key=candle.symbol,
        )

    @staticmethod
    def _to_dataframe(candles: list[OHLCVCandle]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "vwap": c.vwap,
            }
            for c in candles
        ]).set_index("timestamp").sort_index()


from pydantic import BaseModel
from typing import Any


class _FeatureEvent(BaseModel):
    data: dict[str, Any]
