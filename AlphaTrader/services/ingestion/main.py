"""
Ingestion service entrypoint.
Starts Zerodha WebSocket adapter (Upstox as hot-standby)
and the OHLCV candle producer for all tracked symbols.
"""

import asyncio
import logging
import signal
import sys

from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaProducerClient
from services.ingestion.adapters.zerodha_adapter import ZerodhaAdapter
from services.ingestion.producers.ohlcv_producer import OHLCVProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)
settings = get_settings()

# Top 50 NSE symbols to track — extend as needed
WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "WIPRO", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "BAJAJFINSV", "TECHM",
    "ONGC", "COALINDIA", "DIVISLAB", "DRREDDY", "CIPLA",
]


async def main():
    producer = KafkaProducerClient()
    adapter = ZerodhaAdapter(producer)
    ohlcv = OHLCVProducer(producer)

    log.info("Connecting market data adapter...")
    await adapter.subscribe(WATCHLIST)
    await adapter.connect()

    log.info("Starting OHLCV live streams...")
    tasks = [asyncio.create_task(ohlcv.stream_live_candles(sym, 0, "5minute")) for sym in WATCHLIST[:10]]

    def _shutdown(sig, _frame):
        log.info("Shutdown signal received (%s)", sig)
        for t in tasks:
            t.cancel()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
