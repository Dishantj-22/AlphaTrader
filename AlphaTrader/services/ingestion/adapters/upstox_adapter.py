import asyncio
import logging
import upstox_client
from datetime import datetime
from upstox_client.api import MarketQuoteApi, WebsocketApi
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaProducerClient
from shared.schemas.market import TickData, OHLCVCandle, Exchange
from .zerodha_adapter import MarketDataAdapter

log = logging.getLogger(__name__)
settings = get_settings()


class UpstoxAdapter(MarketDataAdapter):
    """
    Upstox v2 WebSocket adapter — acts as failover when Zerodha is down.
    Publishes ticks to the same Kafka topic ensuring consumers are broker-agnostic.
    """

    def __init__(self, producer: KafkaProducerClient):
        self._producer = producer
        config = upstox_client.Configuration()
        config.access_token = settings.broker_api.upstox_api_key
        self._api_client = upstox_client.ApiClient(configuration=config)
        self._market_api = MarketQuoteApi(self._api_client)
        self._subscribed_symbols: list[str] = []

    async def connect(self) -> None:
        log.info("UpstoxAdapter initialised (polling mode)")

    async def subscribe(self, symbols: list[str]) -> None:
        self._subscribed_symbols = symbols
        log.info("UpstoxAdapter subscribed to %d symbols", len(symbols))
        asyncio.create_task(self._polling_loop())

    async def _polling_loop(self) -> None:
        while True:
            try:
                instrument_keys = [f"NSE_EQ|{s}" for s in self._subscribed_symbols]
                quotes = self._market_api.get_full_market_quote(instrument_keys, api_version="v2")
                for key, data in (quotes.data or {}).items():
                    symbol = key.split("|")[-1]
                    tick = TickData(
                        symbol=symbol,
                        exchange=Exchange.NSE,
                        timestamp=datetime.utcnow(),
                        ltp=data.last_price or 0.0,
                        bid=data.depth.buy[0].price if data.depth and data.depth.buy else 0.0,
                        ask=data.depth.sell[0].price if data.depth and data.depth.sell else 0.0,
                        bid_qty=data.depth.buy[0].quantity if data.depth and data.depth.buy else 0,
                        ask_qty=data.depth.sell[0].quantity if data.depth and data.depth.sell else 0,
                        volume=data.volume or 0,
                        oi=data.oi,
                        change_pct=data.net_change or 0.0,
                    )
                    self._producer.publish(settings.kafka.topic_market_tick, tick, key=symbol)
            except Exception as exc:
                log.exception("UpstoxAdapter poll error: %s", exc)
            await asyncio.sleep(1)

    async def disconnect(self) -> None:
        self._subscribed_symbols = []
        self._producer.flush()
