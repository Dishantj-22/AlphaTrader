import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from kiteconnect import KiteTicker, KiteConnect
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaProducerClient
from shared.schemas.market import TickData, Exchange, OHLCVCandle

log = logging.getLogger(__name__)
settings = get_settings()


class MarketDataAdapter(ABC):
    """Base contract for all broker data adapters."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...


class ZerodhaAdapter(MarketDataAdapter):
    """
    Zerodha Kite Connect WebSocket adapter.
    Streams live tick data and publishes to Kafka.
    """

    def __init__(self, producer: KafkaProducerClient):
        self._kite = KiteConnect(api_key=settings.broker_api.zerodha_api_key)
        self._kite.set_access_token(settings.broker_api.zerodha_access_token)
        self._ticker = KiteTicker(
            api_key=settings.broker_api.zerodha_api_key,
            access_token=settings.broker_api.zerodha_access_token,
        )
        self._producer = producer
        self._instrument_map: dict[int, str] = {}
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        for raw in ticks:
            token = raw.get("instrument_token")
            symbol = self._instrument_map.get(token, str(token))
            tick = TickData(
                symbol=symbol,
                exchange=Exchange.NSE,
                timestamp=raw.get("timestamp", datetime.utcnow()),
                ltp=raw.get("last_price", 0.0),
                bid=raw.get("depth", {}).get("buy", [{}])[0].get("price", 0.0),
                ask=raw.get("depth", {}).get("sell", [{}])[0].get("price", 0.0),
                bid_qty=raw.get("depth", {}).get("buy", [{}])[0].get("quantity", 0),
                ask_qty=raw.get("depth", {}).get("sell", [{}])[0].get("quantity", 0),
                volume=raw.get("volume_traded", 0),
                oi=raw.get("oi"),
                change_pct=raw.get("change", 0.0),
            )
            self._producer.publish(settings.kafka.topic_market_tick, tick, key=symbol)

    def _on_connect(self, ws, response) -> None:
        log.info("ZerodhaAdapter connected")

    def _on_close(self, ws, code, reason) -> None:
        log.warning("ZerodhaAdapter disconnected | code=%s reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        log.error("ZerodhaAdapter error | code=%s reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        log.info("ZerodhaAdapter reconnecting | attempt=%d", attempts_count)

    async def connect(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ticker.connect, True)

    async def subscribe(self, symbols: list[str]) -> None:
        instruments = self._kite.instruments("NSE")
        token_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
        tokens = []
        for sym in symbols:
            token = token_map.get(sym)
            if token:
                self._instrument_map[token] = sym
                tokens.append(token)
        self._ticker.subscribe(tokens)
        self._ticker.set_mode(self._ticker.MODE_FULL, tokens)
        log.info("Subscribed to %d instruments via Zerodha", len(tokens))

    async def disconnect(self) -> None:
        self._ticker.close()
        self._producer.flush()
