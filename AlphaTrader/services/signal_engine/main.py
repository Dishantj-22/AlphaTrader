"""
Signal engine service entrypoint.
Consumes computed feature events and emits trading signals.
"""

import asyncio
import json
import logging
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaConsumerClient
from services.signal_engine.generator.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)
settings = get_settings()


async def main():
    generator = SignalGenerator()
    consumer = KafkaConsumerClient(
        topics=[settings.kafka.topic_features],
        group_id=settings.kafka.consumer_group_signals,
        schema=_FeatureEvent,
    )
    log.info("Signal engine consumer loop started")
    loop = asyncio.get_event_loop()
    while True:
        event = await loop.run_in_executor(None, lambda: consumer.poll(timeout=1.0))
        if event:
            try:
                generator.process_feature_event(event.data)
            except Exception as exc:
                log.exception("Signal generation error: %s", exc)


from pydantic import BaseModel
from typing import Any


class _FeatureEvent(BaseModel):
    data: dict[str, Any]


if __name__ == "__main__":
    asyncio.run(main())
