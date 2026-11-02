import json
import logging
from typing import Any, Callable, Type
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from pydantic import BaseModel
from shared.config.settings import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


def _build_producer_config() -> dict:
    return {
        "bootstrap.servers": settings.kafka.bootstrap_servers,
        "acks": "all",
        "retries": 5,
        "retry.backoff.ms": 300,
        "compression.type": "snappy",
        "linger.ms": 5,
        "batch.size": 65536,
    }


def _build_consumer_config(group_id: str) -> dict:
    return {
        "bootstrap.servers": settings.kafka.bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "max.poll.interval.ms": 300000,
    }


class KafkaProducerClient:
    def __init__(self):
        self._producer = Producer(_build_producer_config())

    def publish(self, topic: str, message: BaseModel, key: str | None = None) -> None:
        payload = message.model_dump_json().encode("utf-8")
        key_bytes = key.encode("utf-8") if key else None
        self._producer.produce(topic, value=payload, key=key_bytes, on_delivery=self._delivery_report)
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()

    @staticmethod
    def _delivery_report(err, msg):
        if err:
            log.error("Kafka delivery failed | topic=%s error=%s", msg.topic(), err)
        else:
            log.debug("Kafka delivered | topic=%s partition=%d offset=%d", msg.topic(), msg.partition(), msg.offset())


class KafkaConsumerClient:
    def __init__(self, topics: list[str], group_id: str, schema: Type[BaseModel]):
        self._consumer = Consumer(_build_consumer_config(group_id))
        self._consumer.subscribe(topics)
        self._schema = schema

    def poll(self, timeout: float = 1.0) -> BaseModel | None:
        msg = self._consumer.poll(timeout=timeout)
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                return None
            raise KafkaException(msg.error())
        try:
            data = json.loads(msg.value().decode("utf-8"))
            parsed = self._schema(**data)
            self._consumer.commit(asynchronous=False)
            return parsed
        except Exception as exc:
            log.exception("Failed to deserialize Kafka message | topic=%s error=%s", msg.topic(), exc)
            return None

    def close(self) -> None:
        self._consumer.close()
