from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class DatabaseSettings(BaseSettings):
    timescale_url: str = Field(default="postgresql+asyncpg://trader:trader@localhost:5432/alphatrader", alias="TIMESCALE_URL")
    postgres_url: str = Field(default="postgresql+asyncpg://trader:trader@localhost:5432/alphatrader", alias="POSTGRES_URL")
    mongo_uri: str = Field(default="mongodb://localhost:27017/alphatrader", alias="MONGO_URI")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    topic_market_tick: str = "market.tick"
    topic_ohlcv: str = "market.ohlcv"
    topic_news_raw: str = "news.raw"
    topic_sentiment: str = "news.sentiment"
    topic_features: str = "features.computed"
    topic_signals: str = "signals.generated"
    consumer_group_features: str = "cg-feature-engine"
    consumer_group_ml: str = "cg-ml-engine"
    consumer_group_signals: str = "cg-signal-engine"


class MLSettings(BaseSettings):
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    model_registry_bucket: str = Field(default="alphatrader-models", alias="MODEL_REGISTRY_BUCKET")
    feature_lookback_candles: int = 60
    prediction_horizon_minutes: int = 75
    retrain_cron: str = "0 18 * * 1-5"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    db: DatabaseSettings = DatabaseSettings()
    kafka: KafkaSettings = KafkaSettings()
    ml: MLSettings = MLSettings()


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
