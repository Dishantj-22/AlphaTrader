"""
FastAPI application — API Gateway.
Single entry point for mobile apps, dashboards, and external clients.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from shared.config.settings import get_settings
from services.ml_engine.inference.engine import get_inference_engine
from .routers import signals, market, auth, health
from .websocket import hub

log = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AlphaTrader API Gateway starting...")
    get_inference_engine()      # warm model cache
    yield
    log.info("AlphaTrader API Gateway shutting down.")


app = FastAPI(
    title="AlphaTrader API",
    description="Indian Stock Market Prediction & Signal Engine",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://alphatrader.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/internal/metrics")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["Signals"])
app.include_router(market.router, prefix="/api/v1/market", tags=["Market"])
app.include_router(health.router, prefix="/internal", tags=["Ops"])
app.include_router(hub.router, prefix="/ws", tags=["WebSocket"])
