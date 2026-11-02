FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
RUN pip install -r requirements/base.txt

COPY shared/ shared/
COPY services/ services/

FROM base AS api_gateway
COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt
EXPOSE 8000
CMD ["uvicorn", "services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

FROM base AS ingestion
COPY requirements/ingestion.txt requirements/ingestion.txt
RUN pip install -r requirements/ingestion.txt
CMD ["python", "-m", "services.ingestion.main"]

FROM base AS feature_engine
CMD ["python", "-m", "services.feature_engine.main"]

FROM base AS ml_engine
COPY requirements/ml.txt requirements/ml.txt
RUN pip install -r requirements/ml.txt
CMD ["python", "-m", "services.ml_engine.main"]

FROM base AS signal_engine
CMD ["python", "-m", "services.signal_engine.main"]

FROM base AS news_sentiment
COPY requirements/nlp.txt requirements/nlp.txt
RUN pip install -r requirements/nlp.txt
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
CMD ["python", "-m", "services.news_sentiment.main"]
