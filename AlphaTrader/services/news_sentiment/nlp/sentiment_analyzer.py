"""
Sentiment analysis pipeline using FinBERT.
Consumes raw news from Kafka, scores each article, and publishes
structured SentimentResult events.
"""

import logging
from datetime import datetime
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.config.settings import get_settings
from shared.messaging.kafka_client import KafkaConsumerClient, KafkaProducerClient
from shared.schemas.news import RawNewsArticle, SentimentResult, SentimentLabel

log = logging.getLogger(__name__)
settings = get_settings()

FINBERT_MODEL = "ProsusAI/finbert"
MAX_LENGTH = 512
IMPACT_THRESHOLDS = {"HIGH": 0.7, "MEDIUM": 0.4}


class SentimentAnalyzer:
    """
    FinBERT-based financial sentiment engine.
    Runs inference on CPU or GPU depending on availability.
    Batches multiple articles for throughput efficiency.
    """

    def __init__(self):
        log.info("Loading FinBERT model: %s", FINBERT_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._labels = ["positive", "negative", "neutral"]
        self._consumer = KafkaConsumerClient(
            topics=[settings.kafka.topic_news_raw],
            group_id="cg-sentiment",
            schema=RawNewsArticle,
        )
        self._producer = KafkaProducerClient()
        log.info("FinBERT loaded on device: %s", self._device)

    def analyze(self, text: str) -> dict[str, float]:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        return {label: prob for label, prob in zip(self._labels, probs)}

    def score_to_composite(self, scores: dict[str, float]) -> float:
        return scores["positive"] - scores["negative"]

    def classify_impact(self, composite: float) -> str:
        abs_score = abs(composite)
        if abs_score >= IMPACT_THRESHOLDS["HIGH"]:
            return "HIGH"
        if abs_score >= IMPACT_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def run(self) -> None:
        log.info("SentimentAnalyzer consumer loop started")
        while True:
            article: Optional[RawNewsArticle] = self._consumer.poll(timeout=1.0)
            if not article:
                continue
            try:
                text = article.headline
                scores = self.analyze(text)
                composite = self.score_to_composite(scores)
                label = SentimentLabel.POSITIVE if composite > 0.05 else (
                    SentimentLabel.NEGATIVE if composite < -0.05 else SentimentLabel.NEUTRAL
                )
                result = SentimentResult(
                    article_id=article.article_id,
                    symbol=article.companies_mentioned[0] if article.companies_mentioned else None,
                    label=label,
                    positive_score=scores["positive"],
                    negative_score=scores["negative"],
                    neutral_score=scores["neutral"],
                    composite_score=composite,
                    impact_level=self.classify_impact(composite),
                    model_used="finbert",
                    analyzed_at=datetime.utcnow(),
                )
                key = result.symbol or "MARKET"
                self._producer.publish(settings.kafka.topic_sentiment, result, key=key)
            except Exception as exc:
                log.exception("Sentiment analysis failed for article %s: %s", article.article_id, exc)
