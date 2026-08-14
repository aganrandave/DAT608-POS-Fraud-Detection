"""Publishes each scored transaction to the pos-fraud-scores Kafka topic,
materialized by alerts/streams/02_create_scored_stream.sql - the source
02_create_scored_stream.sql was written against, which nothing ever
actually published to (append_score() only ever wrote to
data/fraud_scores.xlsx).

Kept best-effort and separate from scoring/main.py's core request path:
/score's contract (score, write Excel, respond) must not depend on Kafka
being reachable, so failures here are logged and swallowed rather than
raised.
"""
import json
import os

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_SCORES = os.getenv("KAFKA_TOPIC_SCORES", "pos-fraud-scores")

_producer: KafkaProducer | None = None
_connect_failed = False


def _get_producer() -> KafkaProducer | None:
    global _producer, _connect_failed
    if _producer is not None:
        return _producer
    if _connect_failed:
        return None
    try:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            request_timeout_ms=3000,
        )
        return _producer
    except NoBrokersAvailable as exc:
        # Only pay the connection-timeout cost once - a request-serving
        # process can't afford to retry-with-backoff on every /score call
        # the way a long-running consumer like producer/pipeline can.
        _connect_failed = True
        print(f"kafka_publisher: could not connect to Kafka, publishing disabled: {exc}")
        return None


def publish_score(score: dict) -> None:
    producer = _get_producer()
    if producer is None:
        return
    try:
        producer.send(KAFKA_TOPIC_SCORES, key=score["transaction_id"], value=score)
    except Exception as exc:  # noqa: BLE001 - best-effort side channel, never raise into /score
        print(f"kafka_publisher: failed to publish score for {score.get('transaction_id')}: {exc}")
