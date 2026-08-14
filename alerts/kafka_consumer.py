"""Consumes the pos-fraud-alerts topic (materialized by
streams/03_create_alert_stream.sql) and appends each alert to
data/alerts.xlsx via excel_writer.append_alert.

This was the missing piece behind SCRUM-36's "runs as a Docker service"
gap: excel_writer.py only ever had the write helper, nothing actually read
from Kafka and called it.
"""
import json
import os
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from excel_writer import append_alert

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "pos-fraud-alerts")

CONNECT_RETRIES = 12
CONNECT_RETRY_DELAY_SECONDS = 5


def normalize_alert(raw: dict) -> dict:
    """ksqlDB serializes JSON values keyed by its (uppercased) column
    names - e.g. TRANSACTION_ID, ALERT_TIER - but excel_writer.COLUMNS and
    append_alert() expect lowercase keys, matching data/alerts.xlsx's
    actual header row."""
    return {key.lower(): value for key, value in raw.items()}


def build_consumer() -> KafkaConsumer:
    """Mirrors producer/kafka_producer.py's build_producer() retry loop -
    KafkaConsumer's constructor also connects synchronously and can lose a
    race against Kafka's listener becoming ready right after boot."""
    last_error: NoBrokersAvailable | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return KafkaConsumer(
                KAFKA_TOPIC_ALERTS,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="alerts-excel-writer",
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Kafka not reachable yet (attempt {attempt}/{CONNECT_RETRIES}), retrying...")
            time.sleep(CONNECT_RETRY_DELAY_SECONDS)
    raise last_error


def run(consumer: KafkaConsumer | None = None) -> None:
    consumer = consumer or build_consumer()
    for message in consumer:
        alert = append_alert(normalize_alert(message.value))
        print(f"Appended alert {alert['transaction_id']} tier={alert['alert_tier']}")


if __name__ == "__main__":
    run()
