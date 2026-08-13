"""Publishes synthetic transactions to the Kafka transactions topic.

Also mirrors every generated record into data/transactions_raw.xlsx via
excel_writer.py so the Excel data store stays in sync with the stream.
"""
import json
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

import config
from excel_writer import append_transaction
from transaction_generator import generate_transaction

CONNECT_RETRIES = 12
CONNECT_RETRY_DELAY_SECONDS = 5


def build_producer() -> KafkaProducer:
    """KafkaProducer's constructor connects synchronously and raises
    immediately on the first failed attempt - no built-in retry. A
    container started right after Kafka's port opens can still hit this,
    since the broker's internal listener isn't necessarily ready to accept
    client connections the instant the TCP port is reachable. Retry with a
    fixed backoff instead of crashing on the first attempt."""
    last_error: NoBrokersAvailable | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Kafka not reachable yet (attempt {attempt}/{CONNECT_RETRIES}), retrying...")
            time.sleep(CONNECT_RETRY_DELAY_SECONDS)
    raise last_error


def run(producer: KafkaProducer | None = None) -> None:
    producer = producer or build_producer()
    interval = 1.0 / config.TRANSACTIONS_PER_SECOND

    try:
        while True:
            transaction = generate_transaction()
            producer.send(
                config.KAFKA_TOPIC_TRANSACTIONS,
                key=transaction["transaction_id"],
                value=transaction,
            )
            append_transaction(transaction)
            time.sleep(interval)
    except KeyboardInterrupt:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
