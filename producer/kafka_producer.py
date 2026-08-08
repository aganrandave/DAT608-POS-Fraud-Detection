"""Publishes synthetic transactions to the Kafka transactions topic.

Also mirrors every generated record into data/transactions_raw.xlsx via
excel_writer.py so the Excel data store stays in sync with the stream.
"""
import json
import time

from kafka import KafkaProducer

import config
from excel_writer import append_transaction
from transaction_generator import generate_transaction


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )


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
