import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scoring"))

import kafka_publisher  # noqa: E402
from kafka.errors import NoBrokersAvailable  # noqa: E402


def test_publish_score_does_not_raise_when_kafka_unreachable():
    # publish_score is a best-effort side channel off /score's main
    # response path - a Kafka outage must never surface as an error to the
    # caller of the scoring API.
    kafka_publisher._producer = None
    kafka_publisher._connect_failed = False
    with patch("kafka_publisher.KafkaProducer", side_effect=NoBrokersAvailable("no broker")):
        kafka_publisher.publish_score({"transaction_id": "txn-1", "alert_tier": "low"})
    assert kafka_publisher._connect_failed is True


def test_publish_score_reuses_cached_producer_without_reconnecting():
    kafka_publisher._producer = None
    kafka_publisher._connect_failed = False
    with patch("kafka_publisher.KafkaProducer") as mock_producer_cls:
        mock_producer_cls.return_value.send.return_value = None
        kafka_publisher.publish_score({"transaction_id": "txn-1", "alert_tier": "low"})
        kafka_publisher.publish_score({"transaction_id": "txn-2", "alert_tier": "high"})
    mock_producer_cls.assert_called_once()
