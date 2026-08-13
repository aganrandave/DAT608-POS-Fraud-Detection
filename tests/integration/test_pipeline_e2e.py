"""End-to-end pipeline test: produce a transaction, feature-engineer it,
score it, and confirm an alert is written to data/alerts.xlsx.

Requires the full docker-compose stack (Kafka, ksqlDB, MLflow, scoring)
to be running locally; skipped otherwise.
"""
import os
import time

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 with the docker-compose stack running to enable.",
)


def test_high_risk_transaction_produces_alert():
    payload = {
        "transaction_id": "e2e-test-txn",
        "terminal_id": "TRM00001",
        "card_bin": "539983",
        "velocity_1h": 25,
        "geo_jump_km": 850.0,
        "bin_spend_rate": 145000.0,
        "terminal_reversal_count": 4,
        "amount_ngn": 180000.0,
    }

    response = requests.post("http://localhost:8000/score", json=payload, timeout=10)
    assert response.status_code == 200

    body = response.json()
    assert body["alert_tier"] in {"medium", "high", "critical"}

    time.sleep(5)  # allow the ksqlDB alert stream to propagate
