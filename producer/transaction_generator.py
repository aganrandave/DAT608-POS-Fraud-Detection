"""Synthetic POS transaction generator for Nigerian card-present payments.

Generates realistic transaction records using the reference distributions
in nibss_distributions.py, occasionally injecting labeled fraud scenarios
from fraud_scenarios.py. Records can be streamed to Kafka (kafka_producer.py)
and/or persisted to data/transactions_raw.xlsx (excel_writer.py).
"""
import random
import uuid
from datetime import datetime, timezone

import config
from fraud_scenarios import inject_fraud
from nibss_distributions import (
    AMOUNT_BANDS,
    CARD_BINS,
    STATE_CENTROIDS,
    STATE_LGAS,
    STATE_WEIGHTS,
)

random.seed(config.RANDOM_SEED)


def _sample_amount() -> float:
    band = random.choices(
        AMOUNT_BANDS,
        weights=[w for *_, w in AMOUNT_BANDS],
        k=1,
    )[0]
    low, high, _ = band
    return round(random.uniform(low, high), 2)


def _sample_location() -> tuple[str, str, float, float]:
    state = random.choices(list(STATE_WEIGHTS), weights=list(STATE_WEIGHTS.values()), k=1)[0]
    lga = random.choice(STATE_LGAS[state])
    base_lat, base_lon = STATE_CENTROIDS[state]
    lat = round(base_lat + random.uniform(-0.15, 0.15), 6)
    lon = round(base_lon + random.uniform(-0.15, 0.15), 6)
    return state, lga, lat, lon


def generate_transaction(terminal_id: str | None = None, merchant_id: str | None = None) -> dict:
    """Generate a single synthetic, mostly-legitimate transaction record."""
    state, lga, lat, lon = _sample_location()
    card_bin = random.choice(list(CARD_BINS.keys()))

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "terminal_id": terminal_id or f"TRM{random.randint(1, 500):05d}",
        "merchant_id": merchant_id or f"MRC{random.randint(1, 200):04d}",
        "card_bin": card_bin,
        "amount_ngn": _sample_amount(),
        "state": state,
        "lga": lga,
        "latitude": lat,
        "longitude": lon,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_fraud": False,
        "fraud_type": "",
    }

    if random.random() < config.FRAUD_RATE:
        transaction = inject_fraud(transaction)

    return transaction


def generate_batch(n: int) -> list[dict]:
    return [generate_transaction() for _ in range(n)]


if __name__ == "__main__":
    for txn in generate_batch(5):
        print(txn)
