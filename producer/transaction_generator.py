"""Synthetic POS transaction generator for Nigerian card-present payments.

Generates realistic transaction records using the reference distributions
in nibss_distributions.py, occasionally injecting labeled fraud scenarios
from fraud_scenarios.py. Records can be streamed to Kafka (kafka_producer.py)
and/or persisted to data/transactions_raw.xlsx (excel_writer.py).

Terminal identity: each terminal_id is pinned to one fixed
merchant/state/lga/lat/lon in TERMINAL_REGISTRY, built once at import time.
Earlier versions sampled a fresh location per transaction regardless of
terminal_id, so a "terminal" could jump between Lagos and Kano
transaction-to-transaction even for ordinary legitimate traffic - this made
geo_jump_km pure noise with no correlation to is_fraud, since real POS
terminals don't move. Pinning gives legitimate same-terminal traffic a
geo_jump_km of 0, so fraud that actually relocates a terminal (see
fraud_scenarios.apply_agent_collusion) has a real baseline to violate.
"""
import random
import uuid
from datetime import datetime, timedelta, timezone

import config
from fraud_scenarios import apply_cloned_card, inject_fraud
from nibss_distributions import (
    AMOUNT_BANDS,
    CARD_BINS,
    STATE_CENTROIDS,
    STATE_LGAS,
    STATE_WEIGHTS,
)

random.seed(config.RANDOM_SEED)

N_TERMINALS = 500
N_MERCHANTS = 200


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


def _build_terminal_registry() -> dict[str, dict]:
    """One fixed location and merchant per terminal_id, built once at
    import time (deterministic given config.RANDOM_SEED)."""
    registry = {}
    for i in range(1, N_TERMINALS + 1):
        terminal_id = f"TRM{i:05d}"
        state, lga, lat, lon = _sample_location()
        registry[terminal_id] = {
            "merchant_id": f"MRC{random.randint(1, N_MERCHANTS):04d}",
            "state": state,
            "lga": lga,
            "latitude": lat,
            "longitude": lon,
        }
    return registry


TERMINAL_REGISTRY = _build_terminal_registry()


def generate_transaction(
    terminal_id: str | None = None, merchant_id: str | None = None, timestamp: datetime | None = None
) -> dict:
    """Generate a single synthetic, mostly-legitimate transaction record."""
    terminal_id = terminal_id or random.choice(list(TERMINAL_REGISTRY.keys()))
    profile = TERMINAL_REGISTRY.get(terminal_id)
    if profile is not None:
        state, lga, lat, lon = profile["state"], profile["lga"], profile["latitude"], profile["longitude"]
        merchant_id = merchant_id or profile["merchant_id"]
    else:
        # An explicit terminal_id outside the registry (e.g. ad hoc test
        # data) falls back to a one-off independent location.
        state, lga, lat, lon = _sample_location()
        merchant_id = merchant_id or f"MRC{random.randint(1, N_MERCHANTS):04d}"

    card_bin = random.choice(list(CARD_BINS.keys()))

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "terminal_id": terminal_id,
        "merchant_id": merchant_id,
        "card_bin": card_bin,
        "amount_ngn": _sample_amount(),
        "state": state,
        "lga": lga,
        "latitude": lat,
        "longitude": lon,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "is_fraud": False,
        "fraud_type": "",
    }

    if random.random() < config.FRAUD_RATE:
        transaction = inject_fraud(transaction)

    return transaction


def _generate_cloned_card_burst(seed_txn: dict, seed_timestamp: datetime) -> list[dict]:
    """Cloned-card fraud is rarely a single swipe: the stolen card is
    typically tested with several rapid transactions on the same terminal,
    reusing the same card_bin, within minutes. This is what should make
    velocity_1h and bin_spend_rate spike for these rows - without it, a
    single scattered cloned_card row is statistically indistinguishable
    from ordinary noise in a trailing-window feature."""
    burst = []
    for _ in range(random.randint(2, 4)):
        burst_timestamp = seed_timestamp + timedelta(seconds=random.uniform(30, 600))
        follow_up = generate_transaction(terminal_id=seed_txn["terminal_id"], timestamp=burst_timestamp)
        follow_up["card_bin"] = seed_txn["card_bin"]
        follow_up = apply_cloned_card(follow_up)
        burst.append(follow_up)
    return burst


def generate_batch(n: int, spread_days: int = 60) -> list[dict]:
    """Generate n transactions (plus any cloned_card burst follow-ups, so
    the returned count is trimmed back to exactly n) with timestamps spread
    uniformly at random across the trailing `spread_days` days ending now -
    not the real wall-clock seconds this function actually takes to run.
    Without this, velocity_1h/bin_spend_rate/terminal_reversal_count's
    trailing windows would include nearly the entire batch regardless of
    terminal, since a bulk run finishes in seconds. Live streaming
    (kafka_producer.py) does not use this path - it timestamps each
    transaction at the real moment it's produced, which is already
    realistically paced."""
    now = datetime.now(timezone.utc)
    transactions: list[dict] = []
    while len(transactions) < n:
        timestamp = now - timedelta(seconds=random.uniform(0, spread_days * 86400))
        txn = generate_transaction(timestamp=timestamp)
        transactions.append(txn)
        if txn["fraud_type"] == "cloned_card":
            transactions.extend(_generate_cloned_card_burst(txn, timestamp))

    transactions = transactions[:n]
    transactions.sort(key=lambda t: t["timestamp"])
    return transactions


if __name__ == "__main__":
    for txn in generate_batch(5):
        print(txn)
