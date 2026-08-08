import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "producer"))

from fraud_scenarios import inject_fraud  # noqa: E402
from nibss_distributions import CARD_BINS, STATE_WEIGHTS  # noqa: E402
from transaction_generator import generate_transaction  # noqa: E402


def test_generate_transaction_has_required_fields():
    txn = generate_transaction()
    required = {
        "transaction_id",
        "terminal_id",
        "merchant_id",
        "card_bin",
        "amount_ngn",
        "state",
        "lga",
        "latitude",
        "longitude",
        "timestamp",
        "is_fraud",
        "fraud_type",
    }
    assert required.issubset(txn.keys())


def test_generate_transaction_uses_known_state():
    txn = generate_transaction()
    assert txn["state"] in STATE_WEIGHTS


def test_generate_transaction_uses_known_card_bin():
    txn = generate_transaction()
    assert txn["card_bin"] in CARD_BINS


def test_inject_fraud_marks_transaction_as_fraud():
    txn = generate_transaction()
    fraud_txn = inject_fraud(txn, fraud_type="cloned_card")
    assert fraud_txn["is_fraud"] is True
    assert fraud_txn["fraud_type"] == "cloned_card"


def test_amount_is_positive():
    txn = generate_transaction()
    assert txn["amount_ngn"] > 0
