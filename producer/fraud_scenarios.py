"""Synthetic fraud scenario injectors.

Each function mutates a base (legitimate) transaction dict to look like a
specific fraud pattern, and tags it with the matching fraud_type so the
label is available for supervised model training.
"""
import random

from nibss_distributions import CARD_BINS


def apply_cloned_card(transaction: dict) -> dict:
    """Simulate a cloned card used far from the cardholder's usual terminal,
    with an amount higher than the state's typical POS ticket size."""
    transaction["amount_ngn"] = round(transaction["amount_ngn"] * random.uniform(2.5, 6.0), 2)
    transaction["is_fraud"] = True
    transaction["fraud_type"] = "cloned_card"
    return transaction


def apply_agent_collusion(transaction: dict) -> dict:
    """Simulate a POS agent colluding with a cardholder to run repeated
    round-numbered transactions just under a monitoring threshold."""
    transaction["amount_ngn"] = float(random.choice([49_500, 49_800, 49_900, 49_950]))
    transaction["is_fraud"] = True
    transaction["fraud_type"] = "agent_collusion"
    return transaction


def apply_fake_reversal(transaction: dict) -> dict:
    """Simulate a fraudulent reversal claim on a completed sale, typically
    paired with an unusual card BIN / terminal combination."""
    transaction["card_bin"] = random.choice(list(CARD_BINS.keys()))
    transaction["is_fraud"] = True
    transaction["fraud_type"] = "fake_reversal"
    return transaction


FRAUD_INJECTORS = {
    "cloned_card": apply_cloned_card,
    "agent_collusion": apply_agent_collusion,
    "fake_reversal": apply_fake_reversal,
}


def inject_fraud(transaction: dict, fraud_type: str | None = None) -> dict:
    fraud_type = fraud_type or random.choice(list(FRAUD_INJECTORS))
    return FRAUD_INJECTORS[fraud_type](transaction)
