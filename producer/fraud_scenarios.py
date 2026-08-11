"""Synthetic fraud scenario injectors.

Each function mutates a base (legitimate) transaction dict to look like a
specific fraud pattern, and tags it with the matching fraud_type so the
label is available for supervised model training.
"""
import random

from nibss_distributions import CARD_BINS


def apply_cloned_card(transaction: dict) -> dict:
    """Simulate a cloned card used far from the cardholder's usual terminal,
    with an amount higher than the state's typical POS ticket size.

    The rapid-fire same-terminal, same-card_bin follow-up transactions that
    make this pattern show up in velocity_1h/bin_spend_rate are generated
    by transaction_generator._generate_cloned_card_burst(), not here - this
    function only shapes a single row (used for both the seed transaction
    and each burst follow-up)."""
    transaction["amount_ngn"] = round(transaction["amount_ngn"] * random.uniform(2.5, 6.0), 2)
    transaction["is_fraud"] = True
    transaction["fraud_type"] = "cloned_card"
    return transaction


def apply_agent_collusion(transaction: dict) -> dict:
    """Simulate a POS agent colluding with a cardholder to run repeated
    round-numbered transactions just under a monitoring threshold, using a
    portable/compromised terminal away from its registered site.

    Registered terminals (TERMINAL_REGISTRY) don't move, so relocating the
    reading here by ~30-130km is what makes geo_jump_km spike for these
    rows - legitimate same-terminal traffic never triggers it."""
    transaction["amount_ngn"] = float(random.choice([49_500, 49_800, 49_900, 49_950]))
    transaction["latitude"] = round(
        transaction["latitude"] + random.uniform(0.3, 1.2) * random.choice([-1, 1]), 6
    )
    transaction["longitude"] = round(
        transaction["longitude"] + random.uniform(0.3, 1.2) * random.choice([-1, 1]), 6
    )
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
