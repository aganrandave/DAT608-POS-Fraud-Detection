"""Alert tier thresholds, kept in sync with scoring/scorer.py's tier_for_probability.

Documented in docs/alert_logic.md.
"""

TIER_THRESHOLDS = {
    "critical": 0.85,
    "high": 0.60,
    "medium": 0.30,
    "low": 0.0,
}


def tier_for_probability(fraud_probability: float) -> str:
    for tier, threshold in TIER_THRESHOLDS.items():
        if fraud_probability >= threshold:
            return tier
    return "low"


def is_alertable(alert_tier: str) -> bool:
    """Only medium tier and above are surfaced as alerts (see 03_create_alert_stream.sql)."""
    return alert_tier in {"medium", "high", "critical"}
