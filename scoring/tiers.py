"""Alert tier thresholds for the scoring service.

Kept dependency-free (no mlflow import) so it can be unit tested without a
model registry. Mirrored by alerts/alert_tiers.py and covered by
tests/unit/test_scorer.py. Documented in docs/alert_logic.md.
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
