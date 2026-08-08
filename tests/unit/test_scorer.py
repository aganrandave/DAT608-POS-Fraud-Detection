import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scoring"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alerts"))

from alert_tiers import is_alertable  # noqa: E402
from alert_tiers import tier_for_probability as alerts_tier_for_probability  # noqa: E402
from tiers import tier_for_probability as scoring_tier_for_probability  # noqa: E402


def test_tier_thresholds_are_consistent_between_scoring_and_alerts():
    for probability in [0.0, 0.1, 0.29, 0.3, 0.59, 0.6, 0.84, 0.85, 0.99]:
        assert scoring_tier_for_probability(probability) == alerts_tier_for_probability(probability)


def test_critical_tier_at_high_probability():
    assert scoring_tier_for_probability(0.95) == "critical"


def test_low_tier_at_low_probability():
    assert scoring_tier_for_probability(0.05) == "low"


def test_is_alertable_excludes_low_tier():
    assert is_alertable("low") is False
    assert is_alertable("medium") is True
    assert is_alertable("critical") is True
