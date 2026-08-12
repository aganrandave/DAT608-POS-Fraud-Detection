import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))

from components.filters import filter_alerts  # noqa: E402
from components.metrics_panel import top_n_at_risk_terminals  # noqa: E402
from components.tier_colors import highest_severity_tier, tier_hex, tier_rgb  # noqa: E402

ALERTS = [
    {"terminal_id": "TRM1", "alert_tier": "critical", "state": "Lagos", "fraud_probability": 0.93},
    {"terminal_id": "TRM2", "alert_tier": "medium", "state": "Kano", "fraud_probability": 0.35},
    {"terminal_id": "TRM1", "alert_tier": "high", "state": "Lagos", "fraud_probability": 0.70},
    {"terminal_id": "TRM3", "alert_tier": "low", "state": "Abuja", "fraud_probability": 0.10},
]


def test_filter_alerts_by_tier():
    result = filter_alerts(ALERTS, tiers=["critical"])
    assert len(result) == 1
    assert result[0]["terminal_id"] == "TRM1"


def test_filter_alerts_by_state():
    result = filter_alerts(ALERTS, states=["Lagos"])
    assert len(result) == 2
    assert all(a["state"] == "Lagos" for a in result)


def test_filter_alerts_no_filter_returns_all():
    assert filter_alerts(ALERTS) == ALERTS


def test_filter_alerts_combined():
    result = filter_alerts(ALERTS, tiers=["critical", "high"], states=["Lagos"])
    assert len(result) == 2


def test_highest_severity_tier_picks_most_severe():
    assert highest_severity_tier(["medium", "critical", "low"]) == "critical"
    assert highest_severity_tier(["low"]) == "low"
    assert highest_severity_tier([]) is None


def test_tier_rgb_and_hex_cover_all_tiers():
    for tier in ["critical", "high", "medium", "low"]:
        assert len(tier_rgb(tier)) == 3
        assert tier_hex(tier).startswith("#")


def test_tier_rgb_falls_back_for_unknown_tier():
    assert tier_rgb(None) == [128, 128, 128]


def test_top_n_at_risk_terminals_ranks_by_worst_alert_per_terminal():
    top = top_n_at_risk_terminals(ALERTS, n=2)
    assert len(top) == 2
    # TRM1's worst alert is critical (0.93), which should rank first.
    assert top[0]["terminal_id"] == "TRM1"
    assert top[0]["fraud_probability"] == 0.93
    assert top[0]["alert_tier"] == "critical"


def test_top_n_at_risk_terminals_uses_terminal_name_when_available():
    terminals = [{"terminal_id": "TRM1", "terminal_name": "Ikeja Retail Counter"}]
    top = top_n_at_risk_terminals(ALERTS, terminals=terminals, n=1)
    assert top[0]["terminal_name"] == "Ikeja Retail Counter"
