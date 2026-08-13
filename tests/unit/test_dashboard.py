import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))

from components.filters import filter_alerts  # noqa: E402
from components.fraud_summary_stats import (  # noqa: E402
    alerts_by_hour,
    tier_counts,
    top_n_card_bins_by_avg_probability,
    top_n_states_by_alert_count,
    top_n_terminals_by_alert_count,
)
from components.map_view import _terminal_rows  # noqa: E402
from components.metrics_panel import top_n_at_risk_terminals  # noqa: E402
from components.tier_colors import highest_severity_tier, tier_hex, tier_rgb  # noqa: E402

ALERTS = [
    {"terminal_id": "TRM1", "alert_tier": "critical", "state": "Lagos", "fraud_probability": 0.93},
    {"terminal_id": "TRM2", "alert_tier": "medium", "state": "Kano", "fraud_probability": 0.35},
    {"terminal_id": "TRM1", "alert_tier": "high", "state": "Lagos", "fraud_probability": 0.70},
    {"terminal_id": "TRM3", "alert_tier": "low", "state": "Abuja", "fraud_probability": 0.10},
]

ALERTS_WITH_DATES = [
    {"terminal_id": "TRM1", "alert_tier": "critical", "state": "Lagos", "created_at": "2026-08-01T09:00:00+00:00"},
    {"terminal_id": "TRM2", "alert_tier": "medium", "state": "Kano", "created_at": "2026-08-05T09:00:00+00:00"},
    {"terminal_id": "TRM3", "alert_tier": "low", "state": "Abuja", "created_at": "2026-08-10T09:00:00+00:00"},
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


def test_filter_alerts_by_date_range_is_inclusive():
    result = filter_alerts(ALERTS_WITH_DATES, date_range=(date(2026, 8, 1), date(2026, 8, 5)))
    assert {a["terminal_id"] for a in result} == {"TRM1", "TRM2"}


def test_filter_alerts_by_date_range_excludes_out_of_range():
    result = filter_alerts(ALERTS_WITH_DATES, date_range=(date(2026, 8, 2), date(2026, 8, 9)))
    assert {a["terminal_id"] for a in result} == {"TRM2"}


def test_filter_alerts_no_date_range_returns_all():
    assert filter_alerts(ALERTS_WITH_DATES) == ALERTS_WITH_DATES


def test_filter_alerts_combines_tier_state_and_date_range():
    result = filter_alerts(
        ALERTS_WITH_DATES,
        tiers=["critical", "medium"],
        date_range=(date(2026, 8, 1), date(2026, 8, 31)),
    )
    assert {a["terminal_id"] for a in result} == {"TRM1", "TRM2"}


def test_terminal_rows_joins_merchant_name_and_counts_alerts():
    terminals = [
        {"terminal_id": "TRM1", "merchant_id": "MRC1", "latitude": 6.5, "longitude": 3.3},
        {"terminal_id": "TRM2", "merchant_id": "MRC2", "latitude": 9.0, "longitude": 7.4},
    ]
    merchants = [{"merchant_id": "MRC1", "merchant_name": "Ikeja Fashion Emporium"}]
    alerts = [
        {"terminal_id": "TRM1", "alert_tier": "critical", "fraud_probability": 0.93},
        {"terminal_id": "TRM1", "alert_tier": "high", "fraud_probability": 0.60},
    ]

    df = _terminal_rows(terminals, alerts, merchants)
    row1 = df[df["terminal_id"] == "TRM1"].iloc[0]
    row2 = df[df["terminal_id"] == "TRM2"].iloc[0]

    assert row1["merchant_name"] == "Ikeja Fashion Emporium"
    assert row1["alert_count"] == 2
    assert row1["fraud_probability_pct"] == 93.0
    assert row1["tier_label"] == "critical"

    # TRM2 has no matching merchant row and no alerts - falls back to
    # merchant_id as the display name, and reads as unflagged.
    assert row2["merchant_name"] == "MRC2"
    assert row2["alert_count"] == 0
    assert row2["tier_label"] == "none"


FRAUD_SUMMARY_ALERTS = [
    {"terminal_id": "TRM1", "state": "Lagos", "card_bin": "539983", "alert_tier": "critical",
     "fraud_probability": 0.90, "created_at": "2026-08-08T09:15:00+00:00"},
    {"terminal_id": "TRM1", "state": "Lagos", "card_bin": "539983", "alert_tier": "high",
     "fraud_probability": 0.70, "created_at": "2026-08-08T09:45:00+00:00"},
    {"terminal_id": "TRM2", "state": "Kano", "card_bin": "440066", "alert_tier": "medium",
     "fraud_probability": 0.40, "created_at": "2026-08-08T14:00:00+00:00"},
]


def test_tier_counts_covers_all_tiers_including_zero():
    counts = tier_counts(FRAUD_SUMMARY_ALERTS)
    assert counts == {"critical": 1, "high": 1, "medium": 1, "low": 0}


def test_top_n_terminals_by_alert_count():
    top = top_n_terminals_by_alert_count(FRAUD_SUMMARY_ALERTS, n=5)
    assert top[0] == {"terminal_id": "TRM1", "alert_count": 2}
    assert top[1] == {"terminal_id": "TRM2", "alert_count": 1}


def test_top_n_card_bins_by_avg_probability():
    top = top_n_card_bins_by_avg_probability(FRAUD_SUMMARY_ALERTS, n=5)
    # 539983's average is (0.90 + 0.70) / 2 = 0.80, ranks above 440066's 0.40.
    assert top[0]["card_bin"] == "539983"
    assert top[0]["avg_fraud_probability"] == 0.80
    assert top[1]["card_bin"] == "440066"


def test_top_n_states_by_alert_count():
    top = top_n_states_by_alert_count(FRAUD_SUMMARY_ALERTS, n=5)
    assert top[0] == {"state": "Lagos", "alert_count": 2}
    assert top[1] == {"state": "Kano", "alert_count": 1}


def test_alerts_by_hour_covers_all_24_hours_in_order():
    hourly = alerts_by_hour(FRAUD_SUMMARY_ALERTS)
    assert len(hourly) == 24
    assert [row["hour"] for row in hourly] == list(range(24))
    assert hourly[9]["alert_count"] == 2  # two alerts created at 09:xx
    assert hourly[14]["alert_count"] == 1
    assert hourly[0]["alert_count"] == 0
