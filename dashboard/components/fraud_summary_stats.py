"""Pure aggregation helpers for the Fraud Summary page.

No streamlit import here (deliberately, same reasoning as
components/filters.py and components/metrics_panel.py) - these stay
importable and unit-testable without a Streamlit runtime.
"""
from datetime import datetime

TIER_ORDER = ["critical", "high", "medium", "low"]


def tier_counts(alerts: list[dict]) -> dict[str, int]:
    """Count of alerts per tier - always includes all four tiers, 0 if none."""
    counts = {tier: 0 for tier in TIER_ORDER}
    for a in alerts:
        tier = a.get("alert_tier")
        if tier in counts:
            counts[tier] += 1
    return counts


def top_n_terminals_by_alert_count(alerts: list[dict], n: int = 5) -> list[dict]:
    counts: dict[str, int] = {}
    for a in alerts:
        tid = a["terminal_id"]
        counts[tid] = counts.get(tid, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"terminal_id": tid, "alert_count": c} for tid, c in ranked]


def top_n_card_bins_by_avg_probability(alerts: list[dict], n: int = 5) -> list[dict]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for a in alerts:
        bin_ = a.get("card_bin")
        if not bin_:
            continue
        sums[bin_] = sums.get(bin_, 0.0) + (a.get("fraud_probability") or 0.0)
        counts[bin_] = counts.get(bin_, 0) + 1

    averages = [{"card_bin": b, "avg_fraud_probability": sums[b] / counts[b]} for b in sums]
    return sorted(averages, key=lambda r: r["avg_fraud_probability"], reverse=True)[:n]


def top_n_states_by_alert_count(alerts: list[dict], n: int = 5) -> list[dict]:
    counts: dict[str, int] = {}
    for a in alerts:
        state = a.get("state")
        if not state:
            continue
        counts[state] = counts.get(state, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"state": s, "alert_count": c} for s, c in ranked]


def alerts_by_hour(alerts: list[dict]) -> list[dict]:
    """One row per hour 0-23 (UTC, from created_at), count of alerts in
    that hour, always all 24 hours present in order - ready to feed
    straight into a bar chart without any further reindexing."""
    counts = {h: 0 for h in range(24)}
    for a in alerts:
        created_at = a.get("created_at")
        if not created_at:
            continue
        hour = datetime.fromisoformat(str(created_at)).hour
        counts[hour] += 1
    return [{"hour": h, "alert_count": counts[h]} for h in range(24)]
