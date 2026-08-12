"""Reusable alert filters: tier and state, shared by Live Alerts and Terminal Map.

Split into a pure filtering function (filter_alerts, easy to unit test) and
a Streamlit sidebar renderer (render_filters) that collects the widget
selections and calls it - keeps the filtering logic testable without a
Streamlit runtime.
"""
import streamlit as st

from components.tier_colors import TIER_ORDER


def filter_alerts(
    alerts: list[dict], tiers: list[str] | None = None, states: list[str] | None = None
) -> list[dict]:
    """Return only alerts whose alert_tier is in tiers and whose state is in
    states. None (or an empty list) for either means "no filter on that field"."""
    result = alerts
    if tiers:
        result = [a for a in result if a.get("alert_tier") in tiers]
    if states:
        result = [a for a in result if a.get("state") in states]
    return result


def render_filters(alerts: list[dict], key_prefix: str = "") -> tuple[list[str], list[str]]:
    """Renders tier and state multiselects in the sidebar, defaulting to
    "all selected" (no filtering) so pages behave the same as before this
    feature was added until the user actually narrows something down.
    key_prefix disambiguates widget keys when the same filters are rendered
    on more than one page in the same session."""
    available_tiers = [t for t in TIER_ORDER if t in {a.get("alert_tier") for a in alerts}]
    available_states = sorted({a.get("state") for a in alerts if a.get("state")})

    st.sidebar.subheader("Filters")
    selected_tiers = st.sidebar.multiselect(
        "Alert tier", options=available_tiers, default=available_tiers, key=f"{key_prefix}_tier_filter"
    )
    selected_states = st.sidebar.multiselect(
        "State", options=available_states, default=available_states, key=f"{key_prefix}_state_filter"
    )
    return selected_tiers, selected_states
