"""Reusable alert filters: tier, state, and date range, shared by Live
Alerts and Terminal Map.

Split into a pure filtering function (filter_alerts, easy to unit test) and
a Streamlit sidebar renderer (render_filters) that collects the widget
selections and calls it - keeps the filtering logic testable without a
Streamlit runtime. streamlit is imported inside render_filters, not at
module level, so importing filter_alerts alone (as tests/unit/test_dashboard.py
does) never requires streamlit to be installed.
"""
from datetime import date, datetime

from components.tier_colors import TIER_ORDER


def _alert_date(alert: dict) -> date | None:
    created_at = alert.get("created_at")
    if not created_at:
        return None
    return datetime.fromisoformat(str(created_at)).date()


def filter_alerts(
    alerts: list[dict],
    tiers: list[str] | None = None,
    states: list[str] | None = None,
    date_range: tuple[date, date] | None = None,
) -> list[dict]:
    """Return only alerts whose alert_tier is in tiers, whose state is in
    states, and whose created_at date falls within date_range (inclusive).
    None (or an empty list, for tiers/states) means "no filter on that
    field"."""
    result = alerts
    if tiers:
        result = [a for a in result if a.get("alert_tier") in tiers]
    if states:
        result = [a for a in result if a.get("state") in states]
    if date_range:
        start, end = date_range
        result = [a for a in result if (d := _alert_date(a)) is not None and start <= d <= end]
    return result


def render_filters(
    alerts: list[dict], key_prefix: str = ""
) -> tuple[list[str], list[str], tuple[date, date] | None]:
    """Renders tier, state, and date-range controls in the sidebar,
    defaulting to "all selected" / the full observed date range (no
    filtering) so pages behave the same as before this feature was added
    until the user actually narrows something down. key_prefix
    disambiguates widget keys when the same filters are rendered on more
    than one page in the same session."""
    import streamlit as st

    available_tiers = [t for t in TIER_ORDER if t in {a.get("alert_tier") for a in alerts}]
    available_states = sorted({a.get("state") for a in alerts if a.get("state")})
    alert_dates = [d for a in alerts if (d := _alert_date(a)) is not None]

    st.sidebar.subheader("Filters")
    selected_tiers = st.sidebar.multiselect(
        "Alert tier", options=available_tiers, default=available_tiers, key=f"{key_prefix}_tier_filter"
    )
    selected_states = st.sidebar.multiselect(
        "State", options=available_states, default=available_states, key=f"{key_prefix}_state_filter"
    )

    selected_range: tuple[date, date] | None = None
    if alert_dates:
        min_date, max_date = min(alert_dates), max(alert_dates)
        picked = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_date_filter",
        )
        # date_input returns a single date while the user has only picked
        # one endpoint - only apply the filter once both ends are chosen.
        if isinstance(picked, tuple) and len(picked) == 2:
            selected_range = picked

    return selected_tiers, selected_states, selected_range
