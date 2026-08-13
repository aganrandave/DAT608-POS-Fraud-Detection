"""Live alerts table, refreshed from data/alerts.xlsx."""
import time

import pandas as pd
import streamlit as st

from components.filters import filter_alerts, render_filters
from components.metrics_panel import render_metrics_panel
from components.tier_colors import tier_hex
from db import get_alerts, get_terminals

REFRESH_SECONDS = 5

st.title("Live Alerts")

auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5s)", value=True, key="live_alerts_auto_refresh")

alerts = get_alerts()
terminals = get_terminals()

selected_tiers, selected_states, selected_range = render_filters(alerts, key_prefix="live_alerts")
filtered = filter_alerts(alerts, tiers=selected_tiers, states=selected_states, date_range=selected_range)

render_metrics_panel(filtered, terminals)

if not filtered:
    st.info("No alerts match the current filters.")
else:
    df = pd.DataFrame(filtered).sort_values("created_at", ascending=False)
    styled = df.style.apply(
        lambda row: [f"background-color: {tier_hex(row['alert_tier'])}33"] * len(row), axis=1
    )
    st.dataframe(styled, use_container_width=True)

if auto_refresh:
    # Simple polling refresh: block for REFRESH_SECONDS then force a rerun,
    # which re-reads data/alerts.xlsx from scratch via get_alerts() above -
    # so a new alert written to the workbook while this page is open
    # appears on the next tick. Toggleable in the sidebar (default on) so
    # it doesn't fight with a human trying to click a filter mid-cycle.
    time.sleep(REFRESH_SECONDS)
    st.rerun()
