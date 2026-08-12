"""Live alerts table, refreshed from data/alerts.xlsx."""
import pandas as pd
import streamlit as st

from components.filters import filter_alerts, render_filters
from components.metrics_panel import render_metrics_panel
from components.tier_colors import tier_hex
from db import get_alerts, get_terminals

st.title("Live Alerts")

alerts = get_alerts()
terminals = get_terminals()

selected_tiers, selected_states = render_filters(alerts, key_prefix="live_alerts")
filtered = filter_alerts(alerts, tiers=selected_tiers, states=selected_states)

render_metrics_panel(filtered, terminals)

if not filtered:
    st.info("No alerts match the current filters.")
else:
    df = pd.DataFrame(filtered).sort_values("created_at", ascending=False)
    styled = df.style.apply(
        lambda row: [f"background-color: {tier_hex(row['alert_tier'])}33"] * len(row), axis=1
    )
    st.dataframe(styled, use_container_width=True)
