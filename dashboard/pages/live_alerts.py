"""Live alerts table, refreshed from data/alerts.xlsx."""
import pandas as pd
import streamlit as st

from components.metrics_panel import render_metrics_panel
from db import get_alerts

st.title("Live Alerts")

alerts = get_alerts()
render_metrics_panel(alerts)

df = pd.DataFrame(alerts).sort_values("created_at", ascending=False)
st.dataframe(df, use_container_width=True)
