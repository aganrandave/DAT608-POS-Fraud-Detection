"""Top-of-page KPI metrics: alert counts by tier and total fraud probability distribution."""
import streamlit as st


def render_metrics_panel(alerts: list[dict]) -> None:
    total = len(alerts)
    critical = sum(1 for a in alerts if a["alert_tier"] == "critical")
    high = sum(1 for a in alerts if a["alert_tier"] == "high")
    medium = sum(1 for a in alerts if a["alert_tier"] == "medium")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total alerts", total)
    col2.metric("Critical", critical)
    col3.metric("High", high)
    col4.metric("Medium", medium)
