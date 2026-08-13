"""Fraud summary page: today's tier split, top-5 breakdowns by terminal,
card BIN, and state, and an hourly alert distribution - sourced from
data/alerts.xlsx, plus the overall scored-transaction distribution from
data/fraud_scores.xlsx.
"""
import pandas as pd
import streamlit as st

from components.fraud_summary_stats import (
    alerts_by_hour,
    tier_counts,
    top_n_card_bins_by_avg_probability,
    top_n_states_by_alert_count,
    top_n_terminals_by_alert_count,
)
from db import get_alerts, get_fraud_scores

st.title("Fraud Summary")

alerts = get_alerts()
scores = get_fraud_scores()

counts = tier_counts(alerts)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total alerts", len(alerts))
col2.metric("Critical", counts["critical"])
col3.metric("High", counts["high"])
col4.metric("Medium", counts["medium"])

if not alerts:
    st.info("No alerts yet.")
else:
    left, right = st.columns(2)

    with left:
        st.subheader("Top 5 terminals by alert count")
        st.dataframe(
            pd.DataFrame(top_n_terminals_by_alert_count(alerts)), use_container_width=True, hide_index=True
        )

        st.subheader("Top 5 states by alert count")
        st.dataframe(
            pd.DataFrame(top_n_states_by_alert_count(alerts)), use_container_width=True, hide_index=True
        )

    with right:
        st.subheader("Top 5 card BINs by avg. fraud probability")
        bins_df = pd.DataFrame(top_n_card_bins_by_avg_probability(alerts))
        if not bins_df.empty:
            bins_df["avg_fraud_probability"] = (bins_df["avg_fraud_probability"] * 100).round(1)
            bins_df = bins_df.rename(columns={"avg_fraud_probability": "avg_fraud_probability_pct"})
        st.dataframe(bins_df, use_container_width=True, hide_index=True)

    st.subheader("Alerts by hour")
    hourly_df = pd.DataFrame(alerts_by_hour(alerts)).set_index("hour")
    st.bar_chart(hourly_df["alert_count"])

st.subheader("Fraud score distribution (all scored transactions)")
scores_df = pd.DataFrame(scores)
if scores_df.empty:
    st.info("No scored transactions yet.")
else:
    st.bar_chart(scores_df["fraud_probability"])
