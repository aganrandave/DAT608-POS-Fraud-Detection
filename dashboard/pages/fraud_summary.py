"""Aggregate fraud-score summary charts, sourced from data/fraud_scores.xlsx."""
import pandas as pd
import streamlit as st

from db import get_fraud_scores

st.title("Fraud Summary")

scores = get_fraud_scores()
df = pd.DataFrame(scores)

if df.empty:
    st.info("No scored transactions yet.")
else:
    st.subheader("Fraud probability distribution")
    st.bar_chart(df["fraud_probability"])

    st.subheader("Alerts by tier")
    st.bar_chart(df["alert_tier"].value_counts())
