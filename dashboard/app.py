"""Streamlit dashboard entry point: landing page + navigation to sub-pages."""
import streamlit as st

st.set_page_config(page_title="POS Fraud Detection", layout="wide")

st.title("POS Agent Fraud Detection")
st.markdown(
    """
    Use the sidebar to navigate:

    - **Live Alerts** — real-time alert feed from `data/alerts.xlsx`
    - **Terminal Map** — geographic view of flagged terminals
    - **Fraud Summary** — aggregate score distributions from `data/fraud_scores.xlsx`
    """
)
