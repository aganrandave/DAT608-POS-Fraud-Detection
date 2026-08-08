"""Renders terminal locations on a map, colored by whether they have open alerts."""
import pandas as pd
import streamlit as st


def render_terminal_map(terminals: list[dict], alerts: list[dict]) -> None:
    flagged_terminal_ids = {a["terminal_id"] for a in alerts}

    df = pd.DataFrame(terminals)
    if df.empty:
        st.info("No terminal reference data available.")
        return

    df["flagged"] = df["terminal_id"].isin(flagged_terminal_ids)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    st.map(df[["lat", "lon"]])
    st.caption(f"{df['flagged'].sum()} of {len(df)} terminals have at least one open alert.")
