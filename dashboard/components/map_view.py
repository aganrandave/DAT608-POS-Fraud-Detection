"""Renders terminal locations on a map, colored by their most severe open alert tier.

Uses pydeck (via st.pydeck_chart) instead of the plain st.map used
previously - st.map cannot color individual points, which made it
impossible to tell a critical-tier terminal from a flagged-but-low-risk
one at a glance. No Mapbox token is required: pydeck falls back to its
built-in Carto basemap when none is configured.
"""
import pandas as pd
import pydeck as pdk
import streamlit as st

from components.tier_colors import TIER_HEX, UNFLAGGED_HEX, highest_severity_tier, tier_rgb

NIGERIA_VIEW = pdk.ViewState(latitude=9.0820, longitude=8.6753, zoom=5.2)


def _terminal_rows(terminals: list[dict], alerts: list[dict]) -> pd.DataFrame:
    tiers_by_terminal: dict[str, list[str]] = {}
    for a in alerts:
        tiers_by_terminal.setdefault(a["terminal_id"], []).append(a["alert_tier"])

    df = pd.DataFrame(terminals)
    if df.empty:
        return df

    df["tier"] = df["terminal_id"].map(lambda t: highest_severity_tier(tiers_by_terminal.get(t, [])))
    df["flagged"] = df["tier"].notna()
    df["color"] = df["tier"].apply(tier_rgb)
    df["tier_label"] = df["tier"].fillna("none")
    return df.rename(columns={"latitude": "lat", "longitude": "lon"})


def render_terminal_map(terminals: list[dict], alerts: list[dict]) -> None:
    df = _terminal_rows(terminals, alerts)
    if df.empty:
        st.info("No terminal reference data available.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=8000,
        pickable=True,
        opacity=0.8,
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(
        latitude=float(df["lat"].mean()) if len(df) else NIGERIA_VIEW.latitude,
        longitude=float(df["lon"].mean()) if len(df) else NIGERIA_VIEW.longitude,
        zoom=5.5,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{terminal_name}\n{state}, {lga}\nAlert tier: {tier_label}"},
        )
    )

    legend_items = " &nbsp;&nbsp; ".join(
        f'<span style="color:{color}">●</span> {tier.capitalize()}'
        for tier, color in {**TIER_HEX, "none": UNFLAGGED_HEX}.items()
    )
    st.markdown(legend_items, unsafe_allow_html=True)
    st.caption(f"{int(df['flagged'].sum())} of {len(df)} terminals have at least one open alert.")
