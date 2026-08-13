"""Renders terminal locations on a map, colored by their most severe open alert tier.

Uses pydeck (via st.pydeck_chart) instead of the plain st.map used
previously - st.map cannot color individual points, which made it
impossible to tell a critical-tier terminal from a flagged-but-low-risk
one at a glance. No Mapbox token is required: pydeck falls back to its
built-in Carto basemap when none is configured.

pydeck and streamlit are imported inside render_terminal_map, not at
module level, so _terminal_rows (the pure data-shaping logic) stays
importable and unit-testable without either installed - same reasoning
as components/filters.py and components/metrics_panel.py.
"""
import pandas as pd

from components.tier_colors import TIER_HEX, UNFLAGGED_HEX, highest_severity_tier, tier_rgb

NIGERIA_CENTER = (9.0820, 8.6753)  # (latitude, longitude) - plain tuple, not a pdk object,
# so this stays available without importing pydeck at module level.


def _terminal_rows(terminals: list[dict], alerts: list[dict], merchants: list[dict] | None = None) -> pd.DataFrame:
    tiers_by_terminal: dict[str, list[str]] = {}
    max_prob_by_terminal: dict[str, float] = {}
    count_by_terminal: dict[str, int] = {}
    for a in alerts:
        tid = a["terminal_id"]
        tiers_by_terminal.setdefault(tid, []).append(a["alert_tier"])
        count_by_terminal[tid] = count_by_terminal.get(tid, 0) + 1
        prob = a.get("fraud_probability") or 0.0
        max_prob_by_terminal[tid] = max(max_prob_by_terminal.get(tid, 0.0), prob)

    merchant_names = {m["merchant_id"]: m["merchant_name"] for m in (merchants or [])}

    df = pd.DataFrame(terminals)
    if df.empty:
        return df

    df["tier"] = df["terminal_id"].map(lambda t: highest_severity_tier(tiers_by_terminal.get(t, [])))
    df["flagged"] = df["tier"].notna()
    df["color"] = df["tier"].apply(tier_rgb)
    df["tier_label"] = df["tier"].fillna("none")
    df["merchant_name"] = df["merchant_id"].map(merchant_names).fillna(df["merchant_id"])
    df["alert_count"] = df["terminal_id"].map(count_by_terminal).fillna(0).astype(int)
    df["fraud_probability_pct"] = (df["terminal_id"].map(max_prob_by_terminal).fillna(0.0) * 100).round(1)
    return df.rename(columns={"latitude": "lat", "longitude": "lon"})


def render_terminal_map(terminals: list[dict], alerts: list[dict], merchants: list[dict] | None = None) -> None:
    import pydeck as pdk
    import streamlit as st

    df = _terminal_rows(terminals, alerts, merchants)
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

    default_lat, default_lon = NIGERIA_CENTER
    view_state = pdk.ViewState(
        latitude=float(df["lat"].mean()) if len(df) else default_lat,
        longitude=float(df["lon"].mean()) if len(df) else default_lon,
        zoom=5.5,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": "{terminal_id} - {merchant_name}\n"
                "Fraud probability: {fraud_probability_pct}%\n"
                "Open alerts: {alert_count} ({tier_label})"
            },
        )
    )

    legend_items = " &nbsp;&nbsp; ".join(
        f'<span style="color:{color}">●</span> {tier.capitalize()}'
        for tier, color in {**TIER_HEX, "none": UNFLAGGED_HEX}.items()
    )
    st.markdown(legend_items, unsafe_allow_html=True)
    st.caption(f"{int(df['flagged'].sum())} of {len(df)} terminals have at least one open alert.")
