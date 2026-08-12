"""Top-of-page KPI metrics: alert counts by tier, plus the top-5 at-risk terminals."""
import streamlit as st

from components.tier_colors import tier_hex


def top_n_at_risk_terminals(alerts: list[dict], terminals: list[dict] | None = None, n: int = 5) -> list[dict]:
    """One row per terminal, its highest fraud_probability alert and that
    alert's tier, sorted descending by probability, top n. terminals is
    optional and only used to attach a human-readable terminal_name."""
    names_by_id = {t["terminal_id"]: t.get("terminal_name") for t in (terminals or [])}

    worst_by_terminal: dict[str, dict] = {}
    for a in alerts:
        tid = a["terminal_id"]
        current = worst_by_terminal.get(tid)
        if current is None or a["fraud_probability"] > current["fraud_probability"]:
            worst_by_terminal[tid] = a

    ranked = sorted(worst_by_terminal.values(), key=lambda a: a["fraud_probability"], reverse=True)[:n]
    return [
        {
            "terminal_id": a["terminal_id"],
            "terminal_name": names_by_id.get(a["terminal_id"], a["terminal_id"]),
            "fraud_probability": a["fraud_probability"],
            "alert_tier": a["alert_tier"],
        }
        for a in ranked
    ]


def render_metrics_panel(alerts: list[dict], terminals: list[dict] | None = None) -> None:
    total = len(alerts)
    critical = sum(1 for a in alerts if a["alert_tier"] == "critical")
    high = sum(1 for a in alerts if a["alert_tier"] == "high")
    medium = sum(1 for a in alerts if a["alert_tier"] == "medium")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total alerts", total)
    col2.metric("Critical", critical)
    col3.metric("High", high)
    col4.metric("Medium", medium)

    top5 = top_n_at_risk_terminals(alerts, terminals)
    if top5:
        st.markdown("**Top 5 at-risk terminals**")
        for row in top5:
            color = tier_hex(row["alert_tier"])
            st.markdown(
                f'<span style="color:{color}">●</span> **{row["terminal_name"]}** '
                f'({row["terminal_id"]}) - {row["fraud_probability"]:.0%} probability, {row["alert_tier"]} tier',
                unsafe_allow_html=True,
            )
