"""Geographic view of terminals and their alert status."""
import streamlit as st

from components.filters import filter_alerts, render_filters
from components.map_view import render_terminal_map
from db import get_alerts, get_merchants, get_terminals

st.title("Terminal Map")

terminals = get_terminals()
merchants = get_merchants()
alerts = get_alerts()

selected_tiers, selected_states, selected_range = render_filters(alerts, key_prefix="terminal_map")
filtered = filter_alerts(alerts, tiers=selected_tiers, states=selected_states, date_range=selected_range)

render_terminal_map(terminals, filtered, merchants)
