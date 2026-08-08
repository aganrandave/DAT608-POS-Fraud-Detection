"""Geographic view of terminals and their alert status."""
import streamlit as st

from components.map_view import render_terminal_map
from db import get_alerts, get_terminals

st.title("Terminal Map")

terminals = get_terminals()
alerts = get_alerts()
render_terminal_map(terminals, alerts)
