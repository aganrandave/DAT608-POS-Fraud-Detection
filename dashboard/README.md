# dashboard

Streamlit dashboard for monitoring live fraud alerts.

- `app.py` — entry point and navigation.
- `pages/live_alerts.py` — real-time alert table from `data/alerts.xlsx`, filterable by tier and state, rows tinted by tier.
- `pages/terminal_map.py` — pydeck map of terminals colored by their highest open alert tier, filterable by tier and state.
- `pages/fraud_summary.py` — aggregate score distributions from `data/fraud_scores.xlsx`.
- `components/map_view.py` — pydeck-based terminal map, color-coded by tier, with a legend and tooltips.
- `components/metrics_panel.py` — KPI counts by tier plus a top-5 at-risk terminals list.
- `components/filters.py` — shared tier/state sidebar filters (`filter_alerts` is a pure function, unit tested separately from the Streamlit widgets).
- `components/tier_colors.py` — single source of truth for tier-to-color mapping, shared by the map and the table.
- `excel_reader.py` / `db.py` — read access to the Excel data store (see root README's Data store section).

## Run locally

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Run via Docker

```bash
docker-compose up dashboard
```
