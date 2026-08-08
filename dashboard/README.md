# dashboard

Streamlit dashboard for monitoring live fraud alerts.

- `app.py` — entry point and navigation.
- `pages/live_alerts.py` — real-time alert table from `data/alerts.xlsx`.
- `pages/terminal_map.py` — map of terminals colored by alert status.
- `pages/fraud_summary.py` — aggregate score distributions from `data/fraud_scores.xlsx`.
- `components/map_view.py`, `components/metrics_panel.py` — reusable render helpers.
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
