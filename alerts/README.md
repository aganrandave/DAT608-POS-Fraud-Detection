# alerts

ksqlDB alert logic that joins scored transactions back to their source
transaction and reference data, filters to alertable tiers, and persists
the result.

- `streams/01_create_source_stream.sql` — stream over `pos-transactions`.
- `streams/02_create_scored_stream.sql` — stream over `pos-fraud-scores` (published by `scoring/main.py`).
- `streams/03_create_alert_stream.sql` — joins the two streams plus reference tables, filters to `medium`/`high`/`critical`, writes to `pos-fraud-alerts`.
- `tables/terminal_reference.sql`, `tables/merchant_reference.sql` — ksqlDB tables bootstrapped from `data/reference/terminals.xlsx` and `data/reference/merchants.xlsx`.
- `alert_tiers.py` — tier thresholds shared with `scoring/scorer.py`; see [docs/alert_logic.md](../docs/alert_logic.md).
- `excel_writer.py` — consumes `pos-fraud-alerts` and appends each alert to `data/alerts.xlsx`.

## Apply the ksqlDB statements

```bash
ksql http://localhost:8088 <<'EOF'
RUN SCRIPT 'alerts/tables/terminal_reference.sql';
RUN SCRIPT 'alerts/tables/merchant_reference.sql';
RUN SCRIPT 'alerts/streams/01_create_source_stream.sql';
RUN SCRIPT 'alerts/streams/02_create_scored_stream.sql';
RUN SCRIPT 'alerts/streams/03_create_alert_stream.sql';
EOF
```
