# seed_data

Reference seed data for terminals and merchants used to bootstrap ksqlDB
tables no longer lives here as static files. It is sourced directly from
the Git-tracked Excel reference tables at [`data/reference/terminals.xlsx`](../../data/reference/terminals.xlsx)
and [`data/reference/merchants.xlsx`](../../data/reference/merchants.xlsx)
so there is a single source of truth for reference data.

`alerts/publish_reference_data.py` reads these workbooks and publishes each
row to the `terminal-reference` / `merchant-reference` Kafka topics (keyed
by `terminal_id` / `merchant_id`). It runs once at container startup, before
the long-running alert consumer, via `alerts/entrypoint.sh`.
`alerts/tables/terminal_reference.sql` and `alerts/tables/merchant_reference.sql`
then just declare the ksqlDB tables `TERMINAL_REFERENCE` and
`MERCHANT_REFERENCE` over those topics - they don't read the workbooks
themselves. If `publish_reference_data.py` hasn't run yet (or ever), the
tables register successfully but stay empty, and joins against them return
null.
