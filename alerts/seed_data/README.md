# seed_data

Reference seed data for terminals and merchants used to bootstrap ksqlDB
tables no longer lives here as static files. It is sourced directly from
the Git-tracked Excel reference tables at [`data/reference/terminals.xlsx`](../../data/reference/terminals.xlsx)
and [`data/reference/merchants.xlsx`](../../data/reference/merchants.xlsx)
so there is a single source of truth for reference data.

`alerts/tables/terminal_reference.sql` and `alerts/tables/merchant_reference.sql`
load these workbooks (via the loader in `alerts/excel_writer.py`'s sibling
reader helper) into the ksqlDB tables `TERMINAL_REFERENCE` and
`MERCHANT_REFERENCE` at startup instead of reading from this directory.
