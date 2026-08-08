# pipeline

Spark Structured Streaming job that consumes raw transactions from Kafka and
produces engineered fraud-detection features.

- `schema.py` — shared Spark schemas for the raw transaction and feature streams.
- `geo.py` — dependency-free haversine distance helper, unit tested directly.
- `feature_windows.py` — windowed transforms: `velocity_1h`, `geo_jump_km`, `bin_spend_rate`, `terminal_reversal_count`.
- `spark_consumer.py` — reads `pos-transactions`, applies `feature_windows.build_features`, sinks each micro-batch via `excel_writer.write_batch`.
- `excel_writer.py` — appends feature rows to `data/features.xlsx` using the shared `openpyxl` + `filelock` pattern.

## Run locally

```bash
pip install -r pipeline/requirements.txt
spark-submit pipeline/spark_consumer.py
```

## Run via Docker

```bash
docker-compose up pipeline
```
