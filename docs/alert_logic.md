# Alert Logic

## Tiering thresholds

Fraud probability is mapped to an alert tier by both
[`scoring/scorer.py`](../scoring/scorer.py) (`tier_for_probability`) and
[`alerts/alert_tiers.py`](../alerts/alert_tiers.py) (`TIER_THRESHOLDS`),
which are kept in sync and covered by
[`tests/unit/test_scorer.py`](../tests/unit/test_scorer.py).

| Tier | Threshold (fraud_probability >=) |
|---|---|
| critical | 0.85 |
| high | 0.60 |
| medium | 0.30 |
| low | 0.00 |

## Alertable tiers

Only `medium`, `high`, and `critical` scored transactions are surfaced as
alerts — `low` tier scores are persisted to `data/fraud_scores.xlsx` for
audit purposes but never reach `data/alerts.xlsx`. This filter is applied
in [`alerts/streams/03_create_alert_stream.sql`](../alerts/streams/03_create_alert_stream.sql)
via `WHERE s.alert_tier IN ('medium', 'high', 'critical')`.

## Pipeline

1. `scoring/main.py` computes `fraud_probability` and publishes it to the
   `pos-fraud-scores` Kafka topic.
2. ksqlDB's `scored_stream` (`alerts/streams/02_create_scored_stream.sql`)
   consumes that topic.
3. `alert_stream` (`alerts/streams/03_create_alert_stream.sql`) joins
   `scored_stream` back to `transactions_stream` and the
   `terminal_reference` / `merchant_reference` tables, filters to
   alertable tiers, and emits to `pos-fraud-alerts`.
4. `alerts/excel_writer.py` consumes `pos-fraud-alerts` and appends each
   alert to `data/alerts.xlsx`.
5. The Streamlit dashboard's `pages/live_alerts.py` reads
   `data/alerts.xlsx` for the live alert feed.
