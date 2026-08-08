-- Joins the scored stream back to the source transaction stream and the
-- terminal/merchant reference tables, emitting only medium-tier-or-above
-- alerts to the pos-fraud-alerts topic consumed by alerts/excel_writer.py.
CREATE STREAM IF NOT EXISTS alert_stream
    WITH (KAFKA_TOPIC = 'pos-fraud-alerts', VALUE_FORMAT = 'JSON') AS
SELECT
    s.transaction_id                       AS transaction_id,
    t.terminal_id                          AS terminal_id,
    t.merchant_id                          AS merchant_id,
    t.card_bin                             AS card_bin,
    s.fraud_probability                    AS fraud_probability,
    s.alert_tier                           AS alert_tier,
    t.state                                AS state,
    tr.terminal_name                       AS terminal_name,
    mr.merchant_name                       AS merchant_name
FROM scored_stream s
INNER JOIN transactions_stream t
    WITHIN 1 HOURS
    ON s.transaction_id = t.transaction_id
LEFT JOIN terminal_reference tr
    ON t.terminal_id = tr.terminal_id
LEFT JOIN merchant_reference mr
    ON t.merchant_id = mr.merchant_id
WHERE s.alert_tier IN ('medium', 'high', 'critical')
EMIT CHANGES;
