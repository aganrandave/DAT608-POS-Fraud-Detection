-- Reference table backing terminal lookups in 03_create_alert_stream.sql.
-- Bootstrapped at startup from data/reference/terminals.xlsx via the loader
-- referenced in alerts/README.md, published to the terminal-reference topic
-- keyed by terminal_id, then registered here as a ksqlDB table.
CREATE TABLE IF NOT EXISTS terminal_reference (
    terminal_id     VARCHAR PRIMARY KEY,
    terminal_name   VARCHAR,
    merchant_id     VARCHAR,
    state           VARCHAR,
    lga             VARCHAR,
    latitude        DOUBLE,
    longitude       DOUBLE,
    operator        VARCHAR,
    is_active       BOOLEAN
) WITH (
    KAFKA_TOPIC = 'terminal-reference',
    VALUE_FORMAT = 'JSON'
);
