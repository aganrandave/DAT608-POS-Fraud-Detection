-- Reference table backing merchant lookups in 03_create_alert_stream.sql.
-- Bootstrapped at startup from data/reference/merchants.xlsx, published to
-- the merchant-reference topic keyed by merchant_id, then registered here.
CREATE TABLE IF NOT EXISTS merchant_reference (
    merchant_id         VARCHAR PRIMARY KEY,
    merchant_name       VARCHAR,
    category            VARCHAR,
    state               VARCHAR,
    lga                 VARCHAR,
    registration_date   VARCHAR,
    is_flagged          BOOLEAN
) WITH (
    KAFKA_TOPIC = 'merchant-reference',
    VALUE_FORMAT = 'JSON'
);
