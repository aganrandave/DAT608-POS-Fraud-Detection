-- Source stream over the raw transaction topic produced by producer/kafka_producer.py
CREATE STREAM IF NOT EXISTS transactions_stream (
    transaction_id  VARCHAR KEY,
    terminal_id     VARCHAR,
    merchant_id     VARCHAR,
    card_bin        VARCHAR,
    amount_ngn      DOUBLE,
    state           VARCHAR,
    lga             VARCHAR,
    latitude        DOUBLE,
    longitude       DOUBLE,
    ts              VARCHAR,
    is_fraud        BOOLEAN,
    fraud_type      VARCHAR
) WITH (
    KAFKA_TOPIC = 'pos-transactions',
    VALUE_FORMAT = 'JSON'
);
