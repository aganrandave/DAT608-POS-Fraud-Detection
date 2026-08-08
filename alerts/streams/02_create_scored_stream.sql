-- Stream over the fraud_probability topic published by scoring/main.py
CREATE STREAM IF NOT EXISTS scored_stream (
    transaction_id          VARCHAR KEY,
    fraud_probability       DOUBLE,
    alert_tier              VARCHAR,
    xgboost_score           DOUBLE,
    isolation_forest_score  DOUBLE,
    model_version           VARCHAR,
    scored_at               VARCHAR
) WITH (
    KAFKA_TOPIC = 'pos-fraud-scores',
    VALUE_FORMAT = 'JSON'
);
