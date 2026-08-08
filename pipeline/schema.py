"""Spark schemas shared across the streaming feature-engineering pipeline."""
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), False),
        StructField("terminal_id", StringType(), False),
        StructField("merchant_id", StringType(), False),
        StructField("card_bin", StringType(), False),
        StructField("amount_ngn", DoubleType(), False),
        StructField("state", StringType(), True),
        StructField("lga", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("timestamp", TimestampType(), False),
        StructField("is_fraud", BooleanType(), True),
        StructField("fraud_type", StringType(), True),
    ]
)

FEATURE_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), False),
        StructField("terminal_id", StringType(), False),
        StructField("card_bin", StringType(), False),
        StructField("velocity_1h", DoubleType(), True),
        StructField("geo_jump_km", DoubleType(), True),
        StructField("bin_spend_rate", DoubleType(), True),
        StructField("terminal_reversal_count", DoubleType(), True),
        StructField("timestamp", TimestampType(), False),
    ]
)
