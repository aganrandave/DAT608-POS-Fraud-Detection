"""Windowed feature engineering transforms applied to the transaction stream.

Each function takes and returns a Spark DataFrame so they can be composed
in spark_consumer.py's processing pipeline.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from geo import haversine_km

haversine_km_udf = F.udf(haversine_km, "double")


def with_velocity_1h(df: DataFrame) -> DataFrame:
    """Count of transactions per terminal within a trailing 1-hour window."""
    window_spec = (
        Window.partitionBy("terminal_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-3600, 0)
    )
    return df.withColumn("velocity_1h", F.count("transaction_id").over(window_spec))


def with_geo_jump(df: DataFrame) -> DataFrame:
    """Distance in km between a terminal's current and previous transaction."""
    window_spec = Window.partitionBy("terminal_id").orderBy("timestamp")
    prev_lat = F.lag("latitude").over(window_spec)
    prev_lon = F.lag("longitude").over(window_spec)
    return df.withColumn(
        "geo_jump_km",
        F.when(
            prev_lat.isNotNull(),
            haversine_km_udf(prev_lat, prev_lon, F.col("latitude"), F.col("longitude")),
        ).otherwise(F.lit(0.0)),
    )


def with_bin_spend_rate(df: DataFrame) -> DataFrame:
    """Rolling average spend per card BIN over a trailing 1-hour window."""
    window_spec = (
        Window.partitionBy("card_bin")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-3600, 0)
    )
    return df.withColumn("bin_spend_rate", F.avg("amount_ngn").over(window_spec))


def with_terminal_reversal_count(df: DataFrame) -> DataFrame:
    """Count of fake_reversal-flagged transactions per terminal in a trailing 24h window."""
    window_spec = (
        Window.partitionBy("terminal_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-86400, 0)
    )
    is_reversal = F.when(F.col("fraud_type") == "fake_reversal", 1).otherwise(0)
    return df.withColumn("terminal_reversal_count", F.sum(is_reversal).over(window_spec))


def build_features(df: DataFrame) -> DataFrame:
    df = with_velocity_1h(df)
    df = with_geo_jump(df)
    df = with_bin_spend_rate(df)
    df = with_terminal_reversal_count(df)
    return df.select(
        "transaction_id",
        "terminal_id",
        "card_bin",
        "velocity_1h",
        "geo_jump_km",
        "bin_spend_rate",
        "terminal_reversal_count",
        "timestamp",
    )
