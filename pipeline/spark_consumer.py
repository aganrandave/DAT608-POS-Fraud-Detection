"""Spark Structured Streaming job: Kafka transactions -> engineered features.

Reads raw transaction JSON from the pos-transactions Kafka topic and writes
each micro-batch's windowed features to data/features.xlsx via
excel_writer.write_batch.

feature_windows.build_features() is applied inside the foreachBatch
callback, not on the streaming DataFrame itself. Structured Streaming only
supports time-window aggregation with watermarks - the arbitrary
partition/order/rangeBetween window functions build_features() uses
(velocity_1h, bin_spend_rate, terminal_reversal_count) raise
NON_TIME_WINDOW_NOT_SUPPORTED_IN_STREAMING if applied directly to a
streaming DataFrame. Each call to foreachBatch's callback receives a
static, non-streaming DataFrame for that one micro-batch, where those same
window functions are ordinary batch Spark SQL and work fine.

Known limitation from that split, not hidden: each window function only
sees rows within its own micro-batch, not the full trailing 1h/24h history
across the stream - a true unbounded trailing window would need stateful
streaming (mapGroupsWithState) or a restructure around F.window() with
watermarks, out of scope here. The offline batch path
(batch_feature_engineering.py, used for all model training and CTGAN
validation this project's models are actually built on) already computes
genuine full-history trailing windows over pandas; this live path exists
to demonstrate the streaming wiring works end to end, not to duplicate
that exact semantics online.
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from excel_writer import write_batch
from feature_windows import build_features
from schema import TRANSACTION_SCHEMA

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "pos-transactions")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints")


def build_spark_session() -> SparkSession:
    # readStream.format("kafka") needs the spark-sql-kafka connector on the
    # classpath - it isn't bundled in the base Spark image, so every prior
    # run failed at .load() with "Failed to find data source: kafka".
    # The real fix is pipeline/Dockerfile's spark-submit --packages flag:
    # package resolution happens during spark-submit's own launch phase,
    # before the JVM classloader is fixed, so setting spark.jars.packages
    # here from inside the driver is too late once launched via
    # spark-submit (confirmed - it silently did nothing). Kept here too as
    # a fallback for the case where this module is imported and run
    # directly (python spark_consumer.py) rather than via spark-submit,
    # where builder.config() is the only mechanism available.
    return (
        SparkSession.builder.appName("pos-fraud-feature-pipeline")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def run() -> None:
    spark = build_spark_session()

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC_TRANSACTIONS)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = raw_stream.select(
        F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("txn")
    ).select("txn.*")

    def process_batch(batch_df, batch_id: int) -> None:
        """batch_df is a static DataFrame for this one micro-batch (not the
        streaming DataFrame) - see the module docstring for why the window
        functions in build_features() have to run here rather than upstream."""
        features_df = build_features(batch_df)
        write_batch(features_df, batch_id)

    query = (
        transactions.writeStream.outputMode("append")
        .foreachBatch(process_batch)
        .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "features"))
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
