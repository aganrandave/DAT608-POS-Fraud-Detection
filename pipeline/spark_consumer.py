"""Spark Structured Streaming job: Kafka transactions -> engineered features.

Reads raw transaction JSON from the pos-transactions Kafka topic, applies
the windowed feature transforms in feature_windows.py, and writes each
micro-batch to data/features.xlsx via excel_writer.write_batch, in addition
to publishing the enriched stream back to Kafka for the scoring service.
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from excel_writer import write_batch
from feature_windows import build_features
from schema import TRANSACTION_SCHEMA

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "pos-transactions")
KAFKA_TOPIC_FEATURES = os.getenv("KAFKA_TOPIC_FEATURES", "pos-features")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints")


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("pos-fraud-feature-pipeline")
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

    features = build_features(transactions)

    query = (
        features.writeStream.outputMode("append")
        .foreachBatch(write_batch)
        .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "features"))
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
