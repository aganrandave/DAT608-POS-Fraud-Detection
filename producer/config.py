"""Runtime configuration for the transaction generator and Kafka producer."""
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "pos-transactions")

DATA_DIR = os.getenv("DATA_DIR", "data")
TRANSACTIONS_XLSX = os.getenv("TRANSACTIONS_XLSX", os.path.join(DATA_DIR, "transactions_raw.xlsx"))
TERMINALS_XLSX = os.getenv("TERMINALS_XLSX", os.path.join(DATA_DIR, "reference", "terminals.xlsx"))
MERCHANTS_XLSX = os.getenv("MERCHANTS_XLSX", os.path.join(DATA_DIR, "reference", "merchants.xlsx"))

# Generation cadence
TRANSACTIONS_PER_SECOND = float(os.getenv("TRANSACTIONS_PER_SECOND", "5"))
FRAUD_RATE = float(os.getenv("FRAUD_RATE", "0.02"))

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
