"""One-off batch alternative to kafka_producer.py's streaming loop.

kafka_producer.py generates and publishes transactions one at a time at
TRANSACTIONS_PER_SECOND, mirroring each into transactions_raw.xlsx as it
goes - the right shape for a live demo, but impractical for generating
tens of thousands of backfill rows (that would take hours at 5/sec and
requires a running Kafka broker). This script generates a large batch
in memory and writes it to transactions_raw.xlsx in one locked pass,
with no Kafka dependency, for backfilling training-data volume.

Usage:
    python bulk_generate.py --n 30000
"""
import argparse

from excel_writer import append_transactions
from transaction_generator import generate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30_000, help="number of transactions to generate")
    args = parser.parse_args()

    transactions = generate_batch(args.n)
    n_fraud = sum(1 for t in transactions if t["is_fraud"])

    append_transactions(transactions)

    print(f"Appended {len(transactions)} transactions to transactions_raw.xlsx")
    print(f"Fraud rows: {n_fraud} ({n_fraud / len(transactions):.2%})")


if __name__ == "__main__":
    main()
