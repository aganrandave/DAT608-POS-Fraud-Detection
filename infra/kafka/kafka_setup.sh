#!/usr/bin/env bash
# Creates the Kafka topics used across the pipeline once the broker is up.
set -euo pipefail

BROKER="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"

TOPICS=(
  "pos-transactions"
  "pos-features"
  "pos-fraud-scores"
  "pos-fraud-alerts"
  "terminal-reference"
  "merchant-reference"
)

for topic in "${TOPICS[@]}"; do
  kafka-topics --bootstrap-server "$BROKER" \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions 3 \
    --replication-factor 1
done

echo "Kafka topics ready on $BROKER:"
kafka-topics --bootstrap-server "$BROKER" --list
