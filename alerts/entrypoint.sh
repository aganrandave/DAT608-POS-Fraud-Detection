#!/usr/bin/env bash
# Publishes reference data once per container start, then runs the
# long-lived alert consumer. See publish_reference_data.py's docstring for
# why this needs to happen before/alongside the ksqlDB reference tables
# being created.
set -e
python publish_reference_data.py
exec python kafka_consumer.py
