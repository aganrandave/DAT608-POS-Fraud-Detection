# Architecture

## Overview

The system is a real-time fraud detection pipeline for Nigerian POS agent
transactions, composed of seven layers that communicate over Kafka and
persist state to Git-tracked Excel workbooks under [`data/`](../data/).

```
POS terminal API layer simulator -> Kafka(pos-transactions) -> pipeline -> Kafka(pos-features)
                                                          |
                                                          v
                                              models (offline training, MLflow)
                                                          |
                                                          v
                          scoring (FastAPI) -> Kafka(pos-fraud-scores)
                                                          |
                                                          v
                          alerts (ksqlDB)  -> Kafka(pos-fraud-alerts)
                                                          |
                                                          v
                                                     dashboard (Streamlit)
```

## Layers

| Layer | Responsibility | Reads | Writes |
|---|---|---|---|
| [simulator](../producer/README.md) | Generates synthetic transactions, publishes to Kafka | `data/reference/*.xlsx` | `data/transactions_raw.xlsx` |
| [pipeline](../pipeline/README.md) | Spark Structured Streaming windowed feature engineering | Kafka `pos-transactions` | `data/features.xlsx` |
| [models](../models/README.md) | Trains XGBoost + Isolation Forest, tracked in MLflow | `data/features.xlsx`, `data/transactions_raw.xlsx` | MLflow model registry |
| [scoring](../scoring/README.md) | FastAPI service blending both models into a fraud probability | MLflow registry | `data/fraud_scores.xlsx` |
| [alerts](../alerts/README.md) | ksqlDB stream joins, tiering, filtering | `data/reference/*.xlsx` (bootstrap) | `data/alerts.xlsx` |
| [dashboard](../dashboard/README.md) | Streamlit visualization of alerts and scores | `data/alerts.xlsx`, `data/fraud_scores.xlsx` | — |
| [infra](../infra/README.md) | Docker Compose orchestration of all of the above | — | — |

## Data store

See the root [README's Data store section](../README.md#data-store) and the
[data dictionary](data_dictionary.md) for the full schema of every workbook.

## Alert logic

See [alert_logic.md](alert_logic.md) for the tiering thresholds and the
ksqlDB join logic that produces alerts.

## API

See [api_reference.md](api_reference.md) for the scoring service's REST
contract.
