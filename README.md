# DAT608 — POS Agent Fraud Detection

[![CI](https://github.com/aganrandave/DAT608-POS-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aganrandave/DAT608-POS-Fraud-Detection/actions/workflows/ci.yml)

A real-time fraud detection system for Nigerian point-of-sale (POS) agent
transactions. Synthetic transactions are streamed through Kafka, enriched
with rolling features in Spark Structured Streaming, scored by an ensemble
of XGBoost and Isolation Forest models tracked in MLflow, and surfaced as
tiered alerts through ksqlDB and a Streamlit dashboard. The project is a
DAT608 Big Data capstone at Pan-Atlantic University.

## Team

| Name | Role | GitHub username | Pipeline layer(s) |
|------|------|------------------|--------------------|
| David Aganran | EM/PM | [aganrandave](https://github.com/aganrandave) | infra, dashboard, tests, docs |
| Omobolanle Adegboye     | LLM Evaluator | [bolanle-ea](https://github.com/bolanle-ea) | models |
| Ololade Ajaegbu     | ML Engineer | [Ololade-ajaegbu](https://github.com/Ololade-ajaegbu) | models |
| Oluwatiseunla Abdul     | Data Scientist | [Oluwatiseunla](https://github.com/Oluwatiseunla) | scoring |
| John Babalola     | Data Engineer | [john-babalola1307](https://github.com/john-babalola1307) | producer, pipeline, alerts |
| Jesselyn Ayanka | Data Management | [Jessayanka005](https://github.com/Jessayanka005) | infra, docker |

Unstaffed layers (dashboard, tests, docs) default to the EM/PM until a
named owner is assigned — see [`.github/CODEOWNERS`](.github/CODEOWNERS) for
the enforced mapping.

## Architecture overview

```
 +-------------+     +-------+     +------------------+     +---------------------+
 | Transaction | --> | Kafka | --> | Spark Structured | --> | Feature Store       |
 | Generator   |     | Topic |     | Streaming        |     | (data/features.xlsx)|
 +-------------+     +-------+     +------------------+     +----------+----------+
                                                                       |
                                                                       v
                                                          +------------------------+
                                                          | XGBoost + Isolation    |
                                                          | Forest (MLflow logged) |
                                                          +------------+-----------+
                                                                       |
                                                                       v
+--------------+     +------------+     +----------+     +-------------------------+
| Streamlit    | <-- |  Alerts    | <-- | ksqlDB   | <-- | FastAPI Scoring Service |
| Dashboard    |     | (alerts.   |     | Alert    |     | (data/fraud_scores.xlsx)|
|              |     |  xlsx)     |     | Logic    |     |                         |
+--------------+     +------------+     +----------+     +-------------------------+
```

All services are orchestrated with Docker Compose — see [infra/README.md](infra/README.md).

## Layers

| Layer | README |
|---|---|
| producer — synthetic transactions & Kafka producer | [producer/README.md](producer/README.md) |
| pipeline — Spark Structured Streaming feature engineering | [pipeline/README.md](pipeline/README.md) |
| models — XGBoost + Isolation Forest, MLflow | [models/README.md](models/README.md) |
| scoring — FastAPI scoring service | [scoring/README.md](scoring/README.md) |
| alerts — ksqlDB alert logic | [alerts/README.md](alerts/README.md) |
| dashboard — Streamlit UI | [dashboard/README.md](dashboard/README.md) |
| infra — Docker Compose orchestration | [infra/README.md](infra/README.md) |

## Quickstart

```bash
git clone https://github.com/aganrandave/DAT608-POS-Fraud-Detection.git
cd DAT608-POS-Fraud-Detection
cp .env.example .env
docker compose up --build
```

The Streamlit dashboard will be available at `http://localhost:8501` and the
FastAPI scoring service at `http://localhost:8000/docs`.

## Data store

This project uses **Excel files instead of a live database**. Every
persistent record — raw transactions, engineered features, model scores,
and alerts — lives under [`data/`](data/) as a tracked `.xlsx` file
(`data/transactions_raw.xlsx`, `data/features.xlsx`, `data/fraud_scores.xlsx`,
`data/alerts.xlsx`, `data/reference/terminals.xlsx`,
`data/reference/merchants.xlsx`). Because these files are committed to Git,
every pull request carries its data alongside the code that produced it,
and reviewers can diff data changes the same way they diff code. Each layer
reads and writes these files through a shared `openpyxl` + `filelock`
pattern implemented in that layer's `excel_reader.py` / `excel_writer.py`.

## Documentation

Full technical documentation, including the data dictionary, API reference,
and alert-tier logic, lives in [docs/architecture.md](docs/architecture.md).
