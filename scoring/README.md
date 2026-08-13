# scoring

FastAPI service that scores incoming feature vectors with the known registered
XGBoost + Isolation Forest models and persists the result.

- `main.py` — `/health` and `/score` endpoints.
- `scorer.py` — `FraudScorer` loads both MLflow-registered models and blends their outputs into `fraud_probability` / `alert_tier`.
- `tiers.py` — dependency-free alert tier thresholds (mirrors `alerts/alert_tiers.py`), used by `scorer.py` and unit tested directly.
- `schemas.py` — Pydantic request/response models.
- `health.py` — liveness/readiness payload.
- `excel_writer.py` — appends each scored transaction to `data/fraud_scores.xlsx`.
- `benchmark.py` — p50/p95/max latency benchmark against a running `/score` endpoint.

## Run locally

```bash
pip install -r scoring/requirements.txt
uvicorn scoring.main:app --reload
```
This is where the API is documented
API docs: `http://localhost:8000/docs`

## Run via Docker

```bash
docker-compose up scoring
```
