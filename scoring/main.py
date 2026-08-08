"""FastAPI scoring service: exposes /score, /health, and blends model outputs."""
from datetime import datetime, timezone

from fastapi import FastAPI

from excel_writer import append_score
from health import health_status
from schemas import FeatureRequest, HealthResponse, ScoreResponse
from scorer import FraudScorer

app = FastAPI(title="POS Fraud Scoring Service")
scorer = FraudScorer()


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return health_status()


@app.post("/score", response_model=ScoreResponse)
def score(request: FeatureRequest) -> dict:
    result = scorer.score(request.model_dump())
    result["transaction_id"] = request.transaction_id
    result["scored_at"] = datetime.now(timezone.utc).isoformat()

    append_score(result)

    return result
