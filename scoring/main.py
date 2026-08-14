"""FastAPI scoring service: exposes /score, /health, and blends model outputs."""
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from excel_writer import append_score
from health import health_status
from kafka_publisher import publish_score
from schemas import FeatureRequest, HealthResponse, ScoreResponse
from scorer import FraudScorer, ModelsNotReadyError

app = FastAPI(title="POS Fraud Scoring Service")
scorer = FraudScorer()


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return health_status(scorer)


@app.post("/score", response_model=ScoreResponse)
def score(request: FeatureRequest) -> dict:
    try:
        result = scorer.score(request.model_dump())
    except ModelsNotReadyError as exc:
        raise HTTPException(status_code=503, detail=f"Models not yet registered: {exc}") from exc

    result["transaction_id"] = request.transaction_id
    result["scored_at"] = datetime.now(timezone.utc).isoformat()

    append_score(result)
    publish_score(result)

    return result
