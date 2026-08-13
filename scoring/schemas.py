"""Pydantic request/response models for the FastAPI scoring service."""
from pydantic import BaseModel


class FeatureRequest(BaseModel):
    transaction_id: str
    terminal_id: str
    card_bin: str
    velocity_1h: float
    geo_jump_km: float
    bin_spend_rate: float
    terminal_reversal_count: float
    amount_ngn: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    alert_tier: str
    xgboost_score: float
    isolation_forest_score: float
    model_version: str
    scored_at: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    models_ready: bool
