"""Liveness/readiness helpers for the scoring service."""
from scorer import MODEL_VERSION


def health_status(scorer=None) -> dict:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "models_ready": scorer.is_ready if scorer is not None else False,
    }
