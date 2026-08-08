"""Liveness/readiness helpers for the scoring service."""
from scorer import MODEL_VERSION


def health_status() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}
