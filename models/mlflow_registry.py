"""MLflow experiment/registry configuration shared by both training scripts."""
import os

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pos-fraud-detection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Registered model names and target stage, defined as constants per SCRUM-32.
MODEL_NAME_XGBOOST = "pos-fraud-xgboost"
MODEL_NAME_ISOLATION_FOREST = "pos-fraud-isolation-forest"
PRODUCTION_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def register_model(run_id: str, artifact_path: str, registered_name: str, promote: bool = True):
    """Register a model version and, by default, promote it straight to Production.

    This project has a single environment (no separate staging deployment), so
    promoting immediately on register is the intended workflow — see
    models/README.md's "Retraining and promoting a new version" section for
    the manual alternative if that ever changes.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=registered_name)

    if promote:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=registered_name,
            version=result.version,
            stage=PRODUCTION_STAGE,
            archive_existing_versions=True,
        )

    return result


def load_latest_model(registered_name: str, stage: str = PRODUCTION_STAGE):
    model_uri = f"models:/{registered_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
