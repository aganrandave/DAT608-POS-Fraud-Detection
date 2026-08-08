"""MLflow experiment/registry configuration shared by both training scripts."""
import os

import mlflow

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pos-fraud-detection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def register_model(run_id: str, artifact_path: str, registered_name: str) -> None:
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.register_model(model_uri=model_uri, name=registered_name)


def load_latest_model(registered_name: str, stage: str = "Production"):
    model_uri = f"models:/{registered_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
