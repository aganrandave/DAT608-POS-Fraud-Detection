"""MLflow experiment/registry configuration shared by both training scripts."""
import json
import os

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pos-fraud-detection")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Registered model names and target stage, defined as constants per SCRUM-32.
MODEL_NAME_XGBOOST = "pos-fraud-xgboost"
MODEL_NAME_ISOLATION_FOREST = "pos-fraud-isolation-forest"
PRODUCTION_STAGE = "Production"

# Artifact filenames logged as siblings of each model, inside the SAME
# artifact_path the model itself was logged under (see train_xgboost.py /
# train_isolation_forest.py). Because `models:/{name}/{stage}` resolves to
# that same artifact root, these are retrievable by appending the filename
# to the model's own registry URI - verified end-to-end against a real
# sqlite-backed MLflow store before relying on it here.
XGBOOST_THRESHOLD_ARTIFACT = "decision_threshold.json"
ISOLATION_FOREST_BOUNDS_ARTIFACT = "score_bounds.json"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ModelMetadataError(RuntimeError):
    """Raised when a registered model loads but its required sibling
    metadata artifact (decision threshold / score bounds) is missing or
    malformed. Kept distinct from a plain "model not registered yet"
    failure so callers (scoring/scorer.py) can report *why* the service
    isn't ready instead of masking a real deployment mistake as a normal
    cold-boot condition."""


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


def log_sibling_artifact(data: dict, filename: str, artifact_path: str) -> None:
    """Write `data` as JSON and log it under the SAME artifact_path as the
    model being registered in this run, so it travels with that model
    version as a retrievable sibling file (see the module docstring above
    for why `models:/{name}/{stage}/{filename}` can then fetch it directly).
    Must be called inside the same `with mlflow.start_run():` block that
    logs the model, before register_model()."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, filename)
        with open(local_path, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(local_path, artifact_path=artifact_path)


def _load_sibling_artifact(registered_name: str, filename: str, stage: str) -> dict:
    model_uri = f"models:/{registered_name}/{stage}"
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/{filename}")
        with open(local_path) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - any failure here means "metadata missing/unreadable"
        raise ModelMetadataError(
            f"{registered_name}/{stage} is registered but its required {filename} "
            f"sibling artifact is missing or unreadable: {exc}"
        ) from exc


def load_xgboost_model_and_threshold(stage: str = PRODUCTION_STAGE):
    """Returns (model, decision_threshold) for the registered XGBoost
    classifier. Loaded via the native `mlflow.xgboost` flavor (not
    `mlflow.pyfunc`) specifically so callers get the real XGBClassifier
    object back and can call `.predict_proba()` on it - `pyfunc.predict()`
    on a classifier calls its native `.predict()`, which returns class
    labels, not probabilities. Raises ModelMetadataError, not a silent
    fallback, if the model is registered but decision_threshold.json isn't."""
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME_XGBOOST}/{stage}")
    metadata = _load_sibling_artifact(MODEL_NAME_XGBOOST, XGBOOST_THRESHOLD_ARTIFACT, stage)
    if "decision_threshold" not in metadata:
        raise ModelMetadataError(
            f"{MODEL_NAME_XGBOOST}/{stage}'s {XGBOOST_THRESHOLD_ARTIFACT} is missing the "
            "'decision_threshold' key"
        )
    return model, float(metadata["decision_threshold"])


def load_isolation_forest_model_and_bounds(stage: str = PRODUCTION_STAGE):
    """Returns (model, score_min, score_max) for the registered Isolation
    Forest. Loaded via the native `mlflow.sklearn` flavor so callers can
    call `.score_samples()` directly. score_min/score_max are the fitted
    training-set bounds logged by train_isolation_forest.py, used to
    reproducibly normalize new score_samples() output into a 0-1 range -
    see scoring/scorer.py for how they're applied."""
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME_ISOLATION_FOREST}/{stage}")
    metadata = _load_sibling_artifact(MODEL_NAME_ISOLATION_FOREST, ISOLATION_FOREST_BOUNDS_ARTIFACT, stage)
    if "score_min" not in metadata or "score_max" not in metadata:
        raise ModelMetadataError(
            f"{MODEL_NAME_ISOLATION_FOREST}/{stage}'s {ISOLATION_FOREST_BOUNDS_ARTIFACT} is "
            "missing 'score_min'/'score_max'"
        )
    return model, float(metadata["score_min"]), float(metadata["score_max"])
