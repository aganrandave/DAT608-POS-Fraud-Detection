"""Tests the actual MLflow round trip for the threshold/score-bounds sibling
artifacts - not mocked, a real local sqlite-backed tracking store, because
the thing being verified (that `models:/{name}/{stage}/{filename}` can
retrieve a file logged alongside the model) is precisely the kind of
MLflow-version-specific behaviour a mock would hide a regression in.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


@pytest.fixture()
def mlflow_sqlite_store(tmp_path, monkeypatch):
    """Points mlflow at a throwaway sqlite store under pytest's tmp_path,
    isolated per test, torn down automatically with the directory."""
    import mlflow

    db_path = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db_path}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_xgboost_threshold_round_trips_through_real_mlflow_registry(mlflow_sqlite_store):
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    import mlflow_registry
    from mlflow_registry import log_sibling_artifact

    model_name = _unique_name("test-xgb")
    X = np.random.RandomState(0).rand(20, 2)
    y = np.array([0, 1] * 10)
    model = XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)

    mlflow.set_experiment("test-registry")
    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        log_sibling_artifact(
            {"decision_threshold": 0.337}, "decision_threshold.json", artifact_path="xgboost_model"
        )
        run_id = run.info.run_id

    result = mlflow.register_model(f"runs:/{run_id}/xgboost_model", model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    )

    # load_xgboost_model_and_threshold() hardcodes MODEL_NAME_XGBOOST, which
    # a throwaway uniquely-named test model doesn't match - exercise the
    # exact same code path (native flavor load + the shared private sibling
    # loader) it calls internally instead of duplicating the constant.
    loaded_model = mlflow.xgboost.load_model(f"models:/{model_name}/Production")
    metadata = mlflow_registry._load_sibling_artifact(model_name, "decision_threshold.json", "Production")
    threshold = float(metadata["decision_threshold"])

    assert threshold == pytest.approx(0.337)
    assert hasattr(loaded_model, "predict_proba")


def test_isolation_forest_bounds_round_trip_through_real_mlflow_registry(mlflow_sqlite_store):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    import mlflow_registry
    from mlflow_registry import log_sibling_artifact

    model_name = _unique_name("test-if")
    X = np.random.RandomState(0).rand(20, 2)
    model = IsolationForest(n_estimators=5, random_state=0).fit(X)

    mlflow.set_experiment("test-registry")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="isolation_forest_model")
        log_sibling_artifact(
            {"score_min": -0.42, "score_max": 0.31},
            "score_bounds.json",
            artifact_path="isolation_forest_model",
        )
        run_id = run.info.run_id

    result = mlflow.register_model(f"runs:/{run_id}/isolation_forest_model", model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    )

    loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    metadata = mlflow_registry._load_sibling_artifact(model_name, "score_bounds.json", "Production")

    assert hasattr(loaded_model, "score_samples")
    assert metadata["score_min"] == pytest.approx(-0.42)
    assert metadata["score_max"] == pytest.approx(0.31)


def test_missing_sibling_artifact_raises_model_metadata_error(mlflow_sqlite_store):
    # A model registered WITHOUT its threshold/bounds artifact (e.g. logged
    # by an older version of the training script) must fail with a clear,
    # typed error - not a confusing generic exception, and not a silent
    # fallback to some default.
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    from mlflow_registry import ModelMetadataError, _load_sibling_artifact

    model_name = _unique_name("test-xgb-no-metadata")
    X = np.random.RandomState(0).rand(20, 2)
    y = np.array([0, 1] * 10)
    model = XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)

    mlflow.set_experiment("test-registry")
    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        # Deliberately NOT logging decision_threshold.json.
        run_id = run.info.run_id

    result = mlflow.register_model(f"runs:/{run_id}/xgboost_model", model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=True
    )

    with pytest.raises(ModelMetadataError, match="decision_threshold.json"):
        _load_sibling_artifact(model_name, "decision_threshold.json", "Production")
