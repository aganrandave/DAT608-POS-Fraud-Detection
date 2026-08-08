"""Trains the unsupervised Isolation Forest anomaly detector and logs it to MLflow."""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest

from excel_reader import load_features
from mlflow_registry import EXPERIMENT_NAME, register_model

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]


def load_dataset() -> pd.DataFrame:
    rows = load_features()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df


def train() -> IsolationForest:
    df = load_dataset()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="isolation_forest"):
        model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
        )
        model.fit(df[FEATURE_COLUMNS])

        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, artifact_path="isolation_forest_model")

        run_id = mlflow.active_run().info.run_id
        register_model(
            run_id, artifact_path="isolation_forest_model", registered_name="pos-fraud-isolation-forest"
        )

    return model


if __name__ == "__main__":
    train()
