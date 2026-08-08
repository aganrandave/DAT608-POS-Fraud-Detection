"""Trains the supervised XGBoost fraud classifier and logs it to MLflow."""
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from excel_reader import load_training_frame
from mlflow_registry import EXPERIMENT_NAME, register_model

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]


def load_dataset() -> pd.DataFrame:
    rows = load_training_frame()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def train() -> XGBClassifier:
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["is_fraud"], test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost"):
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="aucpr",
            random_state=42,
        )
        model.fit(X_train, y_train)

        mlflow.log_params(model.get_params())
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        score = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", score)

        run_id = mlflow.active_run().info.run_id
        register_model(run_id, artifact_path="xgboost_model", registered_name="pos-fraud-xgboost")

    return model


if __name__ == "__main__":
    train()
