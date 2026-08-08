"""Extracts and logs feature importance for the trained XGBoost model to MLflow."""
import mlflow
import pandas as pd

from train_xgboost import FEATURE_COLUMNS


def feature_importance_frame(model) -> pd.DataFrame:
    importances = model.feature_importances_
    return pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": importances}
    ).sort_values("importance", ascending=False)


def log_feature_importance(model, run_id: str) -> pd.DataFrame:
    frame = feature_importance_frame(model)
    with mlflow.start_run(run_id=run_id):
        for _, row in frame.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row["importance"])
    return frame


if __name__ == "__main__":
    print("Import log_feature_importance(model, run_id) after training a model.")
