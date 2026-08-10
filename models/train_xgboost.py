"""Trains the supervised XGBoost fraud classifier and logs it to MLflow.

Honesty note: as of this run, the training data (data/features.xlsx joined
with is_fraud) has only 8 rows. 5-fold cross-validation is still performed
here because the ticket calls for it, but with N=8 and 3 positive examples,
individual folds can have as few as one test sample, sometimes zero
positives in that sample — per-fold precision/recall/F1 are not
statistically meaningful at this scale, and neither is the "F1 > 0.75"
target. Treat the numbers this script prints/logs as a correctness check of
the pipeline mechanics, not a production model quality bar. Re-run once
the producer/pipeline have generated real volume (SCRUM-20/23).
"""
import numpy as np
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier

from excel_reader import load_training_frame
from mlflow_registry import EXPERIMENT_NAME, MODEL_NAME_XGBOOST, register_model

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]

N_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    rows = load_training_frame()
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def build_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )


def cross_validate(df: pd.DataFrame, scale_pos_weight: float) -> dict:
    """Plain KFold (not stratified — with only 3 positive rows total,
    StratifiedKFold(n_splits=5) is infeasible). Returns the mean of each
    metric across folds, using zero_division=0 so degenerate folds (no
    positive predictions or no positive ground truth) don't raise."""
    X, y = df[FEATURE_COLUMNS], df["is_fraud"]
    kfold = KFold(n_splits=min(N_SPLITS, len(df)), shuffle=True, random_state=42)

    precisions, recalls, f1s, aucs = [], [], [], []
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = build_model(scale_pos_weight)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if y_test.nunique() > 1:
            y_prob = fold_model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_prob))

    return {
        "cv_precision_mean": float(np.mean(precisions)),
        "cv_recall_mean": float(np.mean(recalls)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_roc_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_folds_with_both_classes": len(aucs),
        "cv_n_splits": kfold.get_n_splits(),
    }


def train() -> XGBClassifier:
    df = load_dataset()
    scale_pos_weight = compute_scale_pos_weight(df["is_fraud"])

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["is_fraud"], test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost"):
        model = build_model(scale_pos_weight)
        model.fit(X_train, y_train)

        mlflow.log_params(model.get_params())
        mlflow.log_param("n_rows_trained_on", len(df))
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # Held-out split metrics
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        mlflow.log_metric("holdout_precision", precision)
        mlflow.log_metric("holdout_recall", recall)
        mlflow.log_metric("holdout_f1", f1)
        if y_test.nunique() > 1:
            y_prob = model.predict_proba(X_test)[:, 1]
            mlflow.log_metric("holdout_roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.log_metric("holdout_average_precision", average_precision_score(y_test, y_prob))

        # 5-fold cross-validation (see module docstring for the N=8 caveat)
        cv_results = cross_validate(df, scale_pos_weight)
        for key, value in cv_results.items():
            mlflow.log_metric(key, value)

        # Feature importance
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            mlflow.log_metric(f"importance_{feature}", float(importance))

        run_id = mlflow.active_run().info.run_id
        register_model(run_id, artifact_path="xgboost_model", registered_name=MODEL_NAME_XGBOOST)

        print(f"Trained on {len(df)} rows (scale_pos_weight={scale_pos_weight:.2f})")
        print(f"Holdout: precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
        print(f"5-fold CV means: {cv_results}")

    return model


if __name__ == "__main__":
    train()
