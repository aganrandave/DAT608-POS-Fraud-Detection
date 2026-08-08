"""Evaluates trained models against a held-out split of the labeled feature set."""
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from excel_reader import load_training_frame

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]


def evaluate_xgboost(model) -> dict:
    df = pd.DataFrame(load_training_frame())
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)

    _, X_test, _, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["is_fraud"], test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
    }


if __name__ == "__main__":
    print("Run train_xgboost.train() first, then pass the returned model to evaluate_xgboost().")
