"""Trains the supervised XGBoost fraud classifier and logs it to MLflow.

Training data as of this run: 30,008 real rows (data/features.xlsx joined
with is_fraud), bulk-generated via producer/bulk_generate.py +
pipeline/batch_feature_engineering.py (SCRUM-20/23), ~3.8% fraud.

Progression across three fixes, holdout F1 (properly out-of-sample each
time - see select_threshold()/train() for the leak-free methodology used
for the final number):
  1. N=8 original data: F1=0.00 (degenerate - too little data).
  2. +30k rows, still-broken generator (unpinned terminals, compressed
     timestamps, no fraud bursts): F1=0.06 (AUC ~0.58) - volume alone did
     not fix it; see producer/transaction_generator.py's docstring for the
     root cause and fix.
  3. +fixed generator, default 0.5 decision threshold: F1=0.69.
  4. +amount_ngn/amount_vs_bin_avg_ratio features, decision threshold
     tuned on a separate validation split (not the reported test set):
     precision=0.85 recall=0.65 **F1=0.74** - just under the ticket's
     F1>0.75 bar, honestly reported (an earlier same-set threshold-tuning
     attempt read F1=0.785, which was optimistically biased by picking the
     threshold and scoring it on the same holdout - see git history).

A CTGAN-based synthetic augmentation of this data was evaluated
(models/synthetic_augmentation.py + validate_synthetic_features.py). ML
utility passes cleanly (TSTR/TRTR AUC gap 0.015) and privacy is clean
(DNNR 12.0), but the gate still does NOT approve it: CTGAN struggles to
reproduce the sparse/bursty velocity_1h and terminal_reversal_count
distributions specifically. USE_SYNTHETIC_AUGMENTATION has no effect
until a synthetic set actually passes that gate.
"""
import numpy as np
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier

from excel_reader import FEATURE_COLUMNS, load_augmented_training_frame
from mlflow_registry import EXPERIMENT_NAME, MODEL_NAME_XGBOOST, register_model

N_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    """Real data by default. Set USE_SYNTHETIC_AUGMENTATION=true to add
    CTGAN-synthesized rows on top - only takes effect if
    validate_synthetic_features.py's gate actually approved them; see
    excel_reader.load_augmented_training_frame()."""
    rows = load_augmented_training_frame()
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


def select_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Decision threshold that maximises F1 on the given (val) set. The
    default classifier threshold of 0.5 is rarely optimal for a heavily
    imbalanced problem trained with scale_pos_weight - this is a standard,
    non-leaky way to pick a better one, PROVIDED it's selected on a set the
    final reported metric doesn't also use (see train())."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    return float(thresholds[np.argmax(f1s[:-1])])


def train() -> XGBClassifier:
    df = load_dataset()
    scale_pos_weight = compute_scale_pos_weight(df["is_fraud"])

    # Three-way split: train the threshold-selection model on X_train, pick
    # the decision threshold on X_val, then fit the final model on the full
    # X_trainval and report at that threshold on the untouched X_test. This
    # avoids the leakage of picking a threshold and reporting F1 on the same
    # holdout set, which is optimistically biased.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["is_fraud"], test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost"):
        threshold_model = build_model(scale_pos_weight)
        threshold_model.fit(X_train, y_train)
        decision_threshold = select_threshold(y_val, threshold_model.predict_proba(X_val)[:, 1])

        model = build_model(scale_pos_weight)
        model.fit(X_trainval, y_trainval)

        mlflow.log_params(model.get_params())
        mlflow.log_param("n_rows_trained_on", len(df))
        mlflow.log_param("decision_threshold", decision_threshold)
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # Held-out test-set metrics, at both the default 0.5 threshold (for
        # comparability with earlier runs) and the tuned threshold above.
        y_prob = model.predict_proba(X_test)[:, 1]
        precision_default, recall_default, f1_default, _ = precision_recall_fscore_support(
            y_test, (y_prob >= 0.5).astype(int), average="binary", zero_division=0
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, (y_prob >= decision_threshold).astype(int), average="binary", zero_division=0
        )
        mlflow.log_metric("holdout_precision_default_threshold", precision_default)
        mlflow.log_metric("holdout_recall_default_threshold", recall_default)
        mlflow.log_metric("holdout_f1_default_threshold", f1_default)
        mlflow.log_metric("holdout_precision", precision)
        mlflow.log_metric("holdout_recall", recall)
        mlflow.log_metric("holdout_f1", f1)
        if y_test.nunique() > 1:
            mlflow.log_metric("holdout_roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.log_metric("holdout_average_precision", average_precision_score(y_test, y_prob))

        # 5-fold cross-validation, default 0.5 threshold (see module docstring)
        cv_results = cross_validate(df, scale_pos_weight)
        for key, value in cv_results.items():
            mlflow.log_metric(key, value)

        # Feature importance
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            mlflow.log_metric(f"importance_{feature}", float(importance))

        run_id = mlflow.active_run().info.run_id
        register_model(run_id, artifact_path="xgboost_model", registered_name=MODEL_NAME_XGBOOST)

        print(f"Trained on {len(df)} rows (scale_pos_weight={scale_pos_weight:.2f})")
        print(f"Holdout @ default 0.5 threshold: precision={precision_default:.2f} "
              f"recall={recall_default:.2f} f1={f1_default:.2f}")
        print(f"Holdout @ tuned threshold {decision_threshold:.3f} (selected on a separate "
              f"validation split): precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
        print(f"5-fold CV means (default threshold): {cv_results}")

    return model


if __name__ == "__main__":
    train()
