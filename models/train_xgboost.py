"""Trains the supervised XGBoost fraud classifier and logs it to MLflow.

Training data as of this run: 30,008 real rows (data/features.xlsx joined
with is_fraud), bulk-generated via producer/bulk_generate.py +
pipeline/batch_feature_engineering.py (SCRUM-20/23), ~3.8% fraud.

Progression across four fixes, holdout F1 (properly out-of-sample each
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
     precision=0.85 recall=0.65 F1=0.74 - just under the ticket's F1>0.75
     bar (an earlier same-set threshold-tuning attempt read F1=0.785,
     optimistically biased by scoring the threshold on the same holdout
     it was picked on - caught and fixed, see git history).
  5. +hyperparameters chosen by models/tune_xgboost.py's CV grid sweep
     (max_depth 5->4): precision=0.85 recall=0.72 **F1=0.78** - clears
     the F1>0.75 bar, honestly, with proper out-of-sample evaluation.

A CTGAN-based synthetic augmentation of this data (models/
synthetic_augmentation.py + validate_synthetic_features.py) now passes
the validation gate after PR #35's fixes. USE_SYNTHETIC_AUGMENTATION=true
adds the approved synthetic rows to the train/val pool only - see
train()'s docstring for why the reported test set stays real-only
regardless, so augmented-vs-real-only runs are directly comparable.

Measured result of actually enabling it: holdout F1 0.78 -> 0.73 (0.85/0.72
precision/recall -> 0.83/0.66) on the SAME real-only test set, 5-fold CV
on real data unchanged (confirming both runs are scored identically).
Passing the validation gate confirms the synthetic data is statistically
valid and privacy-safe - it does not guarantee it helps this specific
downstream model, and here it measurably didn't: with 24,006 real training
rows already, the marginal value of more data appears saturated, and the
synthetic rows' imperfect distributional match (even though "acceptable"
by every gate metric) added noise rather than signal. Left off by default
(USE_SYNTHETIC_AUGMENTATION unset) on the strength of this result.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from excel_reader import FEATURE_COLUMNS, load_approved_synthetic_rows, load_training_frame
from mlflow_registry import EXPERIMENT_NAME, MODEL_NAME_XGBOOST, register_model

N_SPLITS = 5


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def load_dataset() -> pd.DataFrame:
    """Real data only - used by cross_validate() and CV-based reporting,
    which don't have a train()-style held-out test split to protect. See
    train()'s docstring for why the real+synthetic mix used there is
    assembled differently."""
    return _to_frame(load_training_frame())


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Hyperparameters chosen by models/tune_xgboost.py's grid sweep,
    selected on 5-fold CV mean average precision (max_depth=4,
    learning_rate=0.05, n_estimators=200; AP=0.805 vs. 0.800 for the prior
    max_depth=5 default) - see that script for the full grid and results."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )


def cross_validate(df: pd.DataFrame, scale_pos_weight: float) -> dict:
    """Stratified 5-fold CV (feasible now with ~1,146 fraud rows - earlier
    versions of this project had only 3 total and used plain KFold; that
    constraint no longer applies, and stratification gives more reliable
    per-fold estimates). Returns the mean of each metric across folds,
    using zero_division=0 so degenerate folds don't raise."""
    X, y = df[FEATURE_COLUMNS], df["is_fraud"]
    kfold = StratifiedKFold(n_splits=min(N_SPLITS, len(df)), shuffle=True, random_state=42)

    precisions, recalls, f1s, aucs = [], [], [], []
    for train_idx, test_idx in kfold.split(X, y):
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
    """Test set is ALWAYS drawn from real data only, split off BEFORE any
    synthetic augmentation is added - so a USE_SYNTHETIC_AUGMENTATION=true
    run and a real-only run are scored on identical, comparable, genuinely
    real-world holdout data. Synthetic rows (only added if the operator
    opted in AND validate_synthetic_features.py's gate approved them - see
    excel_reader.load_approved_synthetic_rows()) are mixed in ONLY to the
    train/val pool used for fitting and threshold selection. Without this
    split-before-augment ordering, a naive train_test_split on the
    combined frame would let synthetic rows leak into the reported test
    set, partly evaluating the model on data drawn from the same
    generative process it was trained on - not a fair comparison."""
    real_df = _to_frame(load_training_frame())

    # Three-way split: train the threshold-selection model on X_train, pick
    # the decision threshold on X_val, then fit the final model on the full
    # X_trainval and report at that threshold on the untouched, real-only
    # X_test. This avoids the leakage of picking a threshold and reporting
    # F1 on the same holdout set, which is optimistically biased.
    real_trainval, X_test, real_y_trainval, y_test = train_test_split(
        real_df[FEATURE_COLUMNS], real_df["is_fraud"], test_size=0.2, random_state=42, stratify=real_df["is_fraud"]
    )

    synthetic_rows = load_approved_synthetic_rows()
    if synthetic_rows:
        synthetic_df = _to_frame(synthetic_rows)
        X_trainval = pd.concat([real_trainval, synthetic_df[FEATURE_COLUMNS]], ignore_index=True)
        y_trainval = pd.concat([real_y_trainval, synthetic_df["is_fraud"]], ignore_index=True)
    else:
        X_trainval, y_trainval = real_trainval, real_y_trainval

    scale_pos_weight = compute_scale_pos_weight(y_trainval)
    n_synthetic_in_trainval = len(synthetic_rows)

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
        mlflow.log_param("n_rows_trained_on", len(X_trainval))
        mlflow.log_param("n_real_rows_trained_on", len(real_trainval))
        mlflow.log_param("n_synthetic_rows_trained_on", n_synthetic_in_trainval)
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

        # 5-fold cross-validation, real data only regardless of augmentation
        # (keeps CV folds free of the same test-leakage risk train() avoids
        # above), default 0.5 threshold
        cv_results = cross_validate(real_df, compute_scale_pos_weight(real_df["is_fraud"]))
        for key, value in cv_results.items():
            mlflow.log_metric(key, value)

        # Feature importance
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
            mlflow.log_metric(f"importance_{feature}", float(importance))

        run_id = mlflow.active_run().info.run_id
        register_model(run_id, artifact_path="xgboost_model", registered_name=MODEL_NAME_XGBOOST)

        print(f"Trained on {len(X_trainval)} rows ({len(real_trainval)} real + "
              f"{n_synthetic_in_trainval} synthetic), scale_pos_weight={scale_pos_weight:.2f}")
        print(f"Holdout @ default 0.5 threshold: precision={precision_default:.2f} "
              f"recall={recall_default:.2f} f1={f1_default:.2f}")
        print(f"Holdout @ tuned threshold {decision_threshold:.3f} (selected on a separate "
              f"validation split): precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
        print(f"5-fold CV means (default threshold): {cv_results}")

    return model


if __name__ == "__main__":
    train()
