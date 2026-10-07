"""Evaluates trained models against a held-out split of the labeled feature set.

evaluate_xgboost() reports metrics for the STANDALONE XGBoost model only -
these are NOT the deployed blended-ensemble numbers (see scoring/scorer.py
for the actual XGBoost + Isolation Forest blend and alert-tier mapping).
evaluate_blended_scoring_contract() exists specifically to report metrics
for that actual deployed contract instead of leaving it unmeasured - see
its docstring for what it does and doesn't cover, INCLUDING a real leakage
issue in the production Isolation Forest that this module works around
rather than silently inherits (see _fit_evaluation_only_isolation_forest's
docstring).
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from excel_reader import FEATURE_COLUMNS, load_training_frame

# Must match train_isolation_forest.py's build parameters exactly, so the
# evaluation-only refit below is methodologically the same model, just
# fit on less data (the train partition, not the full dataset).
ISOLATION_FOREST_PARAMS = dict(n_estimators=200, contamination=0.02, random_state=42)


def _held_out_split(random_state: int = 42):
    """Same split shape as train_xgboost.train()'s outer real_trainval/X_test
    split (test_size=0.2, random_state=42, stratified) - using the identical
    random_state means this reproduces the SAME held-out rows that
    train_xgboost.py reported its headline metrics on, not a different
    random sample that would make the two numbers incomparable.

    XGBoost's own training (X_trainval -> X_train/X_val -> threshold
    selection on X_val, final fit on X_trainval) never touches X_test, so
    reusing the PRODUCTION XGBoost model against this split is leak-free -
    verified by construction: this is the exact split train_xgboost.train()
    itself carves off before doing anything else. Isolation Forest is a
    different story - see _fit_evaluation_only_isolation_forest()."""
    df = pd.DataFrame(load_training_frame())
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)

    return train_test_split(
        df[FEATURE_COLUMNS], df["is_fraud"], test_size=0.2, random_state=random_state,
        stratify=df["is_fraud"],
    )


def _fit_evaluation_only_isolation_forest(X_trainval: pd.DataFrame, y_trainval: pd.Series):
    """Fits a FRESH Isolation Forest on only the non-fraud rows of
    X_trainval, and computes score_samples() min/max bounds from X_trainval
    only - NOT the production model from train_isolation_forest.py.

    This exists because the production model is a genuine leakage source if
    reused for held-out evaluation: train_isolation_forest.py fits on ALL
    non-fraud rows (no train/test split at all) and computes its score_min/
    score_max over the FULL dataset, including whatever rows any X_test
    split later carves out. Any X_test row's non-fraud examples would
    already have been seen during that model's own fitting, and would have
    contributed to its score bounds - the "held-out" evaluation would not
    actually be held out for the Isolation Forest component.

    Returns (model, score_min, score_max), all computed from X_trainval
    alone - X_test is never touched here. Same hyperparameters as the
    production model (ISOLATION_FOREST_PARAMS) so this is a fair,
    methodologically-identical stand-in fit on less data, not a different
    model entirely."""
    import numpy as np

    non_fraud_trainval = X_trainval[y_trainval == 0]
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    model.fit(non_fraud_trainval)

    trainval_scores = model.score_samples(X_trainval)
    score_min = float(np.min(trainval_scores))
    score_max = float(np.max(trainval_scores))
    return model, score_min, score_max


def evaluate_xgboost(model, decision_threshold: float = 0.5) -> dict:
    """Standalone XGBoost metrics ONLY - precision/recall/F1 at the given
    decision_threshold (pass the validation-selected threshold from
    train_xgboost.py's select_threshold()/decision_threshold.json artifact
    to reproduce its reported numbers; defaults to 0.5 for comparison with
    an untuned classifier). These describe the XGBoost model in isolation,
    not the Isolation Forest blend or alert-tier policy the deployed
    scoring service actually applies - see evaluate_blended_scoring_contract
    for that."""
    X_trainval, X_test, y_trainval, y_test = _held_out_split()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    return {
        "decision_threshold": decision_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "n_test_rows": len(y_test),
        "n_test_fraud": int(y_test.sum()),
    }


def _blend_probability(xgb_prob, raw_if, if_score_min: float, if_score_max: float, xgboost_weight: float):
    """Same normalization/blend formula as scoring/scorer.py's score()
    method - kept in exact sync there and here so evaluation measures the
    deployed contract, not a reimplementation of it."""
    import numpy as np

    raw_if = np.asarray(raw_if)
    score_range = if_score_max - if_score_min
    if score_range <= 0:
        if_score = np.zeros_like(raw_if)
    else:
        if_score = np.clip((if_score_max - raw_if) / score_range, 0.0, 1.0)
    return xgboost_weight * np.asarray(xgb_prob) + (1 - xgboost_weight) * if_score


def _blended_fraud_probability_on_test(xgboost_model, X_trainval, y_trainval, X_test, xgboost_weight: float = 0.7):
    """Fits the evaluation-only Isolation Forest on X_trainval (never
    X_test - see _fit_evaluation_only_isolation_forest), then returns the
    blended fraud_probability Series for X_test. Shared by
    evaluate_blended_scoring_contract() and tier_operating_points() so both
    are built from the identical leak-free pipeline, not two
    re-implementations that could silently drift apart."""
    eval_if_model, if_score_min, if_score_max = _fit_evaluation_only_isolation_forest(X_trainval, y_trainval)

    # Split-discipline assertion, not just a comment - fails loudly if a
    # future edit ever lets X_test rows leak into the Isolation Forest fit.
    assert set(X_test.index).isdisjoint(set(X_trainval.index)), (
        "X_test overlaps X_trainval - the evaluation-only Isolation Forest "
        "would be contaminated by held-out rows"
    )

    xgb_prob = xgboost_model.predict_proba(X_test)[:, 1]
    raw_if = eval_if_model.score_samples(X_test)
    fraud_probability = pd.Series(
        _blend_probability(xgb_prob, raw_if, if_score_min, if_score_max, xgboost_weight), index=X_test.index
    )
    return fraud_probability, eval_if_model, if_score_min, if_score_max


def evaluate_blended_scoring_contract(xgboost_model, xgboost_weight: float = 0.7) -> dict:
    """Metrics for the ACTUAL deployed scoring contract: XGBoost
    predict_proba blended with normalized Isolation Forest score_samples,
    at the same weight scoring/scorer.py uses. Reports precision/recall/F1
    of the "is_alertable" decision (fraud_probability's tier is
    medium/high/critical - see alerts/alert_tiers.py) against the real
    is_fraud label, since that tier decision, not a raw probability cutoff,
    is what actually reaches a human in this system.

    Deliberately does NOT take a pre-loaded Isolation Forest model/bounds
    as a parameter (an earlier version of this function did, and it was a
    real leakage bug: the PRODUCTION Isolation Forest from
    train_isolation_forest.py is fit on every non-fraud row with no
    train/test split, and its score bounds are computed over the full
    dataset - reusing it here would mean X_test rows had already
    contributed to the model being "evaluated" on them). Instead, this
    fits a fresh, evaluation-only Isolation Forest on X_trainval alone, with
    the same hyperparameters as the production model (see
    _fit_evaluation_only_isolation_forest) - this result describes how that
    architecture is expected to generalize, not the literal production
    model object.

    xgboost_weight and the alert-tier thresholds applied inside
    is_alertable()/tier_for_probability() are both fixed constants in this
    codebase (scoring/scorer.py's XGBOOST_WEIGHT, alerts/alert_tiers.py's
    TIER_THRESHOLDS) - neither was selected by fitting anything to X_test,
    so using them here is not itself a leakage source. See
    tier_operating_points() for a breakdown across each individual
    threshold rather than just the deployed default."""
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "alerts"))
    from alert_tiers import is_alertable, tier_for_probability  # noqa: E402

    X_trainval, X_test, y_trainval, y_test = _held_out_split()
    fraud_probability, _, if_score_min, if_score_max = _blended_fraud_probability_on_test(
        xgboost_model, X_trainval, y_trainval, X_test, xgboost_weight
    )
    y_pred = fraud_probability.apply(lambda p: int(is_alertable(tier_for_probability(p))))

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    return {
        "metric": "precision/recall/F1 of is_alertable decision vs is_fraud (blended ensemble, "
                  "leakage-audited: evaluation-only Isolation Forest fit on the train partition only)",
        "xgboost_weight": xgboost_weight,
        "isolation_forest_score_min": if_score_min,
        "isolation_forest_score_max": if_score_max,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, fraud_probability),
        "average_precision": average_precision_score(y_test, fraud_probability),
        "n_test_rows": len(y_test),
        "n_test_fraud": int(y_test.sum()),
    }


def tier_operating_points(xgboost_model, xgboost_decision_threshold: float, xgboost_weight: float = 0.7) -> list:
    """Precision/recall/F1/false-positive-rate/alert-volume/confusion counts
    for each named operating point, all on the SAME leakage-audited
    held-out test set (see evaluate_blended_scoring_contract's docstring):

      - medium_and_above / high_and_above / critical: fixed thresholds on
        the BLENDED fraud_probability (0.30 / 0.60 / 0.85 - see
        alerts/alert_tiers.py's TIER_THRESHOLDS, the actual deployed
        policy).
      - xgboost_standalone_tuned_threshold: the validation-selected
        decision_threshold applied to xgboost_score ALONE, no Isolation
        Forest blend - i.e. what xgboost_decision reports in /score,
        included here specifically so it's comparable to the blended
        operating points in the same table rather than only ever reported
        on its own (see evaluate_xgboost).

    These 0.30/0.60/0.85 thresholds are round-number demonstration
    defaults carried over unchanged from the original ticket - nothing in
    this repository's history documents them being derived from an
    investigator-capacity model, a false-positive cost estimate, or any
    other empirical process. Do not describe the resulting operating
    point as chosen "by design" for its precision/recall tradeoff; the
    tradeoff is a measured CONSEQUENCE of an unvalidated threshold choice,
    not a stated goal."""
    X_trainval, X_test, y_trainval, y_test = _held_out_split()
    fraud_probability, _, _, _ = _blended_fraud_probability_on_test(
        xgboost_model, X_trainval, y_trainval, X_test, xgboost_weight
    )
    xgb_prob_only = pd.Series(xgboost_model.predict_proba(X_test)[:, 1], index=X_test.index)

    operating_points = [
        ("medium_and_above (blended >= 0.30)", fraud_probability >= 0.30),
        ("high_and_above (blended >= 0.60)", fraud_probability >= 0.60),
        ("critical (blended >= 0.85)", fraud_probability >= 0.85),
        (
            f"xgboost_standalone (tuned threshold {xgboost_decision_threshold:.3f})",
            xgb_prob_only >= xgboost_decision_threshold,
        ),
    ]

    n_total = len(y_test)
    results = []
    for name, alerted in operating_points:
        y_pred = alerted.astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        results.append({
            "operating_point": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
            "n_alerted": tp + fp,
            "pct_alerted": round(100 * (tp + fp) / n_total, 2) if n_total else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return results


def _terminal_group_split(test_size: float = 0.2, random_state: int = 42):
    """Conservative alternative to _held_out_split(): splits by UNIQUE
    terminal_id rather than by row, so no terminal appears in both the
    train and test partitions. Guards against the row-random split
    overstating generalization if correlated records (velocity bursts,
    repeated geo/BIN patterns at the same terminal) end up split across
    train and test - a real risk the row-random split does not rule out,
    since this project's fraud scenarios are deliberately terminal-local
    (see producer/fraud_scenarios.py)."""
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.DataFrame(load_training_frame())
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(splitter.split(df, groups=df["terminal_id"]))
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx]

    assert set(trainval_df["terminal_id"]).isdisjoint(set(test_df["terminal_id"])), (
        "terminal_id leaked across the group-aware train/test split"
    )

    return (
        trainval_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS],
        trainval_df["is_fraud"], test_df["is_fraud"],
    )


def evaluate_xgboost_terminal_grouped(xgboost_weight: float = 0.7) -> dict:
    """Conservative evaluation (item 4): trains a FRESH XGBoost model (same
    hyperparameters as train_xgboost.build_model, same scale_pos_weight
    methodology) and a fresh evaluation-only Isolation Forest, both fit
    ONLY on terminals in the train partition, then reports standalone
    XGBoost and blended-ensemble metrics on terminals the training data
    never saw at all - not just different rows.

    This does NOT reuse the production XGBoost model, because that model
    was trained on the row-random split's trainval rows, which can and do
    share terminal_ids with row-random X_test - comparing it against a
    terminal-grouped test set would mix the two split strategies and the
    comparison would not isolate what group-awareness actually changes.
    A full retrain, on a comparable split size, is the only fair way to
    answer "does performance hold up on terminals never seen in training."
    """
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "alerts"))
    from alert_tiers import is_alertable, tier_for_probability  # noqa: E402
    from xgboost import XGBClassifier

    X_trainval, X_test, y_trainval, y_test = _terminal_group_split()

    positives = int(y_trainval.sum())
    negatives = int((len(y_trainval) - positives))
    scale_pos_weight = negatives / positives if positives else 1.0

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_trainval, y_trainval)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_default, average="binary", zero_division=0
    )

    fraud_probability, _, _, _ = _blended_fraud_probability_on_test(
        model, X_trainval, y_trainval, X_test, xgboost_weight
    )
    y_pred_alertable = fraud_probability.apply(lambda p: int(is_alertable(tier_for_probability(p))))
    blend_precision, blend_recall, blend_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_alertable, average="binary", zero_division=0
    )

    return {
        "split": "terminal-grouped (GroupShuffleSplit on terminal_id, no shared terminals between "
                 "train and test)",
        "n_test_rows": len(y_test),
        "n_test_fraud": int(y_test.sum()),
        "xgboost_standalone_default_threshold": {
            "precision": xgb_precision, "recall": xgb_recall, "f1": xgb_f1,
            "roc_auc": roc_auc_score(y_test, y_prob),
        },
        "blended_is_alertable": {
            "precision": blend_precision, "recall": blend_recall, "f1": blend_f1,
            "roc_auc": roc_auc_score(y_test, fraud_probability),
        },
    }


if __name__ == "__main__":
    print("Run train_xgboost.train() first, then pass the returned model to evaluate_xgboost(), "
          "or load the registered XGBoost model via mlflow_registry.py and pass it to "
          "evaluate_blended_scoring_contract()/tier_operating_points() for the leakage-audited "
          "deployed-contract numbers. evaluate_xgboost_terminal_grouped() takes no pre-trained "
          "model - it trains its own, on a conservative group-aware split, for comparison.")
