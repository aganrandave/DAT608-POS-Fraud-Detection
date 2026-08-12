"""One-off hyperparameter sweep for the XGBoost fraud classifier.

Not run as part of normal training - this is how the hyperparameters
baked into train_xgboost.py.build_model() were chosen, kept here for
reproducibility/transparency rather than being a magic number someone
has to take on faith.

Selection metric is mean cross-validated average precision (PR-AUC), not
F1 at a fixed threshold: train_xgboost.py tunes its own decision
threshold separately (see select_threshold()), so the model that ranks
positives best - not the one that happens to score highest at the
default 0.5 threshold - is what we actually want here. Cross-validation
(not the held-out test set) is used for selection so this sweep cannot
leak into the final reported holdout metric in train_xgboost.py.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from excel_reader import FEATURE_COLUMNS
from train_xgboost import compute_scale_pos_weight, load_dataset

N_SPLITS = 5
RANDOM_SEED = 42

PARAM_GRID = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [200, 300, 400],
}


def cv_score(df: pd.DataFrame, scale_pos_weight: float, params: dict) -> dict:
    X, y = df[FEATURE_COLUMNS], df["is_fraud"]
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    aps, f1s = [], []
    for train_idx, test_idx in kfold.split(X, y):
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
            **params,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        aps.append(average_precision_score(y.iloc[test_idx], y_prob))
        f1s.append(f1_score(y.iloc[test_idx], (y_prob >= 0.5).astype(int), zero_division=0))

    return {"cv_average_precision_mean": float(np.mean(aps)), "cv_f1_mean_default_threshold": float(np.mean(f1s))}


def sweep() -> list[dict]:
    df = load_dataset()
    scale_pos_weight = compute_scale_pos_weight(df["is_fraud"])

    keys = list(PARAM_GRID.keys())
    results = []
    for combo in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, combo))
        scores = cv_score(df, scale_pos_weight, params)
        results.append({**params, **scores})
        print(params, scores)

    results.sort(key=lambda r: r["cv_average_precision_mean"], reverse=True)
    return results


if __name__ == "__main__":
    results = sweep()
    print("\nTop 5 by mean CV average precision:")
    for r in results[:5]:
        print(r)
