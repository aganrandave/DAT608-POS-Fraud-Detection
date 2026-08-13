"""Loads the registered XGBoost + Isolation Forest models and scores requests.

The two model scores are blended into a single fraud_probability and mapped
to an alert tier consumed by the ksqlDB alert logic in alerts/.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from mlflow_registry import load_latest_model  # noqa: E402

from tiers import tier_for_probability  # noqa: E402

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Must match models/excel_reader.py's FEATURE_COLUMNS - that's what the
# registered models were actually trained on. amount_vs_bin_avg_ratio isn't
# collected directly from callers; it's derived in _feature_frame the same
# way pipeline/feature_windows.py derives it (amount_ngn / bin_spend_rate).
FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
    "amount_ngn",
    "amount_vs_bin_avg_ratio",
]

# Blend weight applied to the XGBoost probability vs. the normalized
# Isolation Forest anomaly score when computing fraud_probability.
XGBOOST_WEIGHT = 0.7


def derive_amount_vs_bin_avg_ratio(amount_ngn: float, bin_spend_rate: float) -> float:
    """amount_ngn / bin_spend_rate, guarding the same zero-history edge case
    that pipeline/feature_windows.py's Spark division silently nulls out.
    bin_spend_rate is a trailing average that normally includes the current
    transaction's own amount, so it's only ever 0 for a caller-supplied
    placeholder - treat that as "no signal" rather than raising.
    """
    if bin_spend_rate <= 0:
        return 0.0
    return amount_ngn / bin_spend_rate


class FraudScorer:
    def __init__(self):
        self.xgboost_model = load_latest_model("pos-fraud-xgboost")
        self.isolation_forest_model = load_latest_model("pos-fraud-isolation-forest")

    def _feature_frame(self, features: dict):
        import pandas as pd

        row = dict(features)
        row["amount_vs_bin_avg_ratio"] = derive_amount_vs_bin_avg_ratio(
            row["amount_ngn"], row["bin_spend_rate"]
        )
        return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    def score(self, features: dict) -> dict:
        frame = self._feature_frame(features)

        xgboost_score = float(self.xgboost_model.predict(frame)[0])
        # Isolation Forest's decision_function is negative for anomalies;
        # normalize to a 0-1 "fraudiness" score.
        raw_if_score = float(self.isolation_forest_model.predict(frame)[0])
        isolation_forest_score = max(0.0, min(1.0, (1 - raw_if_score) / 2))

        fraud_probability = (
            XGBOOST_WEIGHT * xgboost_score + (1 - XGBOOST_WEIGHT) * isolation_forest_score
        )

        return {
            "fraud_probability": round(fraud_probability, 4),
            "alert_tier": tier_for_probability(fraud_probability),
            "xgboost_score": round(xgboost_score, 4),
            "isolation_forest_score": round(isolation_forest_score, 4),
            "model_version": MODEL_VERSION,
        }
