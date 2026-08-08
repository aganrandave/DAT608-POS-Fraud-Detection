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

FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
]

# Blend weight applied to the XGBoost probability vs. the normalized
# Isolation Forest anomaly score when computing fraud_probability.
XGBOOST_WEIGHT = 0.7


class FraudScorer:
    def __init__(self):
        self.xgboost_model = load_latest_model("pos-fraud-xgboost")
        self.isolation_forest_model = load_latest_model("pos-fraud-isolation-forest")

    def _feature_frame(self, features: dict):
        import pandas as pd

        return pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

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
