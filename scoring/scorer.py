"""Loads the registered XGBoost + Isolation Forest models and scores requests.

The two model scores are blended into a single fraud_probability and mapped
to an alert tier consumed by the ksqlDB alert logic in alerts/.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from mlflow_registry import (  # noqa: E402
    ModelMetadataError,
    load_isolation_forest_model_and_bounds,
    load_xgboost_model_and_threshold,
)

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


class ModelsNotReadyError(RuntimeError):
    """Raised when /score is called before both models are registered."""


class FraudScorer:
    def __init__(self):
        self.xgboost_model = None
        self.xgboost_decision_threshold: float | None = None
        self.isolation_forest_model = None
        self.isolation_forest_score_min: float | None = None
        self.isolation_forest_score_max: float | None = None
        self.load_error: str | None = None
        self.try_load()

    def try_load(self) -> bool:
        """Attempt to (re)load both models, and each one's required sibling
        metadata (the XGBoost decision threshold / Isolation Forest score
        bounds - see mlflow_registry.py), from the MLflow registry. Safe to
        call repeatedly - a fresh docker-compose boot has an empty mlruns
        volume with nothing registered yet, and this app used to call the
        registry loader straight from __init__, which crashed the whole
        FastAPI process before uvicorn ever bound the port. Retrying here
        instead lets the service start and report its own readiness
        honestly via /health, rather than never starting at all.

        A model registered WITHOUT its metadata artifact (ModelMetadataError)
        is treated the same as "not ready" here, but load_error says exactly
        why - that's a real deployment mistake, not a normal cold-boot gap,
        and /health should make that distinguishable rather than silently
        masking it as "still warming up"."""
        try:
            self.xgboost_model, self.xgboost_decision_threshold = load_xgboost_model_and_threshold()
            (
                self.isolation_forest_model,
                self.isolation_forest_score_min,
                self.isolation_forest_score_max,
            ) = load_isolation_forest_model_and_bounds()
            self.load_error = None
            return True
        except ModelMetadataError as exc:
            self.load_error = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001 - any other registry failure just means "not ready yet"
            self.load_error = str(exc)
            return False

    @property
    def is_ready(self) -> bool:
        return self.xgboost_model is not None and self.isolation_forest_model is not None

    def _feature_frame(self, features: dict):
        import pandas as pd

        row = dict(features)
        row["amount_vs_bin_avg_ratio"] = derive_amount_vs_bin_avg_ratio(
            row["amount_ngn"], row["bin_spend_rate"]
        )
        return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    def score(self, features: dict) -> dict:
        if not self.is_ready and not self.try_load():
            raise ModelsNotReadyError(self.load_error or "models not yet registered")

        frame = self._feature_frame(features)

        # Positive-class probability, NOT model.predict() - predict() on an
        # XGBClassifier returns a binary 0/1 label, which would silently
        # collapse fraud_probability to a near-binary value. This is the
        # continuous value blended below; it is never binarized before the
        # blend, only after, via xgboost_decision.
        xgboost_probability = float(self.xgboost_model.predict_proba(frame)[0][1])
        xgboost_decision = bool(xgboost_probability >= self.xgboost_decision_threshold)

        # score_samples() is continuous (lower = more anomalous); predict()
        # would only give the binary contamination-threshold outlier flag.
        # Normalize using the training-fitted score_min/score_max bounds
        # (see train_isolation_forest.py), clipped since a live transaction
        # can fall outside that training-time range in either direction.
        raw_if_score = float(self.isolation_forest_model.score_samples(frame)[0])
        score_range = self.isolation_forest_score_max - self.isolation_forest_score_min
        if score_range <= 0:
            # Degenerate training distribution (every training score_samples
            # value identical) - there's no real range to normalize against.
            isolation_forest_score = 0.0
        else:
            isolation_forest_score = (self.isolation_forest_score_max - raw_if_score) / score_range
            isolation_forest_score = max(0.0, min(1.0, isolation_forest_score))

        # Continuous blend - fraud_probability is never built from a
        # binarized XGBoost output.
        fraud_probability = (
            XGBOOST_WEIGHT * xgboost_probability + (1 - XGBOOST_WEIGHT) * isolation_forest_score
        )

        return {
            "fraud_probability": round(fraud_probability, 4),
            # alert_tier (critical/high/medium/low, see tiers.py) is an
            # OPERATIONAL policy threshold applied to the blended
            # fraud_probability - deliberately a separate concept from
            # xgboost_decision_threshold above, which is a MODEL VALIDATION
            # threshold selected on XGBoost's own held-out split and applies
            # only to the standalone xgboost_score. Conflating the two would
            # let a change to one silently move the other.
            "alert_tier": tier_for_probability(fraud_probability),
            # Field name kept for Excel/Kafka/ksqlDB/dashboard compatibility;
            # the value is now the real positive-class probability this
            # field was always documented (docs/data_dictionary.md) as being.
            "xgboost_score": round(xgboost_probability, 4),
            "xgboost_decision": xgboost_decision,
            "xgboost_decision_threshold": round(self.xgboost_decision_threshold, 4),
            "isolation_forest_score": round(isolation_forest_score, 4),
            "model_version": MODEL_VERSION,
        }
