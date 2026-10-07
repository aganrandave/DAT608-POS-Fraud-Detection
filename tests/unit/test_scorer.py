import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scoring"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alerts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))

import pytest  # noqa: E402

from alert_tiers import is_alertable  # noqa: E402
from alert_tiers import tier_for_probability as alerts_tier_for_probability  # noqa: E402
from excel_reader import FEATURE_COLUMNS as TRAINING_FEATURE_COLUMNS  # noqa: E402
from mlflow_registry import ModelMetadataError  # noqa: E402
from schemas import ScoreResponse  # noqa: E402
from scorer import FEATURE_COLUMNS as SCORER_FEATURE_COLUMNS  # noqa: E402
from scorer import FraudScorer, ModelsNotReadyError, derive_amount_vs_bin_avg_ratio  # noqa: E402
from tiers import tier_for_probability as scoring_tier_for_probability  # noqa: E402


SAMPLE_FEATURES = {
    "velocity_1h": 1,
    "geo_jump_km": 0.0,
    "bin_spend_rate": 1000.0,
    "terminal_reversal_count": 0,
    "amount_ngn": 1000.0,
}


class FakeXGBoostModel:
    """Deterministic stand-in for a loaded XGBClassifier. predict_proba
    returns a fixed [P(not fraud), P(fraud)] row regardless of input, so
    tests can assert on an exact known probability. predict() is also
    provided, returning the OPPOSITE of what predict_proba would round to,
    so a test that accidentally calls predict() instead of predict_proba()
    fails loudly rather than coincidentally passing."""

    def __init__(self, positive_class_probability: float):
        self._p = positive_class_probability

    def predict_proba(self, frame):
        return [[1 - self._p, self._p]]

    def predict(self, frame):
        # Deliberately wrong-looking value: if scorer.py ever regresses to
        # calling .predict() again, this makes that obvious in test output
        # rather than silently producing a plausible-looking probability.
        return [999]


class FakeIsolationForestModel:
    """Deterministic stand-in for a loaded IsolationForest. score_samples
    returns a fixed continuous value; predict() returns a sentinel that
    would fail obviously if scorer.py regressed to calling it."""

    def __init__(self, score_samples_value: float):
        self._s = score_samples_value

    def score_samples(self, frame):
        return [self._s]

    def predict(self, frame):
        return [999]


def _mock_scorer(
    xgb_probability: float = 0.5,
    xgb_threshold: float = 0.5,
    if_score: float = 0.0,
    if_min: float = -1.0,
    if_max: float = 1.0,
):
    """Builds a FraudScorer with both registry loaders patched to return
    deterministic fakes - no real MLflow server or trained model involved."""
    xgb_model = FakeXGBoostModel(xgb_probability)
    if_model = FakeIsolationForestModel(if_score)
    with (
        patch("scorer.load_xgboost_model_and_threshold", return_value=(xgb_model, xgb_threshold)),
        patch(
            "scorer.load_isolation_forest_model_and_bounds",
            return_value=(if_model, if_min, if_max),
        ),
    ):
        return FraudScorer()


def test_tier_thresholds_are_consistent_between_scoring_and_alerts():
    for probability in [0.0, 0.1, 0.29, 0.3, 0.59, 0.6, 0.84, 0.85, 0.99]:
        assert scoring_tier_for_probability(probability) == alerts_tier_for_probability(probability)


def test_critical_tier_at_high_probability():
    assert scoring_tier_for_probability(0.95) == "critical"


def test_low_tier_at_low_probability():
    assert scoring_tier_for_probability(0.05) == "low"


def test_is_alertable_excludes_low_tier():
    assert is_alertable("low") is False
    assert is_alertable("medium") is True
    assert is_alertable("critical") is True


def test_scorer_feature_columns_match_what_the_models_were_trained_on():
    # Regression guard: scoring/scorer.py's FEATURE_COLUMNS silently drifted
    # from models/excel_reader.py's after amount_ngn was added to training,
    # which crashed every /score call with an XGBoost feature-mismatch error.
    assert SCORER_FEATURE_COLUMNS == TRAINING_FEATURE_COLUMNS


def test_derive_amount_vs_bin_avg_ratio_normal_case():
    assert derive_amount_vs_bin_avg_ratio(amount_ngn=6000.0, bin_spend_rate=3000.0) == 2.0


def test_derive_amount_vs_bin_avg_ratio_zero_bin_spend_rate_returns_zero():
    assert derive_amount_vs_bin_avg_ratio(amount_ngn=5000.0, bin_spend_rate=0.0) == 0.0


def test_fraud_scorer_survives_cold_boot_with_no_registered_models():
    # Regression guard: FraudScorer.__init__ used to call the registry
    # loader directly, which raised and crashed the whole FastAPI process on
    # a fresh docker-compose boot with an empty mlruns volume - the service
    # never even started. It must construct cleanly and report not-ready.
    with (
        patch(
            "scorer.load_xgboost_model_and_threshold",
            side_effect=RuntimeError("no versions registered"),
        ),
        patch("scorer.load_isolation_forest_model_and_bounds"),
    ):
        scorer = FraudScorer()

    assert scorer.is_ready is False
    assert scorer.load_error is not None

    with pytest.raises(ModelsNotReadyError):
        scorer.score(SAMPLE_FEATURES)


def test_fraud_scorer_fails_clearly_when_threshold_metadata_is_missing():
    # A model that loaded but is missing its required sibling metadata
    # artifact is a real deployment mistake, not a normal cold-boot gap -
    # ModelMetadataError must still land as "not ready", with a message
    # that says why, not a silent fallback to some default threshold.
    with (
        patch(
            "scorer.load_xgboost_model_and_threshold",
            side_effect=ModelMetadataError(
                "pos-fraud-xgboost/Production is registered but "
                "decision_threshold.json is missing or unreadable"
            ),
        ),
        patch("scorer.load_isolation_forest_model_and_bounds"),
    ):
        scorer = FraudScorer()

    assert scorer.is_ready is False
    assert "decision_threshold.json" in scorer.load_error

    with pytest.raises(ModelsNotReadyError):
        scorer.score(SAMPLE_FEATURES)


def test_xgboost_score_uses_positive_class_probability_not_predict():
    # The core correctness fix: xgboost_score must come from predict_proba's
    # positive-class column, and must NOT equal what FakeXGBoostModel.predict
    # would return (999, a sentinel chosen specifically to be an impossible
    # probability) - that would mean scorer.py regressed to calling predict().
    scorer = _mock_scorer(xgb_probability=0.37, xgb_threshold=0.5, if_score=0.0, if_min=-1.0, if_max=1.0)
    result = scorer.score(SAMPLE_FEATURES)

    assert result["xgboost_score"] == 0.37
    assert 0.0 <= result["xgboost_score"] <= 1.0
    assert result["xgboost_score"] != 999


def test_xgboost_decision_below_threshold_is_false():
    scorer = _mock_scorer(xgb_probability=0.49, xgb_threshold=0.5)
    result = scorer.score(SAMPLE_FEATURES)
    assert result["xgboost_decision"] is False


def test_xgboost_decision_at_threshold_is_true():
    # >= , not > - a probability exactly equal to the tuned threshold counts
    # as a positive decision (matches select_threshold()'s own convention in
    # train_xgboost.py, which is built on precision_recall_curve's
    # >= semantics).
    scorer = _mock_scorer(xgb_probability=0.50, xgb_threshold=0.5)
    result = scorer.score(SAMPLE_FEATURES)
    assert result["xgboost_decision"] is True


def test_xgboost_decision_above_threshold_is_true():
    scorer = _mock_scorer(xgb_probability=0.51, xgb_threshold=0.5)
    result = scorer.score(SAMPLE_FEATURES)
    assert result["xgboost_decision"] is True


def test_xgboost_decision_threshold_is_exposed_and_matches_loaded_value():
    scorer = _mock_scorer(xgb_probability=0.9, xgb_threshold=0.237)
    result = scorer.score(SAMPLE_FEATURES)
    assert result["xgboost_decision_threshold"] == 0.237


def test_isolation_forest_score_is_continuous_not_binary():
    # Three different raw score_samples values, strictly ordered within the
    # training min/max range, must produce three distinct normalized scores
    # - not collapse to {0.0, 1.0} the way the old predict()-based binary
    # implementation did.
    scorer_a = _mock_scorer(if_score=-0.8, if_min=-1.0, if_max=1.0)
    scorer_b = _mock_scorer(if_score=0.0, if_min=-1.0, if_max=1.0)
    scorer_c = _mock_scorer(if_score=0.8, if_min=-1.0, if_max=1.0)

    score_a = scorer_a.score(SAMPLE_FEATURES)["isolation_forest_score"]
    score_b = scorer_b.score(SAMPLE_FEATURES)["isolation_forest_score"]
    score_c = scorer_c.score(SAMPLE_FEATURES)["isolation_forest_score"]

    assert score_a > score_b > score_c  # lower raw score_samples = more anomalous = higher score
    assert len({score_a, score_b, score_c}) == 3


def test_isolation_forest_score_clips_outside_training_range():
    # A live transaction more anomalous than anything seen in training
    # clips to 1.0, not an out-of-range value above it.
    very_anomalous = _mock_scorer(if_score=-5.0, if_min=-1.0, if_max=1.0)
    very_normal = _mock_scorer(if_score=5.0, if_min=-1.0, if_max=1.0)

    assert very_anomalous.score(SAMPLE_FEATURES)["isolation_forest_score"] == 1.0
    assert very_normal.score(SAMPLE_FEATURES)["isolation_forest_score"] == 0.0


def test_isolation_forest_score_handles_degenerate_training_range():
    # score_min == score_max (every training score_samples value identical)
    # must not raise a division-by-zero.
    scorer = _mock_scorer(if_score=0.0, if_min=0.5, if_max=0.5)
    result = scorer.score(SAMPLE_FEATURES)
    assert result["isolation_forest_score"] == 0.0


def test_fraud_probability_blends_continuous_values_at_documented_weight():
    # XGBOOST_WEIGHT is 0.7 (see scorer.py) - verify the exact blended value
    # with known, non-trivial inputs, confirming no binarization happened
    # anywhere upstream of the blend.
    scorer = _mock_scorer(xgb_probability=0.8, xgb_threshold=0.5, if_score=-1.0, if_min=-1.0, if_max=1.0)
    result = scorer.score(SAMPLE_FEATURES)

    # if_score=-1.0 with min=-1.0, max=1.0 -> isolation_forest_score = 1.0
    expected = round(0.7 * 0.8 + 0.3 * 1.0, 4)
    assert result["fraud_probability"] == expected
    assert result["isolation_forest_score"] == 1.0


def test_alert_tier_follows_blended_probability_not_xgboost_decision():
    # alert_tier is driven by fraud_probability (the blend), and is
    # conceptually independent of xgboost_decision/xgboost_decision_threshold
    # - a low xgboost_decision_threshold that makes xgboost_decision True
    # must not by itself push a low blended probability into a higher tier.
    scorer = _mock_scorer(xgb_probability=0.2, xgb_threshold=0.1, if_score=1.0, if_min=-1.0, if_max=1.0)
    result = scorer.score(SAMPLE_FEATURES)

    assert result["xgboost_decision"] is True  # 0.2 >= 0.1
    assert result["alert_tier"] == scoring_tier_for_probability(result["fraud_probability"])
    assert result["fraud_probability"] < 0.3  # stays low tier despite xgboost_decision=True


def test_end_to_end_score_request_with_deterministic_mocks():
    # Full /score contract, deterministic mocks standing in for MLflow -
    # every field FastAPI's ScoreResponse schema requires must be present
    # and correctly typed.
    scorer = _mock_scorer(xgb_probability=0.91, xgb_threshold=0.42, if_score=-0.9, if_min=-1.0, if_max=1.0)
    result = scorer.score(
        {
            "velocity_1h": 12,
            "geo_jump_km": 340.5,
            "bin_spend_rate": 92000.0,
            "terminal_reversal_count": 3,
            "amount_ngn": 185000.0,
        }
    )

    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["alert_tier"], str)
    assert isinstance(result["xgboost_score"], float)
    assert isinstance(result["xgboost_decision"], bool)
    assert isinstance(result["xgboost_decision_threshold"], float)
    assert isinstance(result["isolation_forest_score"], float)
    assert isinstance(result["model_version"], str)
    assert result["xgboost_decision"] is True
    assert result["alert_tier"] in {"low", "medium", "high", "critical"}


def test_score_result_is_compatible_with_fastapi_response_schema():
    # scorer.score()'s dict must validate as a ScoreResponse once main.py
    # adds transaction_id/scored_at (the two fields the route layer fills in
    # - see scoring/main.py) - this would fail loudly if a field were ever
    # renamed/removed/mistyped without updating schemas.py to match.
    scorer = _mock_scorer(xgb_probability=0.6, xgb_threshold=0.5, if_score=0.1, if_min=-1.0, if_max=1.0)
    result = scorer.score(SAMPLE_FEATURES)
    result["transaction_id"] = "txn-test-1"
    result["scored_at"] = "2026-01-01T00:00:00+00:00"

    validated = ScoreResponse(**result)
    assert validated.xgboost_decision is True
    assert validated.xgboost_decision_threshold == 0.5
