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
from scorer import FEATURE_COLUMNS as SCORER_FEATURE_COLUMNS  # noqa: E402
from scorer import FraudScorer, ModelsNotReadyError, derive_amount_vs_bin_avg_ratio  # noqa: E402
from tiers import tier_for_probability as scoring_tier_for_probability  # noqa: E402


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
    # Regression guard: FraudScorer.__init__ used to call load_latest_model()
    # directly, which raised and crashed the whole FastAPI process on a
    # fresh docker-compose boot with an empty mlruns volume - the service
    # never even started. It must construct cleanly and report not-ready.
    with patch("scorer.load_latest_model", side_effect=RuntimeError("no versions registered")):
        scorer = FraudScorer()

    assert scorer.is_ready is False
    assert scorer.load_error is not None

    with pytest.raises(ModelsNotReadyError):
        scorer.score(
            {
                "velocity_1h": 1,
                "geo_jump_km": 0.0,
                "bin_spend_rate": 1000.0,
                "terminal_reversal_count": 0,
                "amount_ngn": 1000.0,
            }
        )
