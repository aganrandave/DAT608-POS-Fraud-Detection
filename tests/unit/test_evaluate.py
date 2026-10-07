"""Protects the leakage-sensitive parts of models/evaluate.py: that the
Isolation Forest used for held-out evaluation is fit ONLY on the train
partition (never the test rows it's then evaluated against), and that the
conservative group-aware split never shares a terminal_id between train
and test. See evaluate.py's module docstring and
evaluate_blended_scoring_contract's docstring for the real leakage bug
this guards against.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    FEATURE_COLUMNS,
    _blend_probability,
    _blended_fraud_probability_on_test,
    _fit_evaluation_only_isolation_forest,
    _held_out_split,
    _terminal_group_split,
)


class FakeXGBoostModel:
    """Deterministic stand-in - identical pattern to tests/unit/test_scorer.py's
    fakes, reused here so evaluate.py's pipeline can be exercised without
    actually training XGBoost."""

    def __init__(self, positive_class_probability: float = 0.5):
        self._p = positive_class_probability

    def predict_proba(self, frame):
        n = len(frame)
        return np.column_stack([np.full(n, 1 - self._p), np.full(n, self._p)])


def _synthetic_trainval_test(n_trainval=120, n_test=30, seed=0):
    """Small, fast synthetic dataset with the same columns
    _fit_evaluation_only_isolation_forest / _blended_fraud_probability_on_test
    expect, avoiding a dependency on the full committed data for these
    specific mechanism tests."""
    rng = np.random.RandomState(seed)

    def make(n, offset):
        idx = pd.RangeIndex(offset, offset + n)
        X = pd.DataFrame(rng.rand(n, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS, index=idx)
        y = pd.Series((rng.rand(n) < 0.1).astype(int), index=idx)
        return X, y

    X_trainval, y_trainval = make(n_trainval, 0)
    X_test, y_test = make(n_test, n_trainval)
    return X_trainval, X_test, y_trainval, y_test


def test_held_out_split_train_and_test_are_disjoint():
    # Real committed data - this is a plain pandas split, fast regardless
    # of row count.
    X_trainval, X_test, y_trainval, y_test = _held_out_split()
    assert set(X_trainval.index).isdisjoint(set(X_test.index))
    assert len(X_trainval) + len(X_test) == len(y_trainval) + len(y_test)
    assert len(X_test) > 0


def test_terminal_group_split_shares_no_terminal_between_train_and_test():
    # The real leakage risk this split exists to rule out: not just
    # disjoint ROWS (row-random already guarantees that trivially), but
    # disjoint TERMINALS - the thing correlated records cluster around.
    X_trainval, X_test, y_trainval, y_test = _terminal_group_split()
    # terminal_id isn't in FEATURE_COLUMNS, so re-derive it from the same
    # source the split function used, keyed by index, to check the actual
    # grouping property rather than just trusting the split's own assertion.
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    from excel_reader import load_training_frame

    df = pd.DataFrame(load_training_frame())
    trainval_terminals = set(df.loc[X_trainval.index, "terminal_id"])
    test_terminals = set(df.loc[X_test.index, "terminal_id"])
    assert trainval_terminals.isdisjoint(test_terminals)
    assert len(test_terminals) > 0


def test_evaluation_only_isolation_forest_bounds_come_from_trainval_only():
    # Construct trainval/test where test contains values far outside
    # trainval's range - if the bounds leaked in test data, score_max/min
    # would reflect test's extreme values. They must not.
    X_trainval, X_test, y_trainval, y_test = _synthetic_trainval_test()
    # Make every test row's features enormous, far outside anything in
    # trainval, so if they influenced score_min/score_max the bounds would
    # visibly shift to cover them.
    X_test_extreme = X_test.copy() + 1000.0

    _, score_min, score_max = _fit_evaluation_only_isolation_forest(X_trainval, y_trainval)

    # Refit again including the "extreme" test rows pretending they were
    # part of trainval, to get the (wrong) bounds a leaky implementation
    # would produce, and confirm the real function's bounds do NOT match.
    leaky_X = pd.concat([X_trainval, X_test_extreme])
    leaky_y = pd.concat([y_trainval, pd.Series(0, index=X_test_extreme.index)])
    _, leaky_min, leaky_max = _fit_evaluation_only_isolation_forest(leaky_X, leaky_y)

    assert (score_min, score_max) != (leaky_min, leaky_max)


def test_blended_fraud_probability_asserts_on_overlapping_indices():
    # If a caller ever passes an X_test that overlaps X_trainval (the
    # literal leakage condition), this must fail loudly, not silently
    # produce a number.
    X_trainval, X_test, y_trainval, y_test = _synthetic_trainval_test()
    contaminated_X_test = pd.concat([X_test, X_trainval.iloc[[0]]])  # one overlapping row

    with pytest.raises(AssertionError):
        _blended_fraud_probability_on_test(
            FakeXGBoostModel(0.5), X_trainval, y_trainval, contaminated_X_test
        )


def test_blended_fraud_probability_runs_clean_on_disjoint_split():
    X_trainval, X_test, y_trainval, y_test = _synthetic_trainval_test()
    fraud_probability, eval_if_model, score_min, score_max = _blended_fraud_probability_on_test(
        FakeXGBoostModel(0.9), X_trainval, y_trainval, X_test
    )
    assert len(fraud_probability) == len(X_test)
    assert (fraud_probability >= 0).all() and (fraud_probability <= 1).all()
    assert score_min <= score_max


def test_blend_probability_matches_scorer_formula():
    # scoring/scorer.py's score() computes:
    #   isolation_forest_score = clip((max - raw) / (max - min), 0, 1)
    #   fraud_probability = weight * xgb + (1 - weight) * isolation_forest_score
    # _blend_probability must reproduce that exactly, or the "evaluates the
    # deployed contract" claim in evaluate.py's docstrings would be false.
    result = _blend_probability(
        xgb_prob=np.array([0.8]), raw_if=np.array([-1.0]), if_score_min=-1.0, if_score_max=1.0,
        xgboost_weight=0.7,
    )
    # raw=-1.0 == score_min -> isolation_forest_score = (1 - (-1)) / (1 - (-1)) = 1.0
    expected = 0.7 * 0.8 + 0.3 * 1.0
    assert result[0] == pytest.approx(expected)


def test_tier_operating_points_structure():
    # End-to-end against real committed data but a fake XGBoost model (no
    # real training needed to verify the table's shape/invariants).
    results = evaluate.tier_operating_points(
        FakeXGBoostModel(positive_class_probability=0.4), xgboost_decision_threshold=0.5
    )
    names = [r["operating_point"] for r in results]
    assert len(results) == 4
    assert any("medium" in n for n in names)
    assert any("high" in n for n in names)
    assert any("critical" in n for n in names)
    assert any("xgboost_standalone" in n for n in names)

    n_test_total = None
    for row in results:
        assert row["n_alerted"] == row["tp"] + row["fp"]
        total = row["tp"] + row["fp"] + row["fn"] + row["tn"]
        assert total > 0
        # Every operating point is scored against the identical held-out
        # set, so the four confusion counts must sum to the same total
        # across all rows in the table.
        n_test_total = n_test_total or total
        assert total == n_test_total
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["false_positive_rate"] <= 1.0

    # Stricter thresholds must never alert MORE transactions than looser ones.
    by_name = {r["operating_point"]: r for r in results}
    medium = next(v for k, v in by_name.items() if "medium" in k)
    high = next(v for k, v in by_name.items() if "high" in k)
    critical = next(v for k, v in by_name.items() if "critical" in k)
    assert medium["n_alerted"] >= high["n_alerted"] >= critical["n_alerted"]
