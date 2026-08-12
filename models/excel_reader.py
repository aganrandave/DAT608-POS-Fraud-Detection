"""Reads engineered features from data/features.xlsx for model training.

Joins against data/transactions_raw.xlsx on transaction_id to recover the
is_fraud label, since features.xlsx itself is unlabeled.
"""
import csv
import json
import os

import openpyxl
from filelock import FileLock

FEATURES_XLSX = "data/features.xlsx"
TRANSACTIONS_XLSX = "data/transactions_raw.xlsx"

# Single source of truth for the modeling feature set - train_xgboost.py,
# train_isolation_forest.py, and synthetic_augmentation.py all import this
# rather than redefining it, so the three stay in sync. Matches the columns
# produced by pipeline/feature_windows.py / batch_feature_engineering.py.
FEATURE_COLUMNS = [
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
    "amount_ngn",
    "amount_vs_bin_avg_ratio",
]

SYNTHETIC_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic_output")
SYNTHETIC_FEATURES_CSV = os.path.join(SYNTHETIC_OUTPUT_DIR, "synthetic_features.csv")
SYNTHETIC_VALIDATION_RESULTS = os.path.join(SYNTHETIC_OUTPUT_DIR, "validation_results.json")


def _read_rows(xlsx_path: str) -> list[dict]:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[1:]]


def load_features(xlsx_path: str = FEATURES_XLSX) -> list[dict]:
    return _read_rows(xlsx_path)


def load_training_frame(
    features_path: str = FEATURES_XLSX, transactions_path: str = TRANSACTIONS_XLSX
) -> list[dict]:
    """Return feature rows enriched with the is_fraud label from transactions_raw.xlsx."""
    features = load_features(features_path)
    transactions = _read_rows(transactions_path)
    labels = {row["transaction_id"]: bool(row["is_fraud"]) for row in transactions}

    for row in features:
        row["is_fraud"] = labels.get(row["transaction_id"], False)

    return features


def load_approved_synthetic_rows() -> list[dict]:
    """CTGAN-synthesized minority-class rows from
    synthetic_output/synthetic_features.csv - but ONLY if both:

      1. USE_SYNTHETIC_AUGMENTATION=true is set in the environment (opt-in,
         default behaviour is unchanged), and
      2. validate_synthetic_features.py's own gate actually approved it
         (synthetic_output/validation_results.json's approved_for_training
         is true - see that script's is_approved_for_training() for the
         exact criteria).

    If augmentation is requested but the gate rejected it (or validation
    was never run), this prints why and returns [] - it never fabricates
    rows into a "pass" that didn't happen. Synthetic rows carry no
    transaction_id/terminal_id/card_bin/timestamp (CTGAN never generated
    those - see synthetic_augmentation.py), so callers get feature+label
    rows only, never disguised as real records.

    Deliberately returns just the synthetic rows rather than a combined
    frame: callers (train_xgboost.py) need to add these ONLY to their
    training pool and keep the held-out test split real-only, so that an
    augmented run and a real-only run are evaluated on identical,
    comparable, real-world holdout data - see train_xgboost.py's train()."""
    if os.getenv("USE_SYNTHETIC_AUGMENTATION", "false").lower() != "true":
        return []

    if not os.path.exists(SYNTHETIC_VALIDATION_RESULTS):
        print("USE_SYNTHETIC_AUGMENTATION=true but no validation_results.json found - "
              "run models/validate_synthetic_features.py first. No synthetic rows added.")
        return []

    with open(SYNTHETIC_VALIDATION_RESULTS) as f:
        validation_results = json.load(f)

    if not validation_results.get("approved_for_training"):
        reasons = "; ".join(validation_results.get("gate_failure_reasons", []))
        print(f"USE_SYNTHETIC_AUGMENTATION=true but the validation gate did NOT approve this "
              f"synthetic set ({reasons}). No synthetic rows added.")
        return []

    with open(SYNTHETIC_FEATURES_CSV, newline="") as f:
        synthetic_rows = [
            {
                "transaction_id": None,
                "terminal_id": None,
                "card_bin": None,
                "timestamp": None,
                "is_fraud": bool(int(row["is_fraud"])),
                **{col: float(row[col]) for col in FEATURE_COLUMNS},
            }
            for row in csv.DictReader(f)
        ]

    print(f"USE_SYNTHETIC_AUGMENTATION=true and gate approved - {len(synthetic_rows)} "
          f"validated synthetic rows available for the training pool.")
    return synthetic_rows


def load_augmented_training_frame(
    features_path: str = FEATURES_XLSX, transactions_path: str = TRANSACTIONS_XLSX
) -> list[dict]:
    """load_training_frame() plus load_approved_synthetic_rows(), for
    callers that don't need to keep a real-only test split separate (e.g.
    train_isolation_forest.py, which is unsupervised and has no held-out
    labelled test set in the same sense)."""
    real_rows = load_training_frame(features_path, transactions_path)
    return real_rows + load_approved_synthetic_rows()
