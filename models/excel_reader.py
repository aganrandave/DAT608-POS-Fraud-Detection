"""Reads engineered features from data/features.xlsx for model training.

Joins against data/transactions_raw.xlsx on transaction_id to recover the
is_fraud label, since features.xlsx itself is unlabeled.
"""
import openpyxl
from filelock import FileLock

FEATURES_XLSX = "data/features.xlsx"
TRANSACTIONS_XLSX = "data/transactions_raw.xlsx"


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
