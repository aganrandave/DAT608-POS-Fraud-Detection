"""Appends scored transactions to data/fraud_scores.xlsx."""
import openpyxl
from filelock import FileLock

FRAUD_SCORES_XLSX = "data/fraud_scores.xlsx"

COLUMNS = [
    "transaction_id",
    "fraud_probability",
    "alert_tier",
    "xgboost_score",
    "isolation_forest_score",
    "model_version",
    "scored_at",
]


def append_score(score: dict, xlsx_path: str = FRAUD_SCORES_XLSX) -> None:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        ws.append([score[col] for col in COLUMNS])
        wb.save(xlsx_path)
