"""Reads data/alerts.xlsx and data/fraud_scores.xlsx for the Streamlit dashboard."""
import openpyxl
from filelock import FileLock

ALERTS_XLSX = "data/alerts.xlsx"
FRAUD_SCORES_XLSX = "data/fraud_scores.xlsx"
TERMINALS_XLSX = "data/reference/terminals.xlsx"
MERCHANTS_XLSX = "data/reference/merchants.xlsx"


def _read_rows(xlsx_path: str) -> list[dict]:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[1:]]


def load_alerts() -> list[dict]:
    return _read_rows(ALERTS_XLSX)


def load_fraud_scores() -> list[dict]:
    return _read_rows(FRAUD_SCORES_XLSX)


def load_terminals() -> list[dict]:
    return _read_rows(TERMINALS_XLSX)


def load_merchants() -> list[dict]:
    return _read_rows(MERCHANTS_XLSX)
