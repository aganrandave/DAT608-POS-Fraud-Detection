"""Appends generated transactions to data/transactions_raw.xlsx.

Follows the shared repo-wide Excel read/write pattern: open with openpyxl,
append a row, save — guarded by a cross-process file lock so the producer
and any concurrent readers never corrupt the workbook.
"""
import openpyxl
from filelock import FileLock

import config

COLUMNS = [
    "transaction_id",
    "terminal_id",
    "merchant_id",
    "card_bin",
    "amount_ngn",
    "state",
    "lga",
    "latitude",
    "longitude",
    "timestamp",
    "is_fraud",
    "fraud_type",
]


def append_transaction(transaction: dict, xlsx_path: str = config.TRANSACTIONS_XLSX) -> None:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        ws.append([transaction[col] for col in COLUMNS])
        wb.save(xlsx_path)


def append_transactions(transactions: list[dict], xlsx_path: str = config.TRANSACTIONS_XLSX) -> None:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        for transaction in transactions:
            ws.append([transaction[col] for col in COLUMNS])
        wb.save(xlsx_path)
