"""Thin data-access facade over excel_reader.py.

Named db.py to keep the import surface stable for pages/components even
though the underlying store is the Excel workbook in data/, not a database.
"""
from excel_reader import load_alerts, load_fraud_scores, load_terminals


def get_alerts() -> list[dict]:
    return load_alerts()


def get_fraud_scores() -> list[dict]:
    return load_fraud_scores()


def get_terminals() -> list[dict]:
    return load_terminals()
