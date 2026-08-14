"""Consumes the pos-fraud-alerts topic (materialized by alert_stream.sql)
and appends each alert to data/alerts.xlsx.
"""
import uuid
from datetime import datetime, timezone

import openpyxl
from filelock import FileLock

ALERTS_XLSX = "data/alerts.xlsx"

COLUMNS = [
    "alert_id",
    "transaction_id",
    "terminal_id",
    "merchant_id",
    "card_bin",
    "fraud_probability",
    "alert_tier",
    "state",
    "created_at",
]


def append_alert(alert: dict, xlsx_path: str = ALERTS_XLSX) -> dict:
    alert = {
        "alert_id": alert.get("alert_id") or str(uuid.uuid4()),
        "created_at": alert.get("created_at") or datetime.now(timezone.utc).isoformat(),
        **alert,
    }

    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        # .get(), not alert[col]: ksqlDB's JSON serializer omits a field
        # entirely when its value is NULL (e.g. merchant_id/terminal_name
        # when the LEFT JOIN in 03_create_alert_stream.sql finds no
        # matching reference row) rather than emitting it as null. A
        # missing key here used to raise KeyError and crash this
        # long-running consumer process outright - one alert with any
        # null field took down alert ingestion for every alert after it,
        # since nothing restarts this container.
        ws.append([alert.get(col) for col in COLUMNS])
        wb.save(xlsx_path)

    return alert
