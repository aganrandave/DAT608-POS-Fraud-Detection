"""Appends engineered feature rows to data/features.xlsx.

Used from the foreachBatch sink in spark_consumer.py — each micro-batch is
collected to the driver and appended under a file lock so concurrent
micro-batches never interleave writes.
"""
import openpyxl
from filelock import FileLock

FEATURES_XLSX = "data/features.xlsx"

COLUMNS = [
    "transaction_id",
    "terminal_id",
    "card_bin",
    "amount_ngn",
    "velocity_1h",
    "geo_jump_km",
    "bin_spend_rate",
    "terminal_reversal_count",
    "amount_vs_bin_avg_ratio",
    "timestamp",
]


def append_features(rows: list[dict], xlsx_path: str = FEATURES_XLSX) -> None:
    lock_path = f"{xlsx_path}.lock"
    with FileLock(lock_path, timeout=30):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        # Spark's foreachBatch guarantees at-least-once delivery, so a
        # replayed micro-batch (e.g. after a consumer restart) can hand the
        # same transaction_id to write_batch twice - skip transaction_ids
        # already on disk so a replay doesn't duplicate rows.
        existing_ids = {row[0] for row in ws.iter_rows(min_row=2, max_col=1, values_only=True)}
        new_rows = [row for row in rows if row["transaction_id"] not in existing_ids]
        for row in new_rows:
            ws.append([row[col] for col in COLUMNS])
        if new_rows:
            wb.save(xlsx_path)


def write_batch(batch_df, batch_id: int) -> None:
    """foreachBatch sink used by spark_consumer.py's writeStream call."""
    rows = [row.asDict() for row in batch_df.collect()]
    if rows:
        append_features(rows)
