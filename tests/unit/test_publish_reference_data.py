import os
import sys
import tempfile
from unittest.mock import MagicMock

import openpyxl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alerts"))

from publish_reference_data import publish, read_rows  # noqa: E402


def test_read_rows_parses_header_and_rows_into_dicts():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "terminals.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["terminal_id", "state", "is_active"])
        ws.append(["TRM00001", "Lagos", True])
        ws.append(["TRM00002", "Kano", False])
        wb.save(path)

        rows = read_rows(path)

        assert rows == [
            {"terminal_id": "TRM00001", "state": "Lagos", "is_active": True},
            {"terminal_id": "TRM00002", "state": "Kano", "is_active": False},
        ]


def test_publish_sends_each_row_keyed_and_flushes():
    producer = MagicMock()
    rows = [{"terminal_id": "TRM00001", "state": "Lagos"}, {"terminal_id": "TRM00002", "state": "Kano"}]

    count = publish(producer, "terminal-reference", "terminal_id", rows)

    assert count == 2
    assert producer.send.call_count == 2
    producer.send.assert_any_call("terminal-reference", key="TRM00001", value=rows[0])
    producer.flush.assert_called_once()
