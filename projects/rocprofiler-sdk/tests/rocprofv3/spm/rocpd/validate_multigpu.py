#!/usr/bin/env python3

import sys
import pytest


def _find_table_or_view(conn, base_name):
    for typ in ("view", "table"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name LIKE ?",
            (typ, f"{base_name}%"),
        ).fetchone()
        if row:
            return row[0]
    return None


def test_validate_spm_multigpu_rocpd_stream_id(rocpd_data):
    spm_view = _find_table_or_view(rocpd_data, "spm_counters")
    assert spm_view is not None

    rows = rocpd_data.execute(
        f"SELECT DISTINCT K.stream_id FROM {spm_view} S"
        f" INNER JOIN rocpd_kernel_dispatch K ON K.event_id = S.event_id"
        f" AND K.guid = S.guid"
        f" WHERE K.stream_id > 0"
    ).fetchall()

    assert len(rows) > 0, "No SPM kernel dispatches with non-zero stream_id in rocpd"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
