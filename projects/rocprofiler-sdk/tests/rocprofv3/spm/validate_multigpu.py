#!/usr/bin/env python3

import sys
import pytest


def test_validate_spm_multigpu_stream_id(spm_json_data):
    data = spm_json_data["rocprofiler-sdk-tool"]
    spm_data = data["callback_records"]["spm_counter_collection"]
    kernel_symbols = data.get("kernel_symbols", [])

    found_subtract = False
    for spm_record in spm_data:
        assert "stream_id" in spm_record, "stream_id missing from SPM record"
        assert "handle" in spm_record["stream_id"]

        dispatch_info = spm_record["dispatch_data"]["dispatch_info"]
        kernel_id = dispatch_info.get("kernel_id")
        if isinstance(kernel_id, dict):
            kernel_id = kernel_id.get("handle")
        kernel_name = kernel_symbols[kernel_id]["formatted_kernel_name"]
        if "subtract_kernel" in kernel_name:
            found_subtract = True
            assert (
                spm_record["stream_id"]["handle"] > 0
            ), f"stream_id should be non-zero for {kernel_name}"

    assert found_subtract, "No subtract_kernel dispatches found in SPM data"


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
        f"SELECT DISTINCT K.stream_id FROM {spm_view} SC "
        f"INNER JOIN rocpd_kernel_dispatch K ON K.event_id = SC.event_id "
        f"AND K.guid = SC.guid "
        f"WHERE K.stream_id > 0"
    ).fetchall()

    assert len(rows) > 0, "No SPM kernel dispatches with non-zero stream_id in rocpd"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
