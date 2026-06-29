# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for streaming native counter CSV processing.

Covers the streaming helpers introduced to replace in-memory list
processing in rocpd_data.py and utils_profile.py.
"""

import sqlite3
import textwrap

import pytest

from utils import rocpd_data
from utils.utils_profile import (
    _aggregate_counter_csv,
    _build_rocprofv3_counter_rows,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def native_counter_csv(tmp_path):
    """Write a minimal native counter CSV and return its path.

    Mirrors the schema produced by the C++ CsvCountersWriter:
    dispatch_id,gpu_id,kernel_id,lds_per_workgroup,
    counter_id,counter_name,counter_value
    """
    csv_path = tmp_path / "0_native_counter_collection.csv"
    csv_path.write_text(
        textwrap.dedent("""\
            dispatch_id,gpu_id,kernel_id,lds_per_workgroup,counter_id,counter_name,counter_value
            0,0,1,0,10,SQ_WAVES,8.0
            0,0,1,0,11,GRBM_GUI_ACTIVE,200.0
            1,0,2,0,10,SQ_WAVES,4.0
            1,0,2,0,10,SQ_WAVES,6.0
        """),
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def kernel_trace_rows():
    """Kernel trace rows for joining with counter data."""
    return [
        {
            "Dispatch_Id": "0",
            "Correlation_Id": "21",
            "Agent_Id": "0",
            "Queue_Id": "0",
            "Thread_Id": "100",
            "Grid_Size_X": "512",
            "Grid_Size_Y": "1",
            "Grid_Size_Z": "1",
            "Kernel_Id": "1",
            "Kernel_Name": "vecAdd",
            "Workgroup_Size_X": "256",
            "Workgroup_Size_Y": "1",
            "Workgroup_Size_Z": "1",
            "LDS_Block_Size": "0",
            "Scratch_Size": "0",
            "VGPR_Count": "16",
            "Accum_VGPR_Count": "0",
            "SGPR_Count": "32",
            "Start_Timestamp": "1000",
            "End_Timestamp": "2000",
        },
        {
            "Dispatch_Id": "1",
            "Correlation_Id": "28",
            "Agent_Id": "0",
            "Queue_Id": "0",
            "Thread_Id": "100",
            "Grid_Size_X": "1024",
            "Grid_Size_Y": "1",
            "Grid_Size_Z": "1",
            "Kernel_Id": "2",
            "Kernel_Name": "vecMul",
            "Workgroup_Size_X": "256",
            "Workgroup_Size_Y": "1",
            "Workgroup_Size_Z": "1",
            "LDS_Block_Size": "0",
            "Scratch_Size": "0",
            "VGPR_Count": "32",
            "Accum_VGPR_Count": "0",
            "SGPR_Count": "32",
            "Start_Timestamp": "3000",
            "End_Timestamp": "4000",
        },
    ]


@pytest.fixture
def rocpd_database(tmp_path):
    """Create a minimal rocpd SQLite database with the expected schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))

    guid_underscored = "aaaa_bbbb"
    guid_dashed = guid_underscored.replace("_", "-")
    pmc_table = f"rocpd_pmc_event_{guid_underscored}"

    conn.execute(
        f"CREATE TABLE {pmc_table} (guid TEXT, event_id TEXT, pmc_id TEXT, value TEXT)"
    )

    conn.execute(
        "CREATE TABLE rocpd_kernel_dispatch "
        "(dispatch_id INTEGER, event_id INTEGER, guid TEXT)"
    )
    conn.execute(
        "INSERT INTO rocpd_kernel_dispatch VALUES (0, 100, ?)",
        (guid_dashed,),
    )
    conn.execute(
        "INSERT INTO rocpd_kernel_dispatch VALUES (1, 101, ?)",
        (guid_dashed,),
    )
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# _aggregate_counter_csv tests
# =============================================================================


def test_aggregate_counter_csv_groups_and_sums(native_counter_csv):
    """Rows with the same (dispatch_id, counter_name) are summed."""
    groupby_columns = ("dispatch_id", "counter_name")
    result = _aggregate_counter_csv(native_counter_csv, groupby_columns)

    assert ("0", "SQ_WAVES") in result
    assert result[("0", "SQ_WAVES")]["counter_value"] == 8.0

    assert ("0", "GRBM_GUI_ACTIVE") in result
    assert result[("0", "GRBM_GUI_ACTIVE")]["counter_value"] == 200.0

    # dispatch_id=1, SQ_WAVES appears twice: 4.0 + 6.0 = 10.0
    assert ("1", "SQ_WAVES") in result
    assert result[("1", "SQ_WAVES")]["counter_value"] == 10.0

    assert len(result) == 3


# =============================================================================
# _build_rocprofv3_counter_rows tests
# =============================================================================


def test_build_rocprofv3_counter_rows_joins_correctly(
    native_counter_csv, kernel_trace_rows
):
    """Counter groups are joined with kernel trace rows by dispatch_id."""
    aggregated = _aggregate_counter_csv(
        native_counter_csv, ("dispatch_id", "counter_name")
    )

    kernel_lookup: dict[str, list[dict]] = {}
    for kernel_row in kernel_trace_rows:
        dispatch_id = kernel_row.get("Dispatch_Id", "")
        kernel_lookup.setdefault(dispatch_id, []).append(kernel_row)

    rows = _build_rocprofv3_counter_rows(aggregated, kernel_lookup)

    assert len(rows) == 3  # 3 unique (dispatch_id, counter_name) groups

    sq_waves_d0 = [
        r for r in rows if r["Counter_Name"] == "SQ_WAVES" and r["Dispatch_Id"] == "0"
    ]
    assert len(sq_waves_d0) == 1
    assert sq_waves_d0[0]["Kernel_Name"] == "vecAdd"
    assert sq_waves_d0[0]["Grid_Size"] == 512
    assert sq_waves_d0[0]["Counter_Value"] == 8.0

    sq_waves_d1 = [
        r for r in rows if r["Counter_Name"] == "SQ_WAVES" and r["Dispatch_Id"] == "1"
    ]
    assert len(sq_waves_d1) == 1
    assert sq_waves_d1[0]["Kernel_Name"] == "vecMul"
    assert sq_waves_d1[0]["Counter_Value"] == 10.0


def test_build_rocprofv3_counter_rows_unmatched_dispatch():
    """Groups without a matching kernel entry are dropped (inner join)."""
    aggregated = {
        ("99", "SQ_WAVES"): {
            "dispatch_id": "99",
            "counter_name": "SQ_WAVES",
            "counter_value": 5.0,
        },
    }
    rows = _build_rocprofv3_counter_rows(aggregated, {})
    assert rows == []


# =============================================================================
# update_rocpd_pmc_events integration tests
# =============================================================================


def test_update_rocpd_pmc_events_inserts_rows(
    tmp_path, native_counter_csv, rocpd_database
):
    """Streaming insert populates the pmc_event table correctly."""
    rocpd_data.update_rocpd_pmc_events(str(native_counter_csv), str(rocpd_database))

    conn = sqlite3.connect(str(rocpd_database))
    rows = conn.execute(
        "SELECT guid, event_id, pmc_id, value FROM rocpd_pmc_event_aaaa_bbbb"
    ).fetchall()
    conn.close()

    assert len(rows) == 4  # 4 CSV data rows

    event_ids = {r[1] for r in rows}
    assert "100" in event_ids  # dispatch_id 0 -> event_id 100
    assert "101" in event_ids  # dispatch_id 1 -> event_id 101


def test_update_rocpd_pmc_events_no_header_raises(tmp_path, rocpd_database):
    """A CSV with no header row raises via console_error."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        rocpd_data.update_rocpd_pmc_events(str(empty_csv), str(rocpd_database))


def test_update_rocpd_pmc_events_no_dispatch_data(tmp_path, native_counter_csv):
    """Returns gracefully when no kernel dispatch data exists in the db."""
    db_path = tmp_path / "empty_dispatch.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE rocpd_pmc_event_aaaa_bbbb "
        "(guid TEXT, event_id TEXT, pmc_id TEXT, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE rocpd_kernel_dispatch "
        "(dispatch_id INTEGER, event_id INTEGER, guid TEXT)"
    )
    conn.commit()
    conn.close()

    # Should log error but not crash
    rocpd_data.update_rocpd_pmc_events(str(native_counter_csv), str(db_path))

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT * FROM rocpd_pmc_event_aaaa_bbbb").fetchall()
    conn.close()
    assert rows == []
