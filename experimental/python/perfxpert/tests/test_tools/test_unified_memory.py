"""Tests for perfxpert.tools.unified_memory."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from perfxpert.tools import unified_memory
from perfxpert.tools._class import ToolClass


def _make_memcpy_db(tmp_path: Path, rows: list, with_agent: bool = False) -> Path:
    """Create a minimal rocprofiler-sdk-style DB with a memory_copy table.

    rows: list of (size, kind, src, dst) tuples.
    """
    uid = uuid.uuid4().hex[:8]
    table = f"rocpd_memory_copy_{uid}"
    db = tmp_path / "um.db"
    conn = sqlite3.connect(db)
    if with_agent:
        conn.execute(
            f"CREATE TABLE {table} (size INTEGER, kind TEXT, "
            f"src_agent_id INTEGER, dst_agent_id INTEGER)"
        )
        for size, kind, src, dst in rows:
            conn.execute(
                f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                (size, kind, src, dst),
            )
    else:
        conn.execute(f"CREATE TABLE {table} (size INTEGER, kind TEXT)")
        for size, kind, _src, _dst in rows:
            conn.execute(f"INSERT INTO {table} VALUES (?, ?)", (size, kind))
    conn.commit()
    conn.close()
    return db


def test_is_read_only_class():
    assert unified_memory.analyze_paging.__tool_class__ == ToolClass.READ_ONLY


def test_happy_path_detects_paging(tmp_path):
    rows = [
        # >= 1 MiB HtoD transfers — paging-like
        (2 << 20, "HTOD", 0, 0),
        (4 << 20, "DTOH", 0, 0),
        # Too small — doesn't count as paging
        (512, "HTOD", 0, 0),
    ]
    db = _make_memcpy_db(tmp_path, rows)
    res = unified_memory.analyze_paging(str(db))
    assert res["paging_events"] == 2
    assert "recommendations" in res
    assert len(res["recommendations"]) >= 1


def test_happy_path_cross_die_estimate(tmp_path):
    # Large cross-die DtoD (src != dst agent) — triggers XCD tax estimate.
    rows = [
        (2 << 30, "DTOD", 0, 1),  # 2 GiB cross-die
        (1 << 20, "DTOD", 0, 0),  # 1 MiB in-die (no penalty)
    ]
    db = _make_memcpy_db(tmp_path, rows, with_agent=True)
    res = unified_memory.analyze_paging(str(db))
    assert res["cross_die_bytes"] >= (2 << 30)
    assert res["estimated_penalty_ns"] > 0
    # MI300X recommendation should be present.
    assert any("MI300X" in r or "XCD" in r for r in res["recommendations"])


def test_edge_case_no_memcpy_table(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    res = unified_memory.analyze_paging(str(db))
    assert res["paging_events"] == 0
    assert res["cross_die_bytes"] == 0
    assert res["page_faults"] == 0
    # A healthy "no pathology" recommendation is emitted.
    assert any("healthy" in r or "No unified" in r for r in res["recommendations"])


def test_edge_case_unreadable_db_path(tmp_path):
    res = unified_memory.analyze_paging(str(tmp_path / "no_such.db"))
    # Opening a new SQLite path still succeeds (creates empty DB), but
    # the absence of any tables yields the zero-shape result.
    assert res["paging_events"] == 0
    assert res["estimated_penalty_ns"] == 0
