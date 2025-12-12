"""Tests for perfxpert.tools.kernel_fusion."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from perfxpert.tools import kernel_fusion
from perfxpert.tools._class import ToolClass


def _make_fixture_db(tmp_path: Path, rows: list) -> Path:
    """Build a minimal rocprofiler-sdk-style trace DB with kernel dispatches.

    Each row is (start_ns, end_ns, kernel_id, block_dims, grid_dims).
    """
    uid = uuid.uuid4().hex[:8]
    kd_table = f"rocpd_kernel_dispatch_{uid}"
    info_table = f"rocpd_info_kernel_symbol_{uid}"
    db_path = tmp_path / "trace.db"
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE {kd_table} (
            start INTEGER, end INTEGER, kernel_id INTEGER,
            workgroup_size_x INTEGER, workgroup_size_y INTEGER, workgroup_size_z INTEGER,
            grid_size_x INTEGER, grid_size_y INTEGER, grid_size_z INTEGER
        )
    """)
    conn.execute(f"CREATE TABLE {info_table} (id INTEGER PRIMARY KEY, display_name TEXT)")

    # Build rows — kernel_id doubles as index into a name list.
    names = set()
    for start, end, kid, name, block, grid in rows:
        names.add((kid, name))
        conn.execute(
            f"INSERT INTO {kd_table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (start, end, kid, block[0], block[1], block[2],
             grid[0], grid[1], grid[2]),
        )
    for kid, n in names:
        conn.execute(f"INSERT INTO {info_table} VALUES (?, ?)", (kid, n))
    conn.commit()
    conn.close()
    return db_path


def test_is_read_only_class():
    assert kernel_fusion.find_fusion_candidates.__tool_class__ == ToolClass.READ_ONLY


def test_happy_path_finds_adjacent_short_kernels(tmp_path):
    """Two short adjacent kernels with the same signature produce a candidate."""
    rows = [
        # start, end, kid, name, block, grid
        (0,     2_000, 1, "add_kernel", (256, 1, 1), (64, 1, 1)),
        (2_100, 4_000, 1, "add_kernel", (256, 1, 1), (64, 1, 1)),  # same sig, 100ns gap
    ]
    db = _make_fixture_db(tmp_path, rows)
    out = kernel_fusion.find_fusion_candidates(str(db))
    assert len(out) == 1
    c = out[0]
    assert c["pair"] == ["add_kernel", "add_kernel"]
    assert c["gap_ns"] == 100
    assert 1.0 <= c["est_speedup_lo"] <= c["est_speedup_hi"]
    assert 0.0 < c["confidence"] <= 1.0


def test_result_sorted_by_hi_speedup(tmp_path):
    rows = [
        # small gap -> high confidence + bigger lo bound
        (0,     2_000, 1, "add_a", (256, 1, 1), (64, 1, 1)),
        (2_050, 4_000, 1, "add_a", (256, 1, 1), (64, 1, 1)),
        # Large gap (but still under 500ns) -> larger hi bound
        (10_000, 12_000, 2, "mul_b", (128, 1, 1), (32, 1, 1)),
        (12_400, 14_000, 2, "mul_b", (128, 1, 1), (32, 1, 1)),
    ]
    db = _make_fixture_db(tmp_path, rows)
    out = kernel_fusion.find_fusion_candidates(str(db))
    assert len(out) == 2
    # Highest hi first
    assert out[0]["est_speedup_hi"] >= out[1]["est_speedup_hi"]


def test_edge_case_empty_db_returns_empty_list(tmp_path):
    """Missing kernel_dispatch table -> empty list, no exception."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (id INTEGER)")
    conn.commit()
    conn.close()
    assert kernel_fusion.find_fusion_candidates(str(db)) == []


def test_edge_case_long_kernels_skipped(tmp_path):
    """Kernels >= 10us are not fusion candidates."""
    rows = [
        (0,          15_000, 1, "long", (256, 1, 1), (64, 1, 1)),
        (15_050,     30_000, 1, "long", (256, 1, 1), (64, 1, 1)),
    ]
    db = _make_fixture_db(tmp_path, rows)
    assert kernel_fusion.find_fusion_candidates(str(db)) == []


def test_edge_case_large_gap_skipped(tmp_path):
    """Gap > max_gap_ns -> not a candidate even if kernels are short."""
    rows = [
        (0,     2_000, 1, "add", (256, 1, 1), (64, 1, 1)),
        (5_000, 6_000, 1, "add", (256, 1, 1), (64, 1, 1)),  # gap = 3000 ns
    ]
    db = _make_fixture_db(tmp_path, rows)
    assert kernel_fusion.find_fusion_candidates(str(db), max_gap_ns=500) == []


def test_edge_case_signature_mismatch_rejects(tmp_path):
    """Different block dims -> different signature -> no candidate."""
    rows = [
        (0,     2_000, 1, "add", (256, 1, 1), (64, 1, 1)),
        (2_100, 4_000, 2, "add", (128, 1, 1), (64, 1, 1)),
    ]
    db = _make_fixture_db(tmp_path, rows)
    assert kernel_fusion.find_fusion_candidates(str(db)) == []
