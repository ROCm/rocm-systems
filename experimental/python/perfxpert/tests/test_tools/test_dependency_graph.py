"""Tests for perfxpert.tools.dependency_graph."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from perfxpert.tools import dependency_graph
from perfxpert.tools._class import ToolClass


def _make_fixture_db(tmp_path: Path, rows: list) -> Path:
    """rows: (start, end, kid, name, stream)"""
    uid = uuid.uuid4().hex[:8]
    kt = f"rocpd_kernel_dispatch_{uid}"
    info = f"rocpd_info_kernel_symbol_{uid}"
    db = tmp_path / "dag.db"
    conn = sqlite3.connect(db)
    conn.execute(
        f"CREATE TABLE {kt} (start INTEGER, end INTEGER, kernel_id INTEGER, "
        f"stream_id INTEGER)"
    )
    conn.execute(f"CREATE TABLE {info} (id INTEGER PRIMARY KEY, display_name TEXT)")

    names = {}
    for start, end, kid, name, stream in rows:
        names[kid] = name
        conn.execute(
            f"INSERT INTO {kt} VALUES (?, ?, ?, ?)", (start, end, kid, stream)
        )
    for kid, name in names.items():
        conn.execute(f"INSERT INTO {info} VALUES (?, ?)", (kid, name))
    conn.commit()
    conn.close()
    return db


def test_is_read_only_class():
    assert dependency_graph.reconstruct_dag.__tool_class__ == ToolClass.READ_ONLY


def test_happy_path_single_stream(tmp_path):
    rows = [
        (0,       5_000, 1, "k_a", 0),
        (5_100,  10_000, 2, "k_b", 0),
        (10_200, 15_000, 3, "k_c", 0),
    ]
    db = _make_fixture_db(tmp_path, rows)
    res = dependency_graph.reconstruct_dag(str(db))
    assert len(res["nodes"]) == 3
    # Two intra-stream edges.
    assert len(res["edges"]) == 2
    # Critical path is the whole stream (only stream).
    assert len(res["critical_path"]) == 3


def test_bubble_detection(tmp_path):
    """Gap of > 2us between same-stream kernels becomes a bubble."""
    rows = [
        (0,       5_000, 1, "k_a", 0),
        (15_000, 20_000, 2, "k_b", 0),   # 10us idle gap
    ]
    db = _make_fixture_db(tmp_path, rows)
    res = dependency_graph.reconstruct_dag(str(db))
    assert len(res["bubbles"]) == 1
    b = res["bubbles"][0]
    assert b["start"] == 5_000
    assert b["end"] == 15_000
    assert b["duration_ns"] == 10_000
    assert res["total_bubble_ns"] == 10_000


def test_multi_stream_critical_path(tmp_path):
    """Critical path is the longer of two streams."""
    rows = [
        # stream 0 — short
        (0,      2_000, 1, "k_s0_a", 0),
        (2_100,  4_000, 2, "k_s0_b", 0),
        # stream 1 — long
        (0,      10_000, 3, "k_s1_a", 1),
        (10_100, 20_000, 4, "k_s1_b", 1),
    ]
    db = _make_fixture_db(tmp_path, rows)
    res = dependency_graph.reconstruct_dag(str(db))
    # Stream 1 has ~20us wall time vs stream 0 ~4us — critical path on s1.
    cp_names = [n["name"] for n in res["nodes"] if n["id"] in res["critical_path"]]
    assert "k_s1_a" in cp_names and "k_s1_b" in cp_names


def test_edge_case_empty_db(tmp_path):
    """No kernel_dispatch table -> empty DAG, no raise."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    res = dependency_graph.reconstruct_dag(str(db))
    assert res == {
        "nodes": [], "edges": [], "critical_path": [],
        "bubbles": [], "total_bubble_ns": 0, "sync_event_count": 0,
    }


def test_edge_case_small_gaps_ignored(tmp_path):
    """Gaps < 2us are scheduler noise, not bubbles."""
    rows = [
        (0,     5_000, 1, "a", 0),
        (5_500, 10_000, 2, "b", 0),   # 500 ns gap
    ]
    db = _make_fixture_db(tmp_path, rows)
    res = dependency_graph.reconstruct_dag(str(db))
    assert res["bubbles"] == []
    assert res["total_bubble_ns"] == 0
