#!/usr/bin/env python3
"""
Tests for core analysis functions in rocinsight.analyze.

Uses in-memory SQLite databases via RocinsightConnection for isolation.
No real GPU required.
"""
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rocinsight.connection import RocinsightConnection
from rocinsight.analyze import (
    compute_time_breakdown,
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
    generate_recommendations,
    _split_pmc_into_passes,
)


# ---------------------------------------------------------------------------
# Helpers: build in-memory SQLite databases
# ---------------------------------------------------------------------------

def _make_connection(setup_sql: str) -> RocinsightConnection:
    """
    Create a temporary on-disk SQLite database, run setup_sql, and return
    a RocinsightConnection pointing at it.
    We cannot use ":memory:" directly because RocinsightConnection wraps a
    Path-based open; instead use a NamedTemporaryFile pattern.
    """
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn_raw = sqlite3.connect(path)
    conn_raw.executescript(setup_sql)
    conn_raw.close()
    return _TmpConnection(path)


class _TmpConnection:
    """Wraps RocinsightConnection so the temp file is cleaned up after use."""
    def __init__(self, path: str):
        self._path = path
        self._conn = RocinsightConnection(path)

    # Delegate everything to the inner connection
    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)
    def cursor(self):
        return self._conn.cursor()
    @property
    def connection(self):
        return self._conn.connection
    def close(self):
        self._conn.close()
        import os; os.unlink(self._path)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def _empty_db() -> _TmpConnection:
    return _make_connection("""
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );
        CREATE TABLE memory_copies (
            id INTEGER PRIMARY KEY,
            category TEXT,
            start INTEGER,
            end INTEGER,
            size INTEGER,
            duration INTEGER
        );
    """)


def _kernel_db(
    kernels: List[Dict[str, Any]],
    memcpy: List[Dict[str, Any]] = None,
) -> _TmpConnection:
    """Build a DB with arbitrary kernel rows.

    The kernels and memory_copies tables mirror the column structure of the
    real rocpd views: both include a `duration` computed column (end - start).
    """
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    raw = sqlite3.connect(path)
    raw.execute("""
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        )
    """)
    # memory_copies must have a `duration` column (mirrors the rocpd view)
    raw.execute("""
        CREATE TABLE memory_copies (
            id INTEGER PRIMARY KEY,
            category TEXT,
            start INTEGER,
            end INTEGER,
            size INTEGER,
            duration INTEGER
        )
    """)
    for i, k in enumerate(kernels):
        raw.execute(
            "INSERT INTO kernels VALUES (?,?,?,?,?)",
            (i + 1, k["name"], k["start"], k["end"], k["end"] - k["start"]),
        )
    for i, m in enumerate(memcpy or []):
        raw.execute(
            "INSERT INTO memory_copies VALUES (?,?,?,?,?,?)",
            (
                i + 1,
                m["category"],
                m["start"],
                m["end"],
                m.get("size", 0),
                m["end"] - m["start"],
            ),
        )
    raw.commit()
    raw.close()
    return _TmpConnection(path)


def _pmc_db(
    kernels: List[Dict[str, Any]],
    pmc_rows: List[Dict[str, Any]],
) -> _TmpConnection:
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );
        CREATE TABLE memory_copies (
            id INTEGER PRIMARY KEY,
            category TEXT,
            start INTEGER,
            end INTEGER,
            size INTEGER
        );
        CREATE TABLE pmc_events (
            id            INTEGER PRIMARY KEY,
            dispatch_id   INTEGER,
            name          TEXT,
            counter_name  TEXT,
            counter_value REAL
        );
    """)
    for i, k in enumerate(kernels):
        raw.execute(
            "INSERT INTO kernels VALUES (?,?,?,?,?)",
            (i + 1, k["name"], k["start"], k["end"], k["end"] - k["start"]),
        )
    for i, p in enumerate(pmc_rows):
        raw.execute(
            "INSERT INTO pmc_events VALUES (?,?,?,?,?)",
            (i + 1, p.get("dispatch_id", 1), p.get("name", "k"), p["counter_name"], p["counter_value"]),
        )
    raw.commit()
    raw.close()
    return _TmpConnection(path)


# ---------------------------------------------------------------------------
# compute_time_breakdown
# ---------------------------------------------------------------------------

class TestComputeTimeBreakdown:
    def test_empty_db_returns_zeros(self):
        with _empty_db() as conn:
            result = compute_time_breakdown(conn)
        assert result["total_runtime"] == 0
        assert result["kernel_percent"] == 0
        assert result["memcpy_percent"] == 0
        assert result["overhead_percent"] == 0

    def test_percentages_add_to_100_with_kernels_and_memcpy(self):
        # kernel: 0-1000, memcpy: 1000-2000 → total=2000, each 50%
        with _kernel_db(
            kernels=[{"name": "k", "start": 0, "end": 1000}],
            memcpy=[{"category": "HostToDevice", "start": 1000, "end": 2000}],
        ) as conn:
            result = compute_time_breakdown(conn)

        kernel_pct = result["kernel_percent"]
        memcpy_pct = result["memcpy_percent"]
        overhead_pct = result["overhead_percent"]
        total = kernel_pct + memcpy_pct + overhead_pct
        assert abs(total - 100.0) < 1.0, f"Expected ~100%, got {total}"

    def test_kernel_only_has_nonzero_kernel_percent(self):
        with _kernel_db(
            kernels=[{"name": "k", "start": 0, "end": 1000}],
        ) as conn:
            result = compute_time_breakdown(conn)
        assert result["kernel_percent"] > 0

    def test_overhead_percent_is_non_negative(self):
        with _kernel_db(
            kernels=[{"name": "k", "start": 100, "end": 900}],
            memcpy=[{"category": "H2D", "start": 0, "end": 1000}],
        ) as conn:
            result = compute_time_breakdown(conn)
        assert result["overhead_percent"] >= 0

    def test_returns_total_runtime_correctly(self):
        # kernel 0-2000, memcpy: none → total = 2000
        with _kernel_db(
            kernels=[{"name": "k", "start": 0, "end": 2000}],
        ) as conn:
            result = compute_time_breakdown(conn)
        assert result["total_runtime"] == 2000


# ---------------------------------------------------------------------------
# identify_hotspots
# ---------------------------------------------------------------------------

class TestIdentifyHotspots:
    def test_returns_top_n_sorted_by_duration(self):
        kernels = [
            {"name": "slow", "start": 0, "end": 5000},
            {"name": "fast", "start": 5000, "end": 5100},
            {"name": "medium", "start": 5100, "end": 5600},
        ]
        with _kernel_db(kernels) as conn:
            hotspots = identify_hotspots(conn, top_n=2)
        assert len(hotspots) == 2
        assert hotspots[0]["name"] == "slow"
        assert hotspots[1]["name"] == "medium"

    def test_empty_table_returns_empty_list(self):
        with _empty_db() as conn:
            hotspots = identify_hotspots(conn)
        assert hotspots == []

    def test_contains_expected_fields(self):
        kernels = [{"name": "k", "start": 0, "end": 1000}]
        with _kernel_db(kernels) as conn:
            hotspots = identify_hotspots(conn, top_n=10)
        assert len(hotspots) == 1
        h = hotspots[0]
        assert "name" in h
        assert "calls" in h
        assert "total_duration" in h
        assert "avg_duration" in h
        assert "percent_of_total" in h

    def test_top_n_limits_results(self):
        kernels = [{"name": f"k{i}", "start": i * 100, "end": (i + 1) * 100} for i in range(10)]
        with _kernel_db(kernels) as conn:
            hotspots = identify_hotspots(conn, top_n=3)
        assert len(hotspots) == 3

    def test_percent_of_total_sums_to_100_for_single_kernel(self):
        with _kernel_db([{"name": "only", "start": 0, "end": 1000}]) as conn:
            hotspots = identify_hotspots(conn)
        assert abs(hotspots[0]["percent_of_total"] - 100.0) < 0.01


# ---------------------------------------------------------------------------
# analyze_memory_copies
# ---------------------------------------------------------------------------

class TestAnalyzeMemoryCopies:
    def test_returns_h2d_stats(self):
        memcpy = [{"category": "HostToDevice", "start": 0, "end": 1000, "size": 1048576}]
        with _kernel_db(kernels=[], memcpy=memcpy) as conn:
            result = analyze_memory_copies(conn)
        assert "Host-to-Device" in result

    def test_returns_d2h_stats(self):
        memcpy = [{"category": "DeviceToHost", "start": 0, "end": 500, "size": 512}]
        with _kernel_db(kernels=[], memcpy=memcpy) as conn:
            result = analyze_memory_copies(conn)
        assert "Device-to-Host" in result

    def test_returns_empty_dict_for_no_copies(self):
        with _empty_db() as conn:
            result = analyze_memory_copies(conn)
        assert result == {}

    def test_count_field_correct(self):
        memcpy = [
            {"category": "HostToDevice", "start": 0, "end": 100},
            {"category": "HostToDevice", "start": 200, "end": 300},
        ]
        with _kernel_db(kernels=[], memcpy=memcpy) as conn:
            result = analyze_memory_copies(conn)
        assert result["Host-to-Device"]["count"] == 2

    def test_total_duration_computed(self):
        memcpy = [{"category": "DeviceToHost", "start": 0, "end": 800}]
        with _kernel_db(kernels=[], memcpy=memcpy) as conn:
            result = analyze_memory_copies(conn)
        assert result["Device-to-Host"]["total_duration"] == 800


# ---------------------------------------------------------------------------
# analyze_hardware_counters
# ---------------------------------------------------------------------------

class TestAnalyzeHardwareCounters:
    def test_returns_has_counters_false_when_table_missing(self):
        with _empty_db() as conn:
            result = analyze_hardware_counters(conn)
        assert result["has_counters"] is False

    def test_returns_has_counters_true_when_data_present(self):
        kernels = [{"name": "k", "start": 0, "end": 1000}]
        pmc = [
            {"name": "k", "counter_name": "SQ_WAVES", "counter_value": 32.0},
            {"name": "k", "counter_name": "GRBM_GUI_ACTIVE", "counter_value": 900.0},
            {"name": "k", "counter_name": "GRBM_COUNT", "counter_value": 1000.0},
        ]
        with _pmc_db(kernels, pmc) as conn:
            result = analyze_hardware_counters(conn)
        assert result["has_counters"] is True

    def test_avg_waves_in_metrics(self):
        kernels = [{"name": "k", "start": 0, "end": 1000}]
        pmc = [{"name": "k", "counter_name": "SQ_WAVES", "counter_value": 8.0}]
        with _pmc_db(kernels, pmc) as conn:
            result = analyze_hardware_counters(conn)
        assert "avg_waves" in result["metrics"]
        assert result["metrics"]["avg_waves"] == pytest.approx(8.0)

    def test_gpu_utilization_computed(self):
        kernels = [{"name": "k", "start": 0, "end": 1000}]
        pmc = [
            {"name": "k", "counter_name": "GRBM_GUI_ACTIVE", "counter_value": 700.0},
            {"name": "k", "counter_name": "GRBM_COUNT", "counter_value": 1000.0},
        ]
        with _pmc_db(kernels, pmc) as conn:
            result = analyze_hardware_counters(conn)
        assert "gpu_utilization_percent" in result["metrics"]
        util = result["metrics"]["gpu_utilization_percent"]
        assert abs(util - 70.0) < 0.1

    def test_counters_dict_contains_sq_waves(self):
        kernels = [{"name": "k", "start": 0, "end": 1000}]
        pmc = [{"name": "k", "counter_name": "SQ_WAVES", "counter_value": 16.0}]
        with _pmc_db(kernels, pmc) as conn:
            result = analyze_hardware_counters(conn)
        assert "SQ_WAVES" in result["counters"]

    def test_empty_pmc_table_returns_no_counters(self):
        # Table exists but has no rows
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        raw = sqlite3.connect(path)
        raw.executescript("""
            CREATE TABLE kernels (id INTEGER PRIMARY KEY, name TEXT, start INTEGER, end INTEGER, duration INTEGER);
            CREATE TABLE memory_copies (id INTEGER PRIMARY KEY, category TEXT, start INTEGER, end INTEGER, size INTEGER);
            CREATE TABLE pmc_events (id INTEGER PRIMARY KEY, dispatch_id INTEGER, name TEXT, counter_name TEXT, counter_value REAL);
        """)
        raw.commit()
        raw.close()
        with _TmpConnection(path) as conn:
            result = analyze_hardware_counters(conn)
        assert result["has_counters"] is False


# ---------------------------------------------------------------------------
# _split_pmc_into_passes
# ---------------------------------------------------------------------------

_BASE_FLAGS = ["--sys-trace"]
_BASE_ARGS: List[Dict] = []
_OUTPUT_DIR = "./out"
_OUTPUT_PREFIX = "profile"
_DESC = "counter collection"


class TestSplitPmcIntoPasses:
    def test_fetch_size_in_own_pass(self):
        """FETCH_SIZE must be isolated when SQ counters are also requested."""
        counters = ["SQ_WAVES", "GRBM_COUNT", "FETCH_SIZE"]
        cmds = _split_pmc_into_passes(
            counters, _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        # FETCH_SIZE should be in its own pass
        fetch_passes = [
            c for c in cmds if "FETCH_SIZE" in c.get("full_command", "")
        ]
        other_passes = [
            c for c in cmds if "FETCH_SIZE" not in c.get("full_command", "")
        ]
        assert len(fetch_passes) >= 1
        # FETCH_SIZE must not share a pass with SQ_WAVES
        for p in fetch_passes:
            assert "SQ_WAVES" not in p.get("full_command", "")

    def test_write_size_in_own_pass(self):
        """WRITE_SIZE must be isolated when SQ counters are also requested."""
        counters = ["SQ_WAVES", "WRITE_SIZE"]
        cmds = _split_pmc_into_passes(
            counters, _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        write_passes = [
            c for c in cmds if "WRITE_SIZE" in c.get("full_command", "")
        ]
        assert len(write_passes) >= 1
        for p in write_passes:
            assert "SQ_WAVES" not in p.get("full_command", "")

    def test_fetch_and_write_in_separate_passes(self):
        """FETCH_SIZE and WRITE_SIZE must each be in separate passes."""
        counters = ["FETCH_SIZE", "WRITE_SIZE"]
        cmds = _split_pmc_into_passes(
            counters, _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        # Must have at least 2 passes — one for each derived counter
        assert len(cmds) >= 2
        fetch_passes = [
            c for c in cmds if "FETCH_SIZE" in c.get("full_command", "")
        ]
        write_passes = [
            c for c in cmds if "WRITE_SIZE" in c.get("full_command", "")
        ]
        assert len(fetch_passes) >= 1
        assert len(write_passes) >= 1
        # They must not share the same pass
        for fp in fetch_passes:
            assert "WRITE_SIZE" not in fp.get("full_command", "")

    def test_regular_sq_counters_grouped_together(self):
        """SQ counters (non-derived) may share a pass when within block limit."""
        counters = ["SQ_WAVES", "SQ_BUSY_CYCLES"]
        cmds = _split_pmc_into_passes(
            counters, _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        # These two SQ counters fit within the 4-counter-per-block limit.
        # They should be in a single pass together.
        assert len(cmds) == 1
        assert "SQ_WAVES" in cmds[0]["full_command"]
        assert "SQ_BUSY_CYCLES" in cmds[0]["full_command"]

    def test_empty_counters_returns_empty_list(self):
        cmds = _split_pmc_into_passes(
            [], _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        assert cmds == []

    def test_single_regular_counter_produces_one_pass(self):
        cmds = _split_pmc_into_passes(
            ["GRBM_COUNT"], _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        assert len(cmds) == 1

    def test_full_command_contains_tool(self):
        cmds = _split_pmc_into_passes(
            ["GRBM_COUNT"], _BASE_FLAGS, _BASE_ARGS, _OUTPUT_DIR, _OUTPUT_PREFIX, _DESC
        )
        assert cmds[0]["tool"] == "rocprofv3"
        assert "rocprofv3" in cmds[0]["full_command"]


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    def _make_breakdown(self, kernel_pct=0, memcpy_pct=0, overhead_pct=0):
        runtime = 1_000_000
        return {
            "total_kernel_time": int(runtime * kernel_pct / 100),
            "total_memcpy_time": int(runtime * memcpy_pct / 100),
            "total_runtime": runtime,
            "kernel_percent": kernel_pct,
            "memcpy_percent": memcpy_pct,
            "overhead_percent": overhead_pct,
        }

    def test_high_memcpy_generates_high_priority_rec(self):
        breakdown = self._make_breakdown(kernel_pct=50, memcpy_pct=30, overhead_pct=20)
        recs = generate_recommendations(breakdown, [], {})
        priorities = [r["priority"] for r in recs]
        categories = [r["category"] for r in recs]
        assert "HIGH" in priorities
        assert any("Memory Transfer" in c for c in categories)

    def test_memcpy_below_threshold_no_memory_transfer_rec(self):
        breakdown = self._make_breakdown(kernel_pct=90, memcpy_pct=5, overhead_pct=5)
        recs = generate_recommendations(breakdown, [], {})
        categories = [r["category"] for r in recs]
        assert not any("Memory Transfer" in c for c in categories)

    def test_dominant_kernel_generates_high_priority_rec(self):
        breakdown = self._make_breakdown(kernel_pct=80, memcpy_pct=5, overhead_pct=15)
        hotspots = [
            {
                "name": "big_kernel",
                "calls": 1,
                "total_duration": 800_000,
                "avg_duration": 800_000,
                "min_duration": 800_000,
                "max_duration": 800_000,
                "percent_of_total": 80.0,
            }
        ]
        recs = generate_recommendations(breakdown, hotspots, {})
        categories = [r["category"] for r in recs]
        priorities = [r["priority"] for r in recs]
        assert "Compute Bottleneck" in categories
        compute_recs = [r for r in recs if r["category"] == "Compute Bottleneck"]
        assert compute_recs[0]["priority"] == "HIGH"

    def test_kernel_below_50pct_no_compute_bottleneck(self):
        breakdown = self._make_breakdown(kernel_pct=40, memcpy_pct=10, overhead_pct=50)
        hotspots = [
            {
                "name": "medium_kernel",
                "calls": 10,
                "total_duration": 400_000,
                "avg_duration": 40_000,
                "min_duration": 40_000,
                "max_duration": 40_000,
                "percent_of_total": 40.0,
            }
        ]
        recs = generate_recommendations(breakdown, hotspots, {})
        categories = [r["category"] for r in recs]
        assert "Compute Bottleneck" not in categories

    def test_high_api_overhead_generates_medium_priority_rec(self):
        breakdown = self._make_breakdown(kernel_pct=60, memcpy_pct=10, overhead_pct=30)
        recs = generate_recommendations(breakdown, [], {})
        api_recs = [r for r in recs if r["category"] == "API Overhead"]
        assert len(api_recs) > 0
        assert api_recs[0]["priority"] == "MEDIUM"

    def test_overhead_below_threshold_no_api_overhead_rec(self):
        breakdown = self._make_breakdown(kernel_pct=90, memcpy_pct=5, overhead_pct=5)
        recs = generate_recommendations(breakdown, [], {})
        categories = [r["category"] for r in recs]
        assert "API Overhead" not in categories

    def test_no_issues_generates_info_rec(self):
        breakdown = self._make_breakdown(kernel_pct=85, memcpy_pct=10, overhead_pct=5)
        recs = generate_recommendations(breakdown, [], {})
        # No issues: should get default INFO recommendation
        info_recs = [r for r in recs if r["priority"] == "INFO"]
        assert len(info_recs) > 0

    def test_recommendations_have_required_fields(self):
        breakdown = self._make_breakdown(kernel_pct=50, memcpy_pct=30, overhead_pct=20)
        recs = generate_recommendations(breakdown, [], {})
        for r in recs:
            assert "priority" in r
            assert "category" in r
            assert "issue" in r
            assert "suggestion" in r

    def test_low_wave_occupancy_generates_high_priority_rec(self):
        breakdown = self._make_breakdown(kernel_pct=90, memcpy_pct=5, overhead_pct=5)
        hw_counters = {
            "has_counters": True,
            "metrics": {"avg_waves": 8.0},
            "counters": {},
            "per_kernel": {},
        }
        recs = generate_recommendations(breakdown, [], {}, hardware_counters=hw_counters)
        occupancy_recs = [r for r in recs if "Occupancy" in r["category"]]
        assert len(occupancy_recs) > 0
        assert occupancy_recs[0]["priority"] == "HIGH"

    def test_no_counters_no_occupancy_rec(self):
        breakdown = self._make_breakdown(kernel_pct=90, memcpy_pct=5, overhead_pct=5)
        hw_counters = {"has_counters": False}
        recs = generate_recommendations(breakdown, [], {}, hardware_counters=hw_counters)
        categories = [r["category"] for r in recs]
        assert not any("Occupancy" in c for c in categories)
