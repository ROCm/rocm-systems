"""Tests for ROCM-21553 PR3: roctx support and profiling scope.

Tests cover:
- C1: analyze_roctx_regions() marker detection and correlation
- E1: rocprof-compute is first command in Kernel Hotspot
- E2: roctxProfilerPause recommendation when unranged > 50%
- GPU idle warning scoped to figure-of-merit region
"""

import pytest
from unittest.mock import MagicMock, patch

from rocinsight.analysis.core import analyze_roctx_regions
from rocinsight.analysis.recommendations import generate_recommendations


# ---------------------------------------------------------------------------
# C1: roctx marker analysis
# ---------------------------------------------------------------------------

class TestC1RoctxRegions:
    def test_returns_none_when_no_markers(self):
        """No roctx markers -> returns None (graceful no-op)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch("rocinsight.analysis.core.execute_statement", return_value=mock_cursor):
            result = analyze_roctx_regions(mock_conn)
        assert result is None

    def test_returns_none_when_regions_view_missing(self):
        """Old DB without regions view -> returns None."""
        mock_conn = MagicMock()
        with patch("rocinsight.analysis.core.execute_statement", side_effect=Exception("no such table")):
            result = analyze_roctx_regions(mock_conn)
        assert result is None

    def test_detects_markers_and_computes_breakdown(self):
        """Detects markers and correlates kernels by timestamp overlap."""
        mock_conn = MagicMock()

        # Marker query returns 1 marker: "timing" from 100 to 200
        marker_cursor = MagicMock()
        marker_cursor.fetchall.return_value = [
            ("timing", 100, 200, 100),
        ]
        # Kernel query returns 2 kernels: one inside marker, one outside
        kernel_cursor = MagicMock()
        kernel_cursor.fetchall.return_value = [
            ("kernel_a", 110, 150, 40),  # inside marker
            ("kernel_b", 250, 300, 50),  # outside marker
        ]
        # Memcpy query returns empty
        memcpy_cursor = MagicMock()
        memcpy_cursor.fetchall.return_value = []

        call_count = [0]
        def mock_exec(conn, query, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return marker_cursor
            elif call_count[0] == 2:
                return kernel_cursor
            return memcpy_cursor

        with patch("rocinsight.analysis.core.execute_statement", side_effect=mock_exec):
            result = analyze_roctx_regions(mock_conn)

        assert result is not None
        assert result["has_markers"] is True
        assert result["marker_count"] == 1
        assert len(result["regions"]) == 1
        assert result["regions"][0]["name"] == "timing"
        assert result["regions"][0]["kernel_count"] == 1
        assert result["regions"][0]["kernel_time_ns"] == 40

    def test_gpu_idle_suppressed_when_fom_region_utilized(self):
        """GPU idle warning suppressed when figure-of-merit region > 90% kernel."""
        tb = {
            "total_kernel_time": 50_000_000,
            "total_memcpy_time": 0,
            "total_runtime": 200_000_000,
            "kernel_percent": 25.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 75.0,
        }
        it = {"idle_pct": 75.0, "total_wall_ns": 200_000_000}
        roctx = {
            "has_markers": True,
            "marker_count": 1,
            "regions": [{
                "name": "timing",
                "wall_time_ns": 50_000_000,
                "kernel_count": 100,
                "kernel_time_ns": 49_500_000,
                "memcpy_count": 0,
                "memcpy_time_ns": 0,
                "overhead_time_ns": 500_000,
                "kernel_pct": 99.0,
                "memcpy_pct": 0.0,
                "overhead_pct": 1.0,
            }],
            "unranged": {"wall_time_ns": 150_000_000, "kernel_count": 0, "kernel_time_ns": 0,
                         "memcpy_count": 0, "memcpy_time_ns": 0, "overhead_time_ns": 150_000_000,
                         "kernel_pct": 0, "memcpy_pct": 0, "overhead_pct": 100},
            "unranged_pct_of_total": 75.0,
            "total_runtime_ns": 200_000_000,
        }
        recs = generate_recommendations(tb, [], {}, None, interval_timeline=it, roctx_regions=roctx)
        # Should have scoped INFO, NOT the HIGH GPU Utilization
        cats = [r["category"] for r in recs]
        assert "GPU Utilization (Scoped)" in cats
        assert "GPU Utilization" not in cats  # HIGH version suppressed


# ---------------------------------------------------------------------------
# E1: rocprof-compute first
# ---------------------------------------------------------------------------

class TestE1RocprofComputeFirst:
    def test_rocprof_compute_is_first_command(self):
        """rocprof-compute should be the first command in Kernel Hotspot."""
        tb = {
            "total_kernel_time": 90_000_000,
            "total_memcpy_time": 0,
            "total_runtime": 100_000_000,
            "kernel_percent": 90.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 10.0,
        }
        hotspots = [{
            "name": "my_kernel",
            "calls": 100,
            "total_duration": 90_000_000,
            "avg_duration": 900_000,
            "percent_of_total": 90.0,
        }]
        recs = generate_recommendations(tb, hotspots, {}, None)
        kernel_recs = [r for r in recs if r["category"] == "Kernel Hotspot"]
        assert len(kernel_recs) == 1
        cmds = kernel_recs[0].get("commands", [])
        assert len(cmds) >= 2
        assert cmds[0]["tool"] == "rocprof-compute"
        assert cmds[1]["tool"] == "rocprofv3"


# ---------------------------------------------------------------------------
# E2: roctxProfilerPause recommendation
# ---------------------------------------------------------------------------

class TestE2ProfilerPause:
    def test_fires_when_unranged_over_50(self):
        """Profiling Scope recommendation fires when unranged > 50%."""
        roctx = {
            "has_markers": True,
            "marker_count": 1,
            "regions": [{"name": "timing", "wall_time_ns": 50_000_000,
                         "kernel_count": 10, "kernel_time_ns": 45_000_000,
                         "memcpy_count": 0, "memcpy_time_ns": 0,
                         "overhead_time_ns": 5_000_000,
                         "kernel_pct": 90.0, "memcpy_pct": 0, "overhead_pct": 10.0}],
            "unranged": {"wall_time_ns": 150_000_000, "kernel_count": 0,
                         "kernel_time_ns": 0, "memcpy_count": 0,
                         "memcpy_time_ns": 0, "overhead_time_ns": 150_000_000,
                         "kernel_pct": 0, "memcpy_pct": 0, "overhead_pct": 100},
            "unranged_pct_of_total": 75.0,
            "total_runtime_ns": 200_000_000,
        }
        tb = {"total_kernel_time": 0, "total_memcpy_time": 0, "total_runtime": 0,
              "kernel_percent": 0, "memcpy_percent": 0, "overhead_percent": 0}
        recs = generate_recommendations(tb, [], {}, None, roctx_regions=roctx)
        scope_recs = [r for r in recs if r["category"] == "Profiling Scope"]
        assert len(scope_recs) == 1
        assert "roctxProfilerPause" in scope_recs[0]["actions"][0]
        assert scope_recs[0]["confidence"] == 0.90

    def test_absent_when_unranged_under_50(self):
        """No Profiling Scope recommendation when unranged < 50%."""
        roctx = {
            "has_markers": True,
            "marker_count": 1,
            "regions": [{"name": "timing", "wall_time_ns": 150_000_000,
                         "kernel_count": 10, "kernel_time_ns": 100_000_000,
                         "memcpy_count": 0, "memcpy_time_ns": 0,
                         "overhead_time_ns": 50_000_000,
                         "kernel_pct": 66.7, "memcpy_pct": 0, "overhead_pct": 33.3}],
            "unranged": {"wall_time_ns": 50_000_000, "kernel_count": 0,
                         "kernel_time_ns": 0, "memcpy_count": 0,
                         "memcpy_time_ns": 0, "overhead_time_ns": 50_000_000,
                         "kernel_pct": 0, "memcpy_pct": 0, "overhead_pct": 100},
            "unranged_pct_of_total": 25.0,
            "total_runtime_ns": 200_000_000,
        }
        tb = {"total_kernel_time": 0, "total_memcpy_time": 0, "total_runtime": 0,
              "kernel_percent": 0, "memcpy_percent": 0, "overhead_percent": 0}
        recs = generate_recommendations(tb, [], {}, None, roctx_regions=roctx)
        scope_recs = [r for r in recs if r["category"] == "Profiling Scope"]
        assert len(scope_recs) == 0
