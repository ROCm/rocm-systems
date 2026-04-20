"""Tests for per-API overhead breakdown and kernel resource/occupancy analysis.

Tests cover:
- C2: analyze_api_overhead() per-API breakdown
- I1: analyze_kernel_resources() + occupancy calculation
- api.py wiring for interactive mode
"""

import pytest
from unittest.mock import MagicMock, patch

from rocinsight.analysis.core import (
    analyze_api_overhead,
    analyze_kernel_resources,
    _ARCH_SPECS,
)
from rocinsight.analysis.recommendations import generate_recommendations


# ---------------------------------------------------------------------------
# C2: Per-API overhead breakdown
# ---------------------------------------------------------------------------

class TestC2ApiOverhead:
    def test_returns_empty_when_no_regions(self):
        """Gracefully handles DBs without regions view."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = Exception("no such table: regions")

        with patch("rocinsight.analysis.core.execute_statement", side_effect=Exception("no such table")):
            result = analyze_api_overhead(mock_conn)
        assert result["has_api_data"] is False
        assert result["api_calls"] == []

    def test_returns_api_calls_sorted_by_time(self):
        """API calls sorted by total_ns descending."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("hipHostMalloc", 2, 350_000_000, 175_000_000),
            ("hipMemcpy", 6, 80_000_000, 13_333_333),
            ("hipLaunchKernel", 100, 5_000_000, 50_000),
        ]

        with patch("rocinsight.analysis.core.execute_statement", return_value=mock_cursor):
            result = analyze_api_overhead(mock_conn)

        assert result["has_api_data"] is True
        assert len(result["api_calls"]) == 3
        assert result["api_calls"][0]["name"] == "hipHostMalloc"
        assert result["launch_overhead_ns"] == 5_000_000
        assert result["total_api_ns"] == 435_000_000

    def test_api_overhead_in_recommendation_issue(self):
        """When api_overhead is available, Rule 2 shows per-API breakdown."""
        tb = {
            "total_kernel_time": 50_000_000,
            "total_memcpy_time": 0,
            "total_runtime": 200_000_000,
            "kernel_percent": 25.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 75.0,
        }
        api = {
            "has_api_data": True,
            "api_calls": [
                {"name": "hipSetDevice", "calls": 1, "total_ns": 100_000_000, "avg_ns": 100_000_000},
            ],
            "launch_overhead_ns": 2_000_000,
            "total_api_ns": 150_000_000,
        }
        recs = generate_recommendations(tb, [], {}, None, api_overhead=api)
        api_recs = [r for r in recs if r["category"] == "API Overhead"]
        assert len(api_recs) == 1
        assert "hipLaunchKernel" in api_recs[0]["issue"]
        assert "hipSetDevice" in api_recs[0]["issue"]


# ---------------------------------------------------------------------------
# I1: Kernel resources + occupancy
# ---------------------------------------------------------------------------

class TestI1KernelResources:
    def test_arch_specs_contains_expected(self):
        """_ARCH_SPECS has all documented architectures."""
        expected = {"gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100"}
        assert expected.issubset(set(_ARCH_SPECS.keys()))

    def test_gfx950_lds_is_160kb(self):
        """CDNA4 (gfx950) has 160 KB LDS per CU."""
        assert _ARCH_SPECS["gfx950"]["lds_per_cu_kb"] == 160

    def test_returns_empty_when_no_hotspots(self):
        """No hotspots -> empty kernel list."""
        mock_conn = MagicMock()
        with patch("rocinsight.analysis.core.execute_statement"):
            result = analyze_kernel_resources(mock_conn, [])
        assert result["kernels"] == []

    def test_occupancy_calculation(self):
        """Known occupancy: 4 VGPRs, no LDS, 256-thread block on gfx942 -> 100%."""
        mock_conn = MagicMock()
        # First call: agent query -> gfx942
        # Second call: kernel query -> resources
        agent_cursor = MagicMock()
        agent_cursor.fetchone.return_value = ("AMD Instinct MI300X gfx942",)
        kernel_cursor = MagicMock()
        kernel_cursor.fetchone.return_value = (
            "test_kernel", 4, 0, 16, 0, 0, 256, 1, 1, 1024, 1, 1
        )

        call_count = [0]
        def mock_exec(conn, query, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return agent_cursor
            return kernel_cursor

        with patch("rocinsight.analysis.core.execute_statement", side_effect=mock_exec):
            result = analyze_kernel_resources(
                mock_conn,
                [{"name": "test_kernel", "calls": 10}],
            )

        assert len(result["kernels"]) == 1
        assert result["arch"] == "gfx942"
        occ = result["kernels"][0].get("occupancy")
        assert occ is not None
        assert occ["percent"] == 100.0
        assert occ["limiting_resource"] == "none"

    def test_kernel_resources_in_json(self):
        """kernel_resources appears in JSON output."""
        from rocinsight.formatters.json_fmt import _format_as_json
        import json

        kr = {
            "arch": "gfx942",
            "arch_specs": _ARCH_SPECS["gfx942"],
            "kernels": [{"name": "k1", "vgpr": 32, "sgpr": 16, "lds_bytes": 0, "scratch_bytes": 0, "block": "256x1x1", "grid": "1024x1x1"}],
        }
        tb = {"total_kernel_time": 100, "total_memcpy_time": 0, "total_runtime": 100, "kernel_percent": 100, "memcpy_percent": 0, "overhead_percent": 0}
        output = _format_as_json(tb, [], {}, [], kernel_resources=kr)
        doc = json.loads(output)
        assert "kernel_resources" in doc
        assert doc["kernel_resources"]["arch"] == "gfx942"
