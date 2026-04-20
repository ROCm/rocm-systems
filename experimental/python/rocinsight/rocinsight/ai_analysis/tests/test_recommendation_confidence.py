"""Tests for recommendation confidence, category rename, data-driven impact, and warmup detection.

Tests cover:
- C3: Rename Compute Bottleneck + confidence percentages
- C4: Data-driven estimated_impact strings
- I3: Warmup detection (detect_warmup_issues)
"""

import pytest
from unittest.mock import MagicMock, patch

from rocinsight.analysis.recommendations import generate_recommendations
from rocinsight.analysis.core import detect_warmup_issues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_time_breakdown(kernel_pct=30, memcpy_pct=10, overhead_pct=60):
    total = 100_000_000  # 100ms
    return {
        "total_kernel_time": int(total * kernel_pct / 100),
        "total_memcpy_time": int(total * memcpy_pct / 100),
        "total_runtime": total,
        "kernel_percent": kernel_pct,
        "memcpy_percent": memcpy_pct,
        "overhead_percent": overhead_pct,
    }


def _make_hotspots(top_pct=90, name="test_kernel"):
    return [
        {
            "name": name,
            "calls": 100,
            "total_duration": 90_000_000,
            "avg_duration": 900_000,
            "percent_of_total": top_pct,
        }
    ]


# ---------------------------------------------------------------------------
# C3: Confidence field present on all recommendations
# ---------------------------------------------------------------------------

class TestC3Confidence:
    def test_confidence_present_on_kernel_hotspot(self):
        """When no counters, Kernel Hotspot has confidence 0.50."""
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        hotspot = [r for r in recs if r["category"] == "Kernel Hotspot"]
        assert len(hotspot) == 1
        assert hotspot[0]["confidence"] == 0.50

    def test_compute_bound_with_high_utilization(self):
        """GPU util > 90% → Compute-Bound Kernel, confidence 0.85."""
        hw = {
            "has_counters": True,
            "metrics": {"gpu_utilization_percent": 95, "avg_waves": 10},
        }
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            hw,
        )
        cb = [r for r in recs if r["category"] == "Compute-Bound Kernel"]
        assert len(cb) == 1
        assert cb[0]["confidence"] == 0.85

    def test_memory_bound_with_low_utilization(self):
        """GPU util < 70% → Memory-Bound Kernel, confidence 0.80."""
        hw = {
            "has_counters": True,
            "metrics": {"gpu_utilization_percent": 50, "avg_waves": 5},
        }
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            hw,
        )
        mb = [r for r in recs if r["category"] == "Memory-Bound Kernel"]
        assert len(mb) == 1
        assert mb[0]["confidence"] == 0.80

    def test_mixed_bottleneck_moderate_utilization(self):
        """GPU util 70-90% → Mixed Bottleneck Kernel, confidence 0.70."""
        hw = {
            "has_counters": True,
            "metrics": {"gpu_utilization_percent": 80, "avg_waves": 12},
        }
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            hw,
        )
        mixed = [r for r in recs if r["category"] == "Mixed Bottleneck Kernel"]
        assert len(mixed) == 1
        assert mixed[0]["confidence"] == 0.70

    def test_all_recommendations_have_confidence(self):
        """Every recommendation must have a confidence field."""
        recs = generate_recommendations(
            _make_time_breakdown(overhead_pct=60, memcpy_pct=25),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        for rec in recs:
            assert "confidence" in rec, f"Missing confidence on: {rec['category']}"
            assert isinstance(rec["confidence"], (int, float))
            assert 0 <= rec["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# C3: Compute Bottleneck renamed
# ---------------------------------------------------------------------------

class TestC3Rename:
    def test_no_compute_bottleneck_category(self):
        """The old 'Compute Bottleneck' category must not appear."""
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        categories = [r["category"] for r in recs]
        assert "Compute Bottleneck" not in categories

    def test_kernel_hotspot_label_without_counters(self):
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        categories = [r["category"] for r in recs]
        assert "Kernel Hotspot" in categories


# ---------------------------------------------------------------------------
# C4: Data-driven estimated impact
# ---------------------------------------------------------------------------

class TestC4DataDrivenImpact:
    def test_api_overhead_impact_has_data(self):
        """API overhead estimated_impact must contain actual percentage."""
        recs = generate_recommendations(
            _make_time_breakdown(overhead_pct=60),
            [],
            {},
            None,
        )
        api = [r for r in recs if r["category"] == "API Overhead"]
        if api:
            assert "60.0%" in api[0]["estimated_impact"]

    def test_kernel_hotspot_impact_has_kernel_time(self):
        """Kernel Hotspot impact must mention actual kernel time."""
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        hs = [r for r in recs if r["category"] == "Kernel Hotspot"]
        if hs:
            # Should contain actual percent
            assert "80.0%" in hs[0]["estimated_impact"]

    def test_no_generic_percentage_ranges(self):
        """estimated_impact must not contain generic ranges like '20-50%' or '10-30%'."""
        recs = generate_recommendations(
            _make_time_breakdown(),
            _make_hotspots(top_pct=80),
            {},
            None,
        )
        for rec in recs:
            impact = rec.get("estimated_impact", "")
            assert "20-50%" not in impact, f"Generic range in: {rec['category']}"
            assert "10-30%" not in impact, f"Generic range in: {rec['category']}"
            assert "5-15%" not in impact, f"Generic range in: {rec['category']}"


# ---------------------------------------------------------------------------
# I3: Warmup detection
# ---------------------------------------------------------------------------

class TestI3WarmupDetection:
    def test_no_outliers_when_consistent(self):
        """No warmup issues when all dispatches have similar duration."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (900_000,)  # Same as avg
        mock_conn.cursor.return_value = mock_cursor

        with patch("rocinsight.analysis.core.execute_statement", return_value=mock_cursor):
            result = detect_warmup_issues(
                mock_conn,
                [{"name": "kernel_a", "calls": 10, "avg_duration": 900_000}],
            )
        assert result["has_warmup_issues"] is False
        assert result["outliers"] == []

    def test_detects_5x_outlier(self):
        """Flags kernel when first dispatch is 5x average."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5_000_000,)  # 5x the avg
        mock_conn.cursor.return_value = mock_cursor

        with patch("rocinsight.analysis.core.execute_statement", return_value=mock_cursor):
            result = detect_warmup_issues(
                mock_conn,
                [{"name": "kernel_a", "calls": 10, "avg_duration": 1_000_000}],
            )
        assert result["has_warmup_issues"] is True
        assert len(result["outliers"]) == 1
        assert result["outliers"][0]["ratio"] == 5.0

    def test_skips_single_dispatch_kernels(self):
        """Kernels with only 1 call are skipped (no baseline to compare)."""
        mock_conn = MagicMock()
        with patch("rocinsight.analysis.core.execute_statement") as mock_exec:
            result = detect_warmup_issues(
                mock_conn,
                [{"name": "kernel_a", "calls": 1, "avg_duration": 100_000}],
            )
        # Should not even query the DB
        mock_exec.assert_not_called()
        assert result["has_warmup_issues"] is False

    def test_warmup_recommendation_in_output(self):
        """Warmup INFO recommendation appears when outlier detected."""
        warmup = {
            "has_warmup_issues": True,
            "outliers": [
                {
                    "kernel_name": "my_kernel",
                    "first_duration_ns": 5_000_000,
                    "avg_duration_ns": 1_000_000,
                    "ratio": 5.0,
                }
            ],
        }
        recs = generate_recommendations(
            _make_time_breakdown(),
            [],
            {},
            None,
            warmup_issues=warmup,
        )
        warmup_recs = [r for r in recs if r["category"] == "Warmup"]
        assert len(warmup_recs) == 1
        assert "5.0x slower" in warmup_recs[0]["issue"]
        assert warmup_recs[0]["priority"] == "INFO"
        assert warmup_recs[0]["confidence"] == 0.80
