# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for memory-chart renderers.

Covers:
- mem_chart_gfx11.py - RDNA3.5 (Rich-based) memory architecture visualization
- mem_chart_gfx9.py  - CDNA (plotille-based) memory architecture visualization
"""

import re

from utils import mem_chart_gfx9, mem_chart_gfx11

ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


# =============================================================================
# Tests for format_bw_human_readable function (gfx11)
# =============================================================================


class TestFormatBwHumanReadable:
    """Tests for format_bw_human_readable - converts Bytes/s to human-readable."""

    def test_terabytes_per_second(self):
        """Test TB/s formatting for values >= 1e12."""
        # 1 TB/s
        result = mem_chart_gfx11.format_bw_human_readable(1e12, "Bytes/s", 1)
        assert result == "1.0 TB/s"

        # 2.5 TB/s
        result = mem_chart_gfx11.format_bw_human_readable(2.5e12, "Bytes/s", 1)
        assert result == "2.5 TB/s"

        # Large value with higher precision
        result = mem_chart_gfx11.format_bw_human_readable(1.234e12, "Bytes/s", 2)
        assert result == "1.23 TB/s"

    def test_gigabytes_per_second(self):
        """Test GB/s formatting for values >= 1e9 and < 1e12."""
        # 1 GB/s
        result = mem_chart_gfx11.format_bw_human_readable(1e9, "Bytes/s", 1)
        assert result == "1.0 GB/s"

        # 100 GB/s (typical HBM bandwidth)
        result = mem_chart_gfx11.format_bw_human_readable(100e9, "Bytes/s", 1)
        assert result == "100.0 GB/s"

        # 512.5 GB/s
        result = mem_chart_gfx11.format_bw_human_readable(512.5e9, "Bytes/s", 1)
        assert result == "512.5 GB/s"

    def test_megabytes_per_second(self):
        """Test MB/s formatting for values >= 1e6 and < 1e9."""
        # 1 MB/s
        result = mem_chart_gfx11.format_bw_human_readable(1e6, "Bytes/s", 1)
        assert result == "1.0 MB/s"

        # 500 MB/s
        result = mem_chart_gfx11.format_bw_human_readable(500e6, "Bytes/s", 1)
        assert result == "500.0 MB/s"

    def test_kilobytes_per_second(self):
        """Test KB/s formatting for values >= 1e3 and < 1e6."""
        # 1 KB/s
        result = mem_chart_gfx11.format_bw_human_readable(1e3, "Bytes/s", 1)
        assert result == "1.0 KB/s"

        # 512 KB/s
        result = mem_chart_gfx11.format_bw_human_readable(512e3, "Bytes/s", 1)
        assert result == "512.0 KB/s"

    def test_bytes_per_second(self):
        """Test B/s formatting for values < 1e3."""
        # 500 B/s
        result = mem_chart_gfx11.format_bw_human_readable(500, "Bytes/s", 1)
        assert result == "500.0 B/s"

        # 0 B/s
        result = mem_chart_gfx11.format_bw_human_readable(0, "Bytes/s", 1)
        assert result == "0.0 B/s"

    def test_legacy_gbps_unit_conversion(self):
        """Test conversion from legacy GB/s unit to human-readable."""
        # Input is 100 GB/s, should convert to Bytes/s first then format
        result = mem_chart_gfx11.format_bw_human_readable(100, "GB/s", 1)
        assert result == "100.0 GB/s"

        # 1500 GB/s -> 1.5 TB/s
        result = mem_chart_gfx11.format_bw_human_readable(1500, "GB/s", 1)
        assert result == "1.5 TB/s"

    def test_none_value(self):
        """Test handling of None values."""
        result = mem_chart_gfx11.format_bw_human_readable(None, "Bytes/s", 1)
        assert result == "N/A"

    def test_invalid_string_value(self):
        """Test handling of non-numeric string values."""
        result = mem_chart_gfx11.format_bw_human_readable("invalid", "Bytes/s", 1)
        assert result == "N/A"

    def test_precision_parameter(self):
        """Test different precision values."""
        value = 123.456789e9  # 123.456789 GB/s

        assert (
            mem_chart_gfx11.format_bw_human_readable(value, "Bytes/s", 0) == "123 GB/s"
        )
        assert (
            mem_chart_gfx11.format_bw_human_readable(value, "Bytes/s", 1)
            == "123.5 GB/s"
        )
        assert (
            mem_chart_gfx11.format_bw_human_readable(value, "Bytes/s", 2)
            == "123.46 GB/s"
        )
        assert (
            mem_chart_gfx11.format_bw_human_readable(value, "Bytes/s", 3)
            == "123.457 GB/s"
        )


# =============================================================================
# Tests for format_value function (gfx11)
# =============================================================================


class TestFormatValue:
    """Tests for format_value - general value formatting."""

    def test_percentage_formatting(self):
        """Test percentage formatting."""
        result = mem_chart_gfx11.format_value(85.5, "%", 1)
        assert result == "85.5%"

        result = mem_chart_gfx11.format_value(100, "%", 0)
        assert result == "100%"

    def test_bytes_per_second_unit(self):
        """Test Bytes/s formatting routes to human-readable."""
        result = mem_chart_gfx11.format_value(100e9, "Bytes/s", 1)
        assert "GB/s" in result

    def test_gbps_unit(self):
        """Test GB/s formatting routes to human-readable."""
        result = mem_chart_gfx11.format_value(100, "GB/s", 1)
        assert "GB/s" in result

    def test_none_value(self):
        """Test handling of None values."""
        result = mem_chart_gfx11.format_value(None, "%", 1)
        assert result == "N/A"

    def test_string_value_conversion(self):
        """Test conversion of string values to float."""
        result = mem_chart_gfx11.format_value("50.5", "%", 1)
        assert result == "50.5%"

    def test_invalid_string_value(self):
        """Test handling of non-convertible string values."""
        result = mem_chart_gfx11.format_value("invalid", "%", 1)
        assert result == "invalid"


# =============================================================================
# Tests for format_sci function (gfx11)
# =============================================================================


class TestFormatSci:
    """Tests for format_sci - scientific notation formatting."""

    def test_small_numbers_no_scientific(self):
        """Test that small numbers are displayed as integers."""
        assert mem_chart_gfx11.format_sci(100) == "100"
        assert mem_chart_gfx11.format_sci(999) == "999"

    def test_large_numbers_scientific(self):
        """Test that large numbers are displayed in scientific notation."""
        result = mem_chart_gfx11.format_sci(1000000)
        assert "e" in result.lower()

        result = mem_chart_gfx11.format_sci(12345678, 2)
        assert "e" in result.lower()

    def test_none_value(self):
        """Test handling of None values."""
        result = mem_chart_gfx11.format_sci(None)
        assert result == "N/A"

    def test_invalid_value(self):
        """Test handling of invalid values."""
        result = mem_chart_gfx11.format_sci("invalid")
        assert result == "N/A"

    def test_negative_numbers(self):
        """Test negative number handling."""
        result = mem_chart_gfx11.format_sci(-500)
        assert result == "-500"

        result = mem_chart_gfx11.format_sci(-1000000)
        assert "e" in result.lower()


# =============================================================================
# Tests for bar function (gfx11)
# =============================================================================


class TestBar:
    """Tests for bar - progress bar visualization."""

    def test_full_bar(self):
        """Test 100% progress bar."""
        result = mem_chart_gfx11.bar(100, 10)
        assert "█" * 10 in result
        assert "░" not in result

    def test_empty_bar(self):
        """Test 0% progress bar."""
        result = mem_chart_gfx11.bar(0, 10)
        assert "░" * 10 in result
        assert "█" not in result

    def test_partial_bar(self):
        """Test 50% progress bar."""
        result = mem_chart_gfx11.bar(50, 10)
        assert "█" * 5 in result
        assert "░" * 5 in result

    def test_none_value(self):
        """Test None value returns empty bar."""
        result = mem_chart_gfx11.bar(None, 10)
        assert "░" * 10 in result

    def test_invalid_value(self):
        """Test invalid value returns empty bar."""
        result = mem_chart_gfx11.bar("invalid", 10)
        assert "░" * 10 in result

    def test_over_100_clamped(self):
        """Test values over 100% are clamped."""
        result = mem_chart_gfx11.bar(150, 10)
        assert "█" * 10 in result

    def test_negative_clamped(self):
        """Test negative values are clamped to 0."""
        result = mem_chart_gfx11.bar(-50, 10)
        assert "░" * 10 in result


# =============================================================================
# Tests for metric_line function (gfx11)
# =============================================================================


class TestMetricLine:
    """Tests for metric_line - formatted metric display."""

    def test_basic_metric(self):
        """Test basic metric line formatting."""
        result = mem_chart_gfx11.metric_line("Util", 75.5, "%", "green")
        assert "Util" in result
        assert "75.5%" in result
        assert "green" in result

    def test_with_none_value(self):
        """Test metric line with None value."""
        result = mem_chart_gfx11.metric_line("BW", None, "GB/s", "cyan")
        assert "BW" in result
        assert "N/A" in result


# =============================================================================
# Tests for get_sample_metrics function (gfx11)
# =============================================================================


class TestGetSampleMetrics:
    """Tests for get_sample_metrics - returns sample test data."""

    def test_returns_dict(self):
        """Test that sample metrics returns a dictionary."""
        metrics = mem_chart_gfx11.get_sample_metrics()
        assert isinstance(metrics, dict)

    def test_contains_key_metrics(self):
        """Test that sample metrics contains expected keys."""
        metrics = mem_chart_gfx11.get_sample_metrics()

        # Check for key bandwidth metrics
        assert "TCP-GL1 Read Bandwidth" in metrics
        assert "GL1-GL2 Read Bandwidth" in metrics
        assert "GL2C Read Bandwidth" in metrics
        assert "DRAM Read Bandwidth" in metrics

        # Check for utilization metrics
        assert "GL1C Utilization" in metrics
        assert "GL2C Utilization" in metrics

        # Check for hit rate metrics
        assert "TCP Hit Rate" in metrics
        assert "GL1C Hit Rate" in metrics
        assert "GL2C Hit Rate" in metrics

    def test_bandwidth_values_in_bytes_per_second(self):
        """Test that bandwidth values are in Bytes/s format."""
        metrics = mem_chart_gfx11.get_sample_metrics()

        # TCP-GL1 Read Bandwidth should be 96 GB/s = 96e9 Bytes/s
        assert metrics["TCP-GL1 Read Bandwidth"] == 96e9

        # DRAM Read Bandwidth should be 100 GB/s = 100e9 Bytes/s
        assert metrics["DRAM Read Bandwidth"] == 100e9

    def test_returns_copy(self):
        """Test that sample metrics returns a copy (not the original)."""
        metrics1 = mem_chart_gfx11.get_sample_metrics()
        metrics2 = mem_chart_gfx11.get_sample_metrics()

        # Modify metrics1 and verify metrics2 is not affected
        metrics1["TCP-GL1 Read Bandwidth"] = 0
        assert metrics2["TCP-GL1 Read Bandwidth"] == 96e9


# =============================================================================
# Tests for plot_mem_chart function (gfx11)
# =============================================================================


class TestPlotMemChartGfx11:
    """Tests for gfx11 plot_mem_chart - main chart generation."""

    def test_returns_string(self):
        """Test that plot_mem_chart returns a string."""
        metrics = mem_chart_gfx11.get_sample_metrics()
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", metrics)
        clean = strip_ansi(result)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "3. Memory Chart" in clean
        assert "Normalization: per_kernel" in clean
        assert "GPU" in clean and "System Memory" in clean

    def test_chart_title_override(self):
        """Explicit chart_title appears in output."""
        metrics = mem_chart_gfx11.get_sample_metrics()
        result = mem_chart_gfx11.plot_mem_chart(
            "per_kernel",
            metrics,
            chart_title="3. Memory Chart (Normalization: per_kernel)",
        )
        assert "3. Memory Chart (Normalization: per_kernel)" in strip_ansi(result)

    def test_contains_architecture_elements(self):
        """Test that output contains RDNA3.5 architecture elements."""
        metrics = mem_chart_gfx11.get_sample_metrics()
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", metrics)

        # Check for key components
        assert "TCP" in result or "L0" in result  # L0 cache
        assert "GL1C" in result  # L1 cache
        assert "GL2C" in result  # L2 cache
        assert (
            "GCEA" in result
        )  # Graphics Core Efficiency Arbiter (block label in diagram)
        assert "DRAM" in result  # System memory

    def test_contains_bandwidth_values(self):
        """Test that output contains formatted bandwidth values."""
        metrics = mem_chart_gfx11.get_sample_metrics()
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", metrics)

        # Should contain GB/s units since sample data uses GB/s range
        assert "GB/s" in result

    def test_normalize_mem_chart_metrics_flat_ordered(self):
        """Metrics are flattened to panel YAML order; extras dropped; missing None."""
        raw = {"TCP Hit Rate": 1.0, "noise_key": 99}
        norm = mem_chart_gfx11.normalize_mem_chart_metrics(raw)
        assert list(norm.keys()) == list(mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS)
        assert norm["TCP Hit Rate"] == 1.0
        assert "noise_key" not in norm
        assert norm["ICache Requests"] is None

    def test_empty_metrics(self):
        """Test with empty metrics dictionary."""
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", {})

        # Should still produce output (with N/A values)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_partial_metrics(self):
        """Test with partial metrics (some missing)."""
        partial_metrics = {
            "TCP-GL1 Read Bandwidth": 50e9,
            "GL1C Utilization": 65.0,
            # Missing many other metrics
        }
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", partial_metrics)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_extreme_bandwidth_values(self):
        """Test with extreme bandwidth values."""
        extreme_metrics = {
            "DRAM Read Bandwidth": 10e12,  # 10 TB/s
            "DRAM Write Bandwidth": 5e12,  # 5 TB/s
        }
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", extreme_metrics)

        assert "TB/s" in result

    def test_zero_bandwidth_values(self):
        """Test with zero bandwidth values."""
        zero_metrics = {
            "DRAM Read Bandwidth": 0,
            "DRAM Write Bandwidth": 0,
        }
        result = mem_chart_gfx11.plot_mem_chart("per_kernel", zero_metrics)

        assert "B/s" in result  # Zero formats as "0.0 B/s"


# =============================================================================
# Tests for DEFAULT_SAMPLE_METRICS constant (gfx11)
# =============================================================================


class TestDefaultSampleMetrics:
    """Tests for DEFAULT_SAMPLE_METRICS constant."""

    def test_keys_match_mem_chart_panel_yaml(self):
        """Keys match gfx1151 Memory Chart YAML (MEM_CHART_PANEL_METRIC_KEYS)."""
        assert set(mem_chart_gfx11.DEFAULT_SAMPLE_METRICS) == set(
            mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS
        )
        assert len(mem_chart_gfx11.DEFAULT_SAMPLE_METRICS) == len(
            mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS
        )

    def test_all_bandwidth_values_positive(self):
        """Test that all bandwidth values are positive."""
        for key, value in mem_chart_gfx11.DEFAULT_SAMPLE_METRICS.items():
            if "Bandwidth" in key:
                assert value >= 0, f"{key} should be non-negative"

    def test_all_rate_values_in_range(self):
        """Test that rate/percentage values are in valid range."""
        for key, value in mem_chart_gfx11.DEFAULT_SAMPLE_METRICS.items():
            if "Rate" in key or "Utilization" in key:
                assert 0 <= value <= 100, f"{key} should be between 0 and 100"

    def test_has_all_memory_hierarchy_levels(self):
        """Test that all memory hierarchy levels are represented."""
        metrics = mem_chart_gfx11.DEFAULT_SAMPLE_METRICS

        # TCP (L0)
        assert any("TCP" in k for k in metrics.keys())

        # LDS
        assert any("LDS" in k for k in metrics.keys())

        # GL1C
        assert any("GL1" in k for k in metrics.keys())

        # GL2C
        assert any("GL2" in k for k in metrics.keys())

        # DRAM
        assert any("DRAM" in k for k in metrics.keys())


# =============================================================================
# Integration Tests (gfx11)
# =============================================================================


class TestIntegrationGfx11:
    """Integration tests for complete gfx11 workflows."""

    def test_full_workflow_with_sample_data(self):
        """Test complete workflow with sample data."""
        # Get sample metrics
        metrics = mem_chart_gfx11.get_sample_metrics()

        # Generate chart
        chart = mem_chart_gfx11.plot_mem_chart("per_dispatch", metrics)

        # Verify chart contains expected elements
        assert isinstance(chart, str)
        assert len(chart) > 100  # Should be substantial output
        assert "Kernel" in chart
        assert "Legend" in chart

    def test_bandwidth_unit_consistency(self):
        """Test that bandwidth units are consistently formatted."""
        # Create metrics with known bandwidth values
        metrics = {
            "TCP-GL1 Read Bandwidth": 100e9,  # 100 GB/s
            "TCP-GL1 Write Bandwidth": 50e9,  # 50 GB/s
            "GL1-GL2 Read Bandwidth": 75e9,  # 75 GB/s
            "GL1-GL2 Write Bandwidth": 25e9,  # 25 GB/s
            "DRAM Read Bandwidth": 200e9,  # 200 GB/s
            "DRAM Write Bandwidth": 100e9,  # 100 GB/s
        }

        chart = mem_chart_gfx11.plot_mem_chart("per_kernel", metrics)

        # All values should show in GB/s since they're in that range
        assert chart.count("GB/s") >= 6

    def test_mixed_bandwidth_scales(self):
        """Test chart with mixed bandwidth scales."""
        metrics = {
            "DRAM Read Bandwidth": 1.5e12,  # 1.5 TB/s
            "GL2C Read Bandwidth": 500e9,  # 500 GB/s
            "GL1-GL2 Read Bandwidth": 100e6,  # 100 MB/s
            "TCP-GL1 Read Bandwidth": 50e3,  # 50 KB/s
        }

        chart = mem_chart_gfx11.plot_mem_chart("per_kernel", metrics)

        # Should contain multiple unit types
        assert "TB/s" in chart
        assert "GB/s" in chart


# =============================================================================
# Tests for plot_mem_chart function (gfx9)
# =============================================================================


GFX9_SAMPLE_METRICS = {
    "Wavefront Occupancy": 1,
    "Wave Life": 2,
    "SALU": 3,
    "SMEM": 4,
    "VALU": 5,
    "Matrix Ops": 6,
    "VMEM": 7,
    "LDS": 8,
    "GWS": 9,
    "BR": 10,
    "Active CUs": 11,
    "Num CUs": 12,
    "VGPR": 13,
    "SGPR": 14,
    "LDS Allocation": 15,
    "Scratch Allocation": 16,
    "Wavefronts": 17,
    "Workgroups": 18,
    "LDS Req": 19,
    "LDS Util": 20,
    "LDS Latency": 21,
    "VL1 Rd": 22,
    "VL1 Wr": 23,
    "VL1 Atomic": 24,
    "VL1 Hit": 25,
    "VL1 Lat": 26,
    "VL1 Coalesce": 27,
    "VL1 Stall": 28,
    "sL1D Rd": 29,
    "sL1D Hit": 30,
    "sL1D Lat": 31,
    "IL1 Fetch": 32,
    "IL1 Hit": 33,
    "IL1 Lat": 34,
    "VL1_L2 Rd": 36,
    "VL1_L2 Wr": 37,
    "VL1_L2 Atomic": 38,
    "sL1D_L2 Rd": 39,
    "sL1D_L2 Wr": 40,
    "sL1D_L2 Atomic": 41,
    "IL1_L2 Rd": 42,
    "L2 Hit": 43,
    "L2 Rd": 44,
    "L2 Wr": 45,
    "L2 Atomic": 46,
    "L2 Rd Lat": 47,
    "L2 Wr Lat": 48,
    "Fabric_L2 Rd": 49,
    "Fabric_L2 Wr": 50,
    "Fabric_L2 Atomic": 51,
    "Fabric Rd Lat": 52,
    "Fabric Wr Lat": 53,
    "Fabric Atomic Lat": 54,
    "HBM Rd": 55,
    "HBM Wr": 56,
}


class TestPlotMemChartGfx9:
    """Tests for gfx9 plot_mem_chart - CDNA memory chart generation."""

    def test_returns_non_empty_string(self):
        """Full sample metrics produce a non-empty chart string."""
        result = mem_chart_gfx9.plot_mem_chart("per_kernel", dict(GFX9_SAMPLE_METRICS))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_metrics(self):
        """Empty metric dict still produces a non-empty chart (N/A placeholders)."""
        result = mem_chart_gfx9.plot_mem_chart("per_kernel", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_partial_metrics(self):
        """Partial metric dict still produces a non-empty chart."""
        partial = {
            "Wavefront Occupancy": 4,
            "L2 Hit": 75,
            "HBM Rd": 100,
        }
        result = mem_chart_gfx9.plot_mem_chart("per_kernel", partial)
        assert isinstance(result, str)
        assert len(result) > 0
