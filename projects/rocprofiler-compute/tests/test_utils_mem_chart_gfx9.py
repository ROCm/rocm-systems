# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for mem_chart_gfx9.py - CDNA (plotille-based) memory architecture
visualization.
"""

from utils import mem_chart_gfx9

# =============================================================================
# Sample data for gfx9 tests
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


# =============================================================================
# Tests for plot_mem_chart function (gfx9)
# =============================================================================


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
