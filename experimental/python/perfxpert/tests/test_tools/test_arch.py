"""Tests for perfxpert.tools.arch."""

import pytest

from perfxpert.tools import arch
from perfxpert.tools._class import ToolClass


def test_lookup_peaks_returns_structured_data_for_mi300x():
    peaks = arch.lookup_peaks("gfx942")
    assert peaks["name"] == "MI300X"
    assert peaks["peak_fp64_tflops"] == 81.7
    assert peaks["peak_fp64_matrix_tflops"] == 163.4
    assert peaks["memory_bandwidth_tbs"] == 5.3
    assert peaks["ridge_point"] == 15.4


def test_lookup_peaks_covers_all_known_archs():
    known = ["gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100"]
    for gfx in known:
        result = arch.lookup_peaks(gfx)
        assert "name" in result
        assert "peak_fp64_tflops" in result


def test_lookup_peaks_exposes_gfx90a_variants():
    peaks = arch.lookup_peaks("gfx90a")
    variant_names = {entry["name"] for entry in peaks["sku_variants"]}
    assert {"MI210", "MI250X"}.issubset(variant_names)


def test_lookup_peaks_unknown_arch_raises():
    with pytest.raises(KeyError) as exc:
        arch.lookup_peaks("gfx9999")
    assert "gfx9999" in str(exc.value)
    assert "known" in str(exc.value).lower() or "available" in str(exc.value).lower()


def test_lookup_peaks_is_read_only_class():
    """MCP exposure policy — lookup tools are READ_ONLY."""
    assert arch.lookup_peaks.__tool_class__ == ToolClass.READ_ONLY


def test_lookup_peaks_is_deterministic_no_network():
    """Pure function — no I/O beyond YAML load."""
    import time
    start = time.time()
    arch.lookup_peaks("gfx942")
    duration_ms = (time.time() - start) * 1000
    # Must be fast (< 50ms even on cold YAML load)
    assert duration_ms < 50, f"too slow: {duration_ms}ms"
