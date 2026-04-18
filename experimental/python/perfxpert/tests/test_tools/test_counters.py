"""Tests for perfxpert.tools.counters."""

import pytest

from perfxpert.tools import counters
from perfxpert.tools._class import ToolClass


def test_lookup_info_known_counter():
    info = counters.lookup_info("SQ_WAVES")
    assert info["name"] == "SQ_WAVES"
    assert info["block"] == "SQ"


def test_lookup_info_unknown_raises():
    with pytest.raises(KeyError):
        counters.lookup_info("DEFINITELY_NOT_A_COUNTER")


def test_validate_splits_tcc_derived_into_own_passes():
    """FETCH_SIZE + WRITE_SIZE must each be in their own pass (anti-Sakana)."""
    result = counters.validate_for_gpu(
        ["SQ_WAVES", "GRBM_COUNT", "FETCH_SIZE", "WRITE_SIZE"],
        gpu_arch="gfx942",
    )
    assert result["ok"]
    # FETCH_SIZE and WRITE_SIZE must be in separate passes
    fetch_passes = [p for p in result["fixed_passes"] if "FETCH_SIZE" in p]
    write_passes = [p for p in result["fixed_passes"] if "WRITE_SIZE" in p]
    assert len(fetch_passes) == 1
    assert len(write_passes) == 1
    assert fetch_passes[0] != write_passes[0]


def test_is_read_only_class():
    assert counters.lookup_info.__tool_class__ == ToolClass.READ_ONLY
    assert counters.validate_for_gpu.__tool_class__ == ToolClass.READ_ONLY
