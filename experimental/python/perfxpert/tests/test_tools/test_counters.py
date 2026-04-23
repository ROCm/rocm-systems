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
    assert any(v.get("auto_fixed") for v in result["violations"])


def test_validate_unknown_counter_fails():
    result = counters.validate_for_gpu(["NOT_A_COUNTER"], gpu_arch="gfx942")
    assert result["ok"] is False
    assert "Unknown counters" in result["violations"][0]["reason"]


def test_validate_balances_regular_counters_across_passes():
    counters_in = [
        "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_LDS",
        "TCC_HIT", "TCC_MISS", "TCC_EA0_RDREQ", "TCC_EA0_WRREQ", "TCC_BUBBLE",
    ]
    result = counters.validate_for_gpu(counters_in, gpu_arch="gfx908")
    assert result["ok"] is True
    assert len(result["fixed_passes"]) == 2


def test_is_read_only_class():
    assert counters.lookup_info.__tool_class__ == ToolClass.READ_ONLY
    assert counters.validate_for_gpu.__tool_class__ == ToolClass.READ_ONLY
