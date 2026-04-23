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


def test_validate_multi_pass_returns_escalation_guidance():
    """Multi-pass plans must include concrete escalation guidance."""
    result = counters.validate_for_gpu(
        ["SQ_WAVES", "GRBM_COUNT", "FETCH_SIZE", "WRITE_SIZE"],
        gpu_arch="gfx942",
    )

    escalation = result["escalation"]
    assert escalation["required"] is True
    assert escalation["pass_count"] == 3
    assert escalation["pmc_groups_path"] == "pmc_groups.txt"
    assert escalation["pmc_groups"] == [
        "pmc: SQ_WAVES GRBM_COUNT",
        "pmc: FETCH_SIZE",
        "pmc: WRITE_SIZE",
    ]
    commands = {cmd["tool"]: cmd["full_command"] for cmd in escalation["commands"]}
    assert commands["rocprof-compute"] == "rocprof-compute profile -- ./app"
    assert commands["rocprofv3"] == "rocprofv3 -i pmc_groups.txt -- ./app"


def test_validate_single_pass_does_not_emit_escalation():
    """Single-pass plans should remain simple and omit escalation."""
    result = counters.validate_for_gpu(
        ["SQ_WAVES", "GRBM_COUNT"],
        gpu_arch="gfx942",
    )
    assert result["fixed_passes"] == [["SQ_WAVES", "GRBM_COUNT"]]
    assert result["escalation"] is None


def test_is_read_only_class():
    assert counters.lookup_info.__tool_class__ == ToolClass.READ_ONLY
    assert counters.validate_for_gpu.__tool_class__ == ToolClass.READ_ONLY
