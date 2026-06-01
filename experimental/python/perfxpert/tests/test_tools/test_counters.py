"""Tests for perfxpert.tools.counters."""

import pytest

from perfxpert.knowledge import load_yaml
from perfxpert.tools import counters
from perfxpert.tools._class import ToolClass


def _arch_with_limit_for(block: str, needed: int = 1) -> str:
    arch_limits = load_yaml("pmc_limits")["gpu_arch_limits"]
    return next(gfx_id for gfx_id, limits in arch_limits.items() if limits.get(block, 0) >= needed)


def test_lookup_info_known_counter():
    gfx_id = _arch_with_limit_for("SQ")
    info = counters.lookup_info("SQ_WAVES", gfx_id=gfx_id)
    assert info["name"] == "SQ_WAVES"
    assert info["block"] == "SQ"
    assert info["block_limit"] == load_yaml("pmc_limits")["gpu_arch_limits"][gfx_id]["SQ"]


def test_lookup_info_unknown_raises():
    with pytest.raises(KeyError):
        counters.lookup_info("DEFINITELY_NOT_A_COUNTER")


def test_validate_splits_tcc_derived_into_own_passes():
    """FETCH_SIZE + WRITE_SIZE must each be in their own pass (anti-Sakana)."""
    result = counters.validate_for_gpu(
        ["SQ_WAVES", "GRBM_COUNT", "FETCH_SIZE", "WRITE_SIZE"],
        gpu_arch=_arch_with_limit_for("SQ"),
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
        gpu_arch=_arch_with_limit_for("SQ"),
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
        gpu_arch=_arch_with_limit_for("SQ"),
    )
    assert result["fixed_passes"] == [["SQ_WAVES", "GRBM_COUNT"]]
    assert result["escalation"] is None


def test_validate_for_gpu_uses_arch_specific_regular_block_limit():
    catalog = load_yaml("counter_catalog")
    sq_counters = [entry["name"] for entry in catalog if entry["block"] == "SQ"]
    gfx_id = _arch_with_limit_for("SQ", needed=len(sq_counters))

    result = counters.validate_for_gpu(sq_counters, gpu_arch=gfx_id)

    assert result["fixed_passes"] == [sq_counters]
    assert result["escalation"] is None


def test_analysis_pmc_split_uses_same_arch_specific_limit():
    from perfxpert.analysis.pmc import _split_pmc_into_passes

    catalog = load_yaml("counter_catalog")
    sq_counters = [entry["name"] for entry in catalog if entry["block"] == "SQ"]
    gfx_id = _arch_with_limit_for("SQ", needed=len(sq_counters))

    commands = _split_pmc_into_passes(
        sq_counters,
        base_flags=[],
        base_args=[],
        output_dir="./out",
        output_prefix="profile",
        description="Collect counters",
        gpu_arch=gfx_id,
    )

    assert [arg["value"] for arg in commands[0]["args"] if arg["name"] == "--pmc"] == [" ".join(sq_counters)]
    assert len(commands) == 1


def test_is_read_only_class():
    assert counters.lookup_info.__tool_class__ == ToolClass.READ_ONLY
    assert counters.validate_for_gpu.__tool_class__ == ToolClass.READ_ONLY
