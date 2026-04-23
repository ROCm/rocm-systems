"""Tests for perfxpert.tools.sol — including anti-Sakana reward-hacking defense."""

import pytest

from perfxpert.tools import sol
from perfxpert.tools._class import ToolClass


# -- sanity_check (anti-Sakana) --------------------------------------------

def test_claimed_speedup_within_sol_passes():
    """Normal optimization — 2× speedup on fp64 kernel ~ 40 TFLOPS on MI300X (peak 81.7)."""
    r = sol.sanity_check(
        achieved_flops_per_sec=40e12,
        kernel_type="fp64",
        gfx_id="gfx942",
    )
    assert r["plausible"] is True


def test_claimed_speedup_exceeding_sol_rejected():
    """Sakana-style: claim 500 TFLOPS fp64 on MI300X whose peak is 81.7."""
    r = sol.sanity_check(
        achieved_flops_per_sec=500e12,
        kernel_type="fp64",
        gfx_id="gfx942",
    )
    assert r["plausible"] is False
    assert "exceeds" in r["reason"].lower() or "peak" in r["reason"].lower()
    assert r["sol_peak"] == 81.7e12


def test_fp32_variant():
    """MI300X FP32 peak = 163.4 TFLOPS."""
    r = sol.sanity_check(
        achieved_flops_per_sec=150e12,
        kernel_type="fp32",
        gfx_id="gfx942",
    )
    assert r["plausible"] is True


def test_bf16_variant():
    """MI300X BF16 peak = 1307 TFLOPS (matrix)."""
    r = sol.sanity_check(
        achieved_flops_per_sec=1000e12,
        kernel_type="bf16",
        gfx_id="gfx942",
    )
    assert r["plausible"] is True


def test_unknown_kernel_type_raises():
    # gpu_specs.yaml now exposes peak_fp16_tflops on every arch (Phase
    # 10 Live-Roofline extension) so "fp16" is no longer an unknown
    # kernel_type. Use a token that will stay unknown forever.
    with pytest.raises(ValueError):
        sol.sanity_check(achieved_flops_per_sec=1e12, kernel_type="fp42", gfx_id="gfx942")


# -- classify_utilization --------------------------------------------------

def test_utilization_high():
    assert sol.classify_utilization(0.85)["category"] == "high"


def test_utilization_medium():
    assert sol.classify_utilization(0.60)["category"] == "medium"


def test_utilization_low():
    assert sol.classify_utilization(0.20)["category"] == "low"


def test_utilization_out_of_range_raises():
    with pytest.raises(ValueError):
        sol.classify_utilization(1.5)


def test_is_read_only_class():
    assert sol.sanity_check.__tool_class__ == ToolClass.READ_ONLY
    assert sol.classify_utilization.__tool_class__ == ToolClass.READ_ONLY
