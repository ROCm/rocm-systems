"""Tests for perfxpert.tools.roofline."""

import pytest

from perfxpert.tools import roofline
from perfxpert.tools._class import ToolClass


def test_compute_bound_when_ai_above_ridge():
    # MI300X default ridge uses FP32 vector peak: 30.8 FLOPS/Byte.
    r = roofline.classify(flops=3.6e12, bytes=1e11, gfx_id="gfx942")
    assert r["regime"] == "compute"
    assert r["arithmetic_intensity"] > 30


def test_memory_bound_when_ai_below_ridge():
    # AI=2 → well below MI300X ridge 30.8 → memory-bound
    r = roofline.classify(flops=1e12, bytes=5e11, gfx_id="gfx942")
    assert r["regime"] == "memory"
    assert r["arithmetic_intensity"] < 5


def test_at_ridge_point_balanced():
    # AI close to ridge point → both/balanced
    r = roofline.classify(flops=3.08e12, bytes=1e11, gfx_id="gfx942")
    # AI = 30.8 → regime should be "balanced" or "compute" (tie-break toward compute)
    assert r["arithmetic_intensity"] == pytest.approx(30.8, rel=0.01)
    assert r["regime"] in ("compute", "balanced")


def test_zero_bytes_raises():
    with pytest.raises(ValueError, match="bytes"):
        roofline.classify(flops=1e12, bytes=0, gfx_id="gfx942")


def test_unknown_arch_raises():
    with pytest.raises(KeyError):
        roofline.classify(flops=1e12, bytes=1e11, gfx_id="gfx9999")


def test_is_read_only_class():
    assert roofline.classify.__tool_class__ == ToolClass.READ_ONLY


def test_returns_distance_to_roof():
    r = roofline.classify(flops=1e12, bytes=1e11, gfx_id="gfx942")
    assert "distance_to_roof" in r
    assert 0 <= r["distance_to_roof"] <= 1  # normalized
