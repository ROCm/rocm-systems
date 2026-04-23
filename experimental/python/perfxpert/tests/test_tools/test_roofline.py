"""Tests for perfxpert.tools.roofline."""

import pytest

from perfxpert.tools import roofline
from perfxpert.tools._class import ToolClass


def test_compute_bound_when_ai_above_ridge():
    # MI300X ridge = 15.4 FLOPS/Byte; ai=30 → compute-bound
    r = roofline.classify(flops=1e12, bytes=3e10, gfx_id="gfx942")
    assert r["regime"] == "compute"
    assert r["arithmetic_intensity"] > 30


def test_memory_bound_when_ai_below_ridge():
    # AI=2 → well below MI300X ridge 15.4 → memory-bound
    r = roofline.classify(flops=1e12, bytes=5e11, gfx_id="gfx942")
    assert r["regime"] == "memory"
    assert r["arithmetic_intensity"] < 5


def test_at_ridge_point_balanced():
    # AI close to ridge point → both/balanced
    specs = {"gfx942": {"ridge_point": 15.4}}
    r = roofline.classify(flops=1.54e12, bytes=1e11, gfx_id="gfx942")
    # AI = 15.4 → regime should be "balanced" or "compute" (tie-break toward compute)
    assert r["arithmetic_intensity"] == pytest.approx(15.4, rel=0.01)
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
