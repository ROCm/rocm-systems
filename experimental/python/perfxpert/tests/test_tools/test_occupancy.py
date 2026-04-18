"""Tests for perfxpert.tools.occupancy."""

import pytest

from perfxpert.tools import occupancy
from perfxpert.tools._class import ToolClass


def test_lookup_vgpr_32_gives_8_waves_on_cdna3():
    assert occupancy.lookup_waves_per_eu(32, "gfx942") == 8


def test_lookup_vgpr_128_gives_2_waves_on_cdna3():
    assert occupancy.lookup_waves_per_eu(128, "gfx942") == 2


def test_lookup_unknown_arch_raises():
    with pytest.raises(KeyError):
        occupancy.lookup_waves_per_eu(32, "gfx9999")


def test_suggest_reduction_for_hot_kernel_with_high_vgpr():
    r = occupancy.suggest_vgpr_reduction(current_vgpr=96, occupancy_pct=0.25, kernel_time_pct=0.50)
    assert r["applicable"]
    assert r["target_vgpr"] < 96
    assert r["expected_occupancy_multiplier"] > 1.0


def test_suggest_skips_small_kernel():
    r = occupancy.suggest_vgpr_reduction(current_vgpr=96, occupancy_pct=0.25, kernel_time_pct=0.02)
    assert not r["applicable"]


def test_is_read_only_class():
    assert occupancy.lookup_waves_per_eu.__tool_class__ == ToolClass.READ_ONLY
    assert occupancy.suggest_vgpr_reduction.__tool_class__ == ToolClass.READ_ONLY
