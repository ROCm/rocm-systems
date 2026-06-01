"""Tests for perfxpert.tools.topdown."""

from perfxpert.tools import topdown
from perfxpert.tools._class import ToolClass


def test_memcpy_above_threshold_triggers_red_flag():
    flags = topdown.classify_overhead(memcpy_pct=0.30, api_pct=0.05, idle_pct=0.05)
    names = {f["red_flag_name"] for f in flags}
    assert "memcpy_dominant" in names


def test_no_red_flags_when_all_healthy():
    flags = topdown.classify_overhead(memcpy_pct=0.05, api_pct=0.05, idle_pct=0.05)
    assert flags == []


def test_is_read_only_class():
    assert topdown.classify_overhead.__tool_class__ == ToolClass.READ_ONLY
