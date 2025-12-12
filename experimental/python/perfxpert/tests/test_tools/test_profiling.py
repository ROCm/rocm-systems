"""Tests for perfxpert.tools.profiling."""

from perfxpert.tools import profiling
from perfxpert.tools._class import ToolClass


def test_empty_fingerprint_starts_with_cheapest():
    r = profiling.fill_gap(current_fingerprint=set(), goal="any")
    assert r["cost_s"] == 0


def test_with_sys_trace_recommends_next():
    r = profiling.fill_gap(current_fingerprint={"--sys-trace", "--hip-trace", "--kernel-trace"}, goal="pmc")
    # Should recommend rocprofv3 --pmc (tier 2)
    assert "pmc" in r["name"].lower() or "source_scan" in r["name"]


def test_is_read_only_class():
    assert profiling.fill_gap.__tool_class__ == ToolClass.READ_ONLY
