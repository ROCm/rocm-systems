"""Tests for perfxpert.tools.tracelens."""

from perfxpert.tools import tracelens
from perfxpert.tools._class import ToolClass


def test_high_idle_triggers_high_severity():
    r = tracelens.classify_overhead(idle_pct=0.25, wasted_pct=0.02)
    assert r["idle_severity"] == "high"
    assert len(r["recommended_actions"]) >= 1


def test_low_overhead_returns_low_severity():
    r = tracelens.classify_overhead(idle_pct=0.05, wasted_pct=0.02)
    assert r["idle_severity"] == "low"


def test_lookup_returns_catalog():
    catalog = tracelens.lookup_metrics()
    assert "interval_timeline" in catalog


def test_is_read_only_class():
    assert tracelens.classify_overhead.__tool_class__ == ToolClass.READ_ONLY
    assert tracelens.lookup_metrics.__tool_class__ == ToolClass.READ_ONLY
