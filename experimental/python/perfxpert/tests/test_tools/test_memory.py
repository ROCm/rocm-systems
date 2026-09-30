"""Tests for perfxpert.tools.memory."""

from perfxpert.tools import memory
from perfxpert.tools._class import ToolClass


def test_good_hit_rates_classify_as_good():
    r = memory.classify_cache_performance(l1_hit_rate=0.90, l2_hit_rate=0.85)
    assert r["l1"]["severity"] == "good"
    assert r["l2"]["severity"] == "good"
    assert r["overall_severity"] == "good"


def test_low_l1_classifies_as_critical():
    r = memory.classify_cache_performance(l1_hit_rate=0.10, l2_hit_rate=0.90)
    assert r["l1"]["severity"] == "critical"
    assert r["overall_severity"] == "critical"


def test_is_read_only_class():
    assert memory.classify_cache_performance.__tool_class__ == ToolClass.READ_ONLY
