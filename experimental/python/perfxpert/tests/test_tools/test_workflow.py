"""Tests for perfxpert.tools.workflow."""

import pytest

from perfxpert.tools import workflow
from perfxpert.tools._class import ToolClass


def test_tier0_collect_returns_rocprofv3():
    r = workflow.next_step(0, "collect")
    assert r["next_tier"] == 1
    assert "rocprofv3" in r["action"]


def test_tier1_measure_goes_to_counters():
    r = workflow.next_step(1, "measure")
    assert r["next_tier"] == 2
    assert "--pmc" in r["action"]


def test_unknown_combo_raises():
    with pytest.raises(KeyError):
        workflow.next_step(99, "do_magic")


def test_is_read_only_class():
    assert workflow.next_step.__tool_class__ == ToolClass.READ_ONLY
