"""Tests for perfxpert.tools.att."""

import pytest

from perfxpert.tools import att
from perfxpert.tools._class import ToolClass


def test_classify_vmem_interlock():
    r = att.classify_stall_reason("INTERLOCK_VMEM")
    assert "VMEM" in r["name"]
    assert r["mitigation"]


def test_unknown_stall_code_raises():
    with pytest.raises(KeyError):
        att.classify_stall_reason("NOT_A_REAL_CODE")


def test_classify_stall_ratio_critical():
    assert att.classify_stall_ratio(0.85)["severity"] == "critical"


def test_classify_stall_ratio_low():
    assert att.classify_stall_ratio(0.20)["severity"] == "low"


def test_is_read_only_class():
    assert att.classify_stall_reason.__tool_class__ == ToolClass.READ_ONLY
    assert att.classify_stall_ratio.__tool_class__ == ToolClass.READ_ONLY
