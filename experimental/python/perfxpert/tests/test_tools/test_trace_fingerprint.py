"""Tests for perfxpert.tools.trace_fingerprint."""

import pytest

from perfxpert.tools import trace_fingerprint
from perfxpert.tools._class import ToolClass


def test_fingerprint_captures_sys_trace():
    fp = trace_fingerprint.fingerprint("rocprofv3 --sys-trace -- ./app")
    assert "--sys-trace" in fp


def test_fingerprint_captures_pmc_counters():
    fp = trace_fingerprint.fingerprint(
        "rocprofv3 --pmc SQ_WAVES GRBM_COUNT -- ./app"
    )
    assert "pmc:SQ_WAVES" in fp
    assert "pmc:GRBM_COUNT" in fp


def test_fingerprint_ignores_ordering():
    f1 = trace_fingerprint.fingerprint("rocprofv3 --sys-trace --stats -- ./app")
    f2 = trace_fingerprint.fingerprint("rocprofv3 --stats --sys-trace -- ./app")
    assert f1 == f2


def test_fingerprint_ignores_output_flags():
    """-d, -o, --output, etc. don't affect what data is collected."""
    f1 = trace_fingerprint.fingerprint("rocprofv3 --sys-trace -d ./out1 -- ./app")
    f2 = trace_fingerprint.fingerprint("rocprofv3 --sys-trace -d ./out2 -- ./app")
    assert f1 == f2


def test_is_read_only_class():
    assert trace_fingerprint.fingerprint.__tool_class__ == ToolClass.READ_ONLY
