"""Tests for perfxpert.tools.compiler."""

import pytest

from perfxpert.tools import compiler
from perfxpert.tools._class import ToolClass


def test_lookup_returns_only_allowlisted():
    flags = compiler.lookup_flags()
    for f in flags:
        assert f["allowlist"] is True


def test_lookup_excludes_denylisted():
    flags = compiler.lookup_flags()
    names = {f["flag"] for f in flags}
    assert "-Xlinker" not in names   # denylisted for security


def test_explain_known_flag():
    info = compiler.explain_flag("-O3")
    assert info["category"] == "optimization_level"


def test_explain_unknown_raises():
    with pytest.raises(KeyError):
        compiler.explain_flag("--definitely-not-a-flag")


def test_is_read_only_class():
    assert compiler.lookup_flags.__tool_class__ == ToolClass.READ_ONLY
    assert compiler.explain_flag.__tool_class__ == ToolClass.READ_ONLY
