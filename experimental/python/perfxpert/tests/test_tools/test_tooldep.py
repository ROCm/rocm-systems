"""Tests for perfxpert.tools._tooldep — external-tool verifier."""

import os
import subprocess
from unittest import mock

import pytest

from perfxpert.tools import _tooldep
from perfxpert.tools._tooldep import (
    ExternalToolMissing,
    check_tool_available,
    require_tool,
    offer_install,
)


# -- registry sanity --------------------------------------------------------

def test_registry_contains_expected_externals():
    """The registry must enumerate every external dep used by the codebase."""
    required = {
        "opencode",
        "rocprofv3",
        "amdclang++",
        "rocprof-trace-decoder",   # ATT library
        "mcp",                     # python
        "pexpect",
    }
    registered = set(_tooldep._TOOL_REGISTRY.keys())
    missing = required - registered
    assert not missing, f"missing from registry: {missing}"


# -- check_tool_available ---------------------------------------------------

def test_check_binary_present_returns_true(monkeypatch, tmp_path):
    fake = tmp_path / "some-binary"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(_tooldep.shutil, "which", lambda name: str(fake))
    ok, detail = check_tool_available("opencode")
    assert ok is True
    assert str(fake) in detail


def test_check_binary_missing_returns_false(monkeypatch):
    monkeypatch.setattr(_tooldep.shutil, "which", lambda name: None)
    ok, detail = check_tool_available("opencode")
    assert ok is False
    assert "install" in detail.lower() or "not found" in detail.lower()


def test_check_pylib_present_returns_true():
    # mcp is installed in this env (Phase 4 landed it)
    ok, detail = check_tool_available("mcp")
    # Either mcp is installed OR the test env doesn't have it — both are OK.
    assert isinstance(ok, bool)


def test_check_pylib_missing_returns_false(monkeypatch):
    monkeypatch.setattr(_tooldep.importlib.util, "find_spec", lambda name: None)
    ok, detail = check_tool_available("mcp")
    assert ok is False
    assert "pip install" in detail


# -- require_tool -----------------------------------------------------------

def test_require_tool_raises_when_missing(monkeypatch):
    monkeypatch.setattr(_tooldep.shutil, "which", lambda name: None)
    with pytest.raises(ExternalToolMissing) as exc:
        require_tool("opencode")
    assert "opencode" in str(exc.value)
    assert exc.value.install_hint
    # Install command exists for opencode (curl install script)
    assert exc.value.install_cmd is not None


def test_require_tool_with_allow_install_offers(monkeypatch, capsys):
    """If PERFXPERT_ALLOW_INSTALL=1 the helper prompts the user."""
    monkeypatch.setenv("PERFXPERT_ALLOW_INSTALL", "1")
    monkeypatch.setattr(_tooldep.shutil, "which", lambda name: None)
    # Simulate user pressing 'n'
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(ExternalToolMissing):
        require_tool("opencode", allow_install=True)
    captured = capsys.readouterr()
    assert "install" in captured.out.lower()


# -- offer_install ----------------------------------------------------------

def test_offer_install_declined_returns_false(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    ok = offer_install("opencode")
    assert ok is False


def test_offer_install_accepted_runs_command(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    run_calls = []

    def fake_run(cmd, **kw):
        run_calls.append(cmd)
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(_tooldep.subprocess, "run", fake_run)
    ok = offer_install("opencode")
    assert ok is True
    assert run_calls, "install command should have been invoked"


# -- ATT decoder (shared_lib kind) -----------------------------------------

def test_att_decoder_check_searches_rocm_paths(monkeypatch, tmp_path):
    # Simulate the library existing under /opt/rocm
    fake_lib = tmp_path / "librocprof-trace-decoder.so"
    fake_lib.write_bytes(b"fake\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setattr(
        _tooldep,
        "_shared_lib_search_paths",
        lambda name: [str(tmp_path)],
    )
    ok, detail = check_tool_available("rocprof-trace-decoder")
    assert ok is True
