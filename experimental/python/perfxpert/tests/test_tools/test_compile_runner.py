"""Tests for perfxpert.tools.compile_runner — EXECUTION class."""

import os
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import compile_runner
from perfxpert.tools._class import ToolClass


def test_build_is_execution_class():
    assert compile_runner.build.__tool_class__ == ToolClass.EXECUTION


# -- flag allowlist ---------------------------------------------------------

def test_build_rejects_unlisted_flag_before_tool_check(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.cpp"
    src.write_text("int main(){return 0;}\n")
    require_tool = mock.MagicMock(side_effect=AssertionError("require_tool reached"))
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", require_tool)

    with pytest.raises(compile_runner.CompileFlagError) as exc:
        compile_runner.build(
            project_root=tmp_path,
            source_rel="src.cpp",
            flags=["-O2", "-Xlinker", "--wrap,write"],
        )
    assert "--wrap,write" in str(exc.value) or "-Xlinker" in str(exc.value)
    require_tool.assert_not_called()


def test_build_accepts_allowlisted_flags(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.cpp"
    src.write_text("int main(){return 0;}\n")

    # Mock subprocess so we don't actually compile
    fake_run = mock.MagicMock(return_value=mock.MagicMock(
        returncode=0, stdout=b"", stderr=b""
    ))
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", mock.MagicMock())
    monkeypatch.setattr("perfxpert.tools.compile_runner.subprocess.run", fake_run)

    result = compile_runner.build(
        project_root=tmp_path,
        source_rel="src.cpp",
        flags=["-O2", "--offload-arch=gfx942"],
    )

    assert result["returncode"] == 0
    # Confirm we passed the *list* not a shell string
    args, kwargs = fake_run.call_args
    assert isinstance(args[0], list)
    assert kwargs.get("shell") is False or "shell" not in kwargs


def test_build_accepts_configured_cxx_path(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.cpp"
    src.write_text("int main(){return 0;}\n")
    fake_cxx = tmp_path / "toolchain" / "amdclang++"
    fake_cxx.parent.mkdir()
    fake_cxx.write_text("#!/bin/sh\n")
    fake_cxx.chmod(0o755)
    fake_run = mock.MagicMock(return_value=mock.MagicMock(
        returncode=0, stdout=b"", stderr=b""
    ))
    monkeypatch.setattr(compile_runner, "_DEFAULT_CXX", str(fake_cxx))
    monkeypatch.setattr("perfxpert.tools.compile_runner.subprocess.run", fake_run)

    result = compile_runner.build(project_root=tmp_path, source_rel="src.cpp", flags=["-O2"])

    assert result["returncode"] == 0
    args, _ = fake_run.call_args
    assert args[0][0] == str(fake_cxx)


def test_build_rejects_path_traversal_in_source_before_tool_check(tmp_path: Path, monkeypatch):
    from perfxpert.tools._safety import PathConfinementError
    require_tool = mock.MagicMock(side_effect=AssertionError("require_tool reached"))
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", require_tool)

    with pytest.raises(PathConfinementError):
        compile_runner.build(
            project_root=tmp_path,
            source_rel="../escape.cpp",
            flags=[],
        )
    require_tool.assert_not_called()


def test_build_uses_safe_env(tmp_path: Path, monkeypatch):
    """API keys must NOT be forwarded to the compiler subprocess."""
    src = tmp_path / "src.cpp"
    src.write_text("int main(){return 0;}\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")

    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("perfxpert.tools.compile_runner.subprocess.run", fake_run)
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", mock.MagicMock())
    compile_runner.build(project_root=tmp_path, source_rel="src.cpp", flags=["-O2"])

    assert "ANTHROPIC_API_KEY" not in captured_env


def test_build_parses_error_output(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.cpp"
    src.write_text("bad syntax\n")
    monkeypatch.setattr(
        "perfxpert.tools.compile_runner.subprocess.run",
        mock.MagicMock(return_value=mock.MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"src.cpp:1:1: error: expected unqualified-id\n",
        )),
    )
    monkeypatch.setattr("perfxpert.tools.compile_runner.require_tool", mock.MagicMock())
    result = compile_runner.build(project_root=tmp_path, source_rel="src.cpp", flags=["-O2"])
    assert result["returncode"] == 1
    assert "error: expected unqualified-id" in result["stderr"]
