"""Tests for perfxpert.tools.anchors — EXECUTION class."""

import os
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import anchors
from perfxpert.tools._class import ToolClass
from perfxpert.tools._safety import PathConfinementError


def test_check_is_execution_class():
    assert anchors.check.__tool_class__ == ToolClass.EXECUTION


def test_check_all_pass(tmp_path: Path, monkeypatch):
    # Fake test runner that returns 0
    monkeypatch.setattr(
        "perfxpert.tools.anchors.subprocess.run",
        mock.MagicMock(return_value=mock.MagicMock(
            returncode=0,
            stdout=b"4 passed in 0.02s\n",
            stderr=b"",
        )),
    )
    r = anchors.check(
        project_root=tmp_path,
        test_command=["pytest", "tests/"],
    )
    assert r["all_passed"] is True
    assert r["returncode"] == 0


def test_check_some_fail(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "perfxpert.tools.anchors.subprocess.run",
        mock.MagicMock(return_value=mock.MagicMock(
            returncode=1,
            stdout=b"3 passed, 1 failed\n",
            stderr=b"",
        )),
    )
    r = anchors.check(project_root=tmp_path, test_command=["pytest", "tests/"])
    assert r["all_passed"] is False


def test_check_rejects_shell_metachars(tmp_path: Path):
    from perfxpert.tools._safety import ShellMetacharError
    with pytest.raises(ShellMetacharError):
        anchors.check(
            project_root=tmp_path,
            test_command=["pytest;rm -rf ~"],
        )


def test_check_rejects_non_allowlisted_executable(tmp_path: Path):
    with pytest.raises(anchors.AnchorCommandError, match="not in allowlist"):
        anchors.check(project_root=tmp_path, test_command=["bash", "-lc", "pytest"])


def test_check_rejects_dangerous_runner_flag(tmp_path: Path):
    with pytest.raises(anchors.AnchorCommandError, match="dangerous flag"):
        anchors.check(project_root=tmp_path, test_command=["pytest", "-p", "evil_plugin"])


def test_check_rejects_attached_dangerous_runner_flag(tmp_path: Path):
    with pytest.raises(anchors.AnchorCommandError, match="dangerous flag"):
        anchors.check(project_root=tmp_path, test_command=["pytest", "-pevil_plugin"])


def test_check_rejects_test_paths_outside_project(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        anchors.check(project_root=tmp_path, test_command=["pytest", "/tmp/evil"])


def test_check_rejects_inline_path_flag_outside_project(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        anchors.check(
            project_root=tmp_path,
            test_command=["pytest", "--rootdir=/tmp/evil"],
        )


def test_check_rejects_attached_short_path_flag_outside_project(tmp_path: Path):
    with pytest.raises(PathConfinementError):
        anchors.check(
            project_root=tmp_path,
            test_command=["pytest", "-c/tmp/evil"],
        )


def test_check_uses_safe_env(tmp_path: Path, monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("perfxpert.tools.anchors.subprocess.run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    anchors.check(project_root=tmp_path, test_command=["pytest"])
    assert "ANTHROPIC_API_KEY" not in captured_env
    assert "/usr/bin" in captured_env["PATH"]
    assert "/tmp/evil" not in captured_env["PATH"]


def test_check_resolves_runner_from_trusted_path(tmp_path: Path, monkeypatch):
    shadow = tmp_path / "pytest"
    shadow.write_text("#!/bin/sh\nexit 0\n")
    shadow.chmod(0o755)
    captured = {}

    def fake_which(exe, path=None):
        captured["which_path"] = path
        if path == anchors._TRUSTED_EXECUTABLE_PATH:
            return "/usr/bin/pytest"
        return str(shadow)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr("perfxpert.tools.anchors.shutil.which", fake_which)
    monkeypatch.setattr("perfxpert.tools.anchors.subprocess.run", fake_run)

    anchors.check(project_root=tmp_path, test_command=["pytest"])

    assert captured["which_path"] == anchors._TRUSTED_EXECUTABLE_PATH
    assert captured["cmd"][0] == "/usr/bin/pytest"
