"""Tests for perfxpert.tools.anchors — EXECUTION class."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import anchors
from perfxpert.tools._class import ToolClass


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


def test_check_uses_safe_env(tmp_path: Path, monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("perfxpert.tools.anchors.subprocess.run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    anchors.check(project_root=tmp_path, test_command=["pytest"])
    assert "ANTHROPIC_API_KEY" not in captured_env
