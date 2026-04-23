"""Tests for perfxpert.tools.profile_runner — EXECUTION class."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import profile_runner
from perfxpert.tools._class import ToolClass


def test_run_is_execution_class():
    assert profile_runner.run.__tool_class__ == ToolClass.EXECUTION


def test_run_rejects_unknown_flag(tmp_path: Path):
    with pytest.raises(profile_runner.RocprofFlagError) as exc:
        profile_runner.run(
            argv=["rocprofv3", "--totally-fake-flag", "--", "./app"],
            cwd=tmp_path,
        )
    assert "--totally-fake-flag" in str(exc.value)


def test_run_accepts_known_flags(tmp_path: Path, monkeypatch):
    fake = mock.MagicMock(return_value=mock.MagicMock(
        returncode=0, stdout=b"", stderr=b""
    ))
    monkeypatch.setattr("perfxpert.tools.profile_runner.subprocess.run", fake)

    result = profile_runner.run(
        argv=["rocprofv3", "--sys-trace", "--kernel-trace", "--", "./app"],
        cwd=tmp_path,
    )
    assert result["returncode"] == 0
    args, kwargs = fake.call_args
    assert isinstance(args[0], list)
    assert kwargs.get("shell", False) is False


def test_run_rejects_shell_injection_in_args(tmp_path: Path):
    from perfxpert.tools._safety import ShellMetacharError
    with pytest.raises(ShellMetacharError):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", "--", "./app;rm -rf ~"],
            cwd=tmp_path,
        )


def test_run_strips_api_keys_from_env(tmp_path: Path, monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("perfxpert.tools.profile_runner.subprocess.run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")

    profile_runner.run(argv=["rocprofv3", "--sys-trace", "--", "./app"], cwd=tmp_path)

    assert "ANTHROPIC_API_KEY" not in captured_env
    # rocprofv3 envs SHOULD pass through
    monkeypatch.setenv("ROCPROFV3_LOG_LEVEL", "debug")
    profile_runner.run(argv=["rocprofv3", "--sys-trace", "--", "./app"], cwd=tmp_path)
    assert captured_env.get("ROCPROFV3_LOG_LEVEL") == "debug"


def test_run_respects_timeout(tmp_path: Path, monkeypatch):
    import subprocess as _sp

    def raise_timeout(*a, **kw):
        raise _sp.TimeoutExpired(cmd=a[0], timeout=kw.get("timeout", 10))

    monkeypatch.setattr("perfxpert.tools.profile_runner.subprocess.run", raise_timeout)
    with pytest.raises(_sp.TimeoutExpired):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", "--", "./app"],
            cwd=tmp_path,
            timeout=10,
        )
