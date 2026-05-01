"""Tests for perfxpert.tools.profile_runner — EXECUTION class."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import profile_runner
from perfxpert.tools._class import ToolClass


def test_run_is_execution_class():
    assert profile_runner.run.__tool_class__ == ToolClass.EXECUTION


def test_run_rejects_unknown_flag_before_tool_check(tmp_path: Path, monkeypatch):
    require_tool = mock.MagicMock(side_effect=AssertionError("require_tool reached"))
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", require_tool)

    with pytest.raises(profile_runner.RocprofFlagError) as exc:
        profile_runner.run(
            argv=["rocprofv3", "--totally-fake-flag", "--", "./app"],
            cwd=tmp_path,
        )
    assert "--totally-fake-flag" in str(exc.value)
    require_tool.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["rocprofv3", "--output-dir", "/etc", "--", "./app"],
        ["rocprofv3", "--output-dir=/etc", "--", "./app"],
        ["rocprofv3", "-d", "../outside", "--", "./app"],
        ["rocprofv3", "-o", r"C:\Windows\Temp\out", "--", "./app"],
        ["rocprofv3", "--att-library-path", "~/libatt.so", "--", "./app"],
    ],
)
def test_validate_argv_rejects_unsafe_path_values(argv):
    with pytest.raises(profile_runner.RocprofFlagError):
        profile_runner._validate_argv(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["rocprofv3", "-d", "out", "-o", "results", "--", "./app"],
        ["rocprofv3", "--pc-sampling", "--", "./app"],
        ["rocprofv3", "--output-dir=out", "--pmc", "SQ_WAVES", "--", "./app"],
        ["rocprofv3", "--pmc", "SQ_WAVES", "GRBM_COUNT", "-d", "out", "--", "./app"],
        ["rocprofv3", "--att", "--att-library-path", "/opt/rocm/lib", "--", "./app"],
        [
            "rocprofv3",
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            "stochastic",
            "--pc-sampling-unit",
            "cycles",
            "--pc-sampling-interval",
            "1048576",
            "--",
            "./app",
        ],
        [
            "rocprofv3",
            "--pc-sampling-beta-enabled",
            "1",
            "--pc-sampling-method=host_trap",
            "--pc-sampling-unit=time",
            "--pc-sampling-interval=1",
            "--",
            "./app",
        ],
    ],
)
def test_validate_argv_accepts_safe_value_flags(argv):
    profile_runner._validate_argv(argv)


def test_validate_argv_rejects_symlink_output_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside-profile-output"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(profile_runner.RocprofFlagError):
        profile_runner._validate_argv(
            ["rocprofv3", "-d", "escape", "--", "./app"],
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["rocprofv3", "--output-dir", "--", "./app"],
        ["rocprofv3", "--sys-trace=true", "--", "./app"],
        ["rocprofv3", "--pmc", "--", "./app"],
        ["rocprofv3", "--pc-sampling", "stochastic", "--", "./app"],
        ["rocprofv3", "--pc-sampling-method", "timer", "--", "./app"],
        ["rocprofv3", "--pc-sampling-unit", "warps", "--", "./app"],
        ["rocprofv3", "--pc-sampling-interval", "fast", "--", "./app"],
        ["rocprofv3", "--pc-sampling-beta-enabled", "maybe", "--", "./app"],
        ["rocprofv3", "out", "--", "./app"],
    ],
)
def test_validate_argv_rejects_malformed_value_flags(argv):
    with pytest.raises(profile_runner.RocprofFlagError):
        profile_runner._validate_argv(argv)


def test_run_accepts_known_flags(tmp_path: Path, monkeypatch):
    fake = mock.MagicMock(return_value=mock.MagicMock(
        returncode=0, stdout=b"", stderr=b""
    ))
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", mock.MagicMock())
    monkeypatch.setattr("perfxpert.tools.profile_runner.subprocess.run", fake)

    result = profile_runner.run(
        argv=["rocprofv3", "--sys-trace", "--kernel-trace", "--", "./app"],
        cwd=tmp_path,
    )
    assert result["returncode"] == 0
    args, kwargs = fake.call_args
    assert isinstance(args[0], list)
    assert kwargs.get("shell", False) is False


def test_run_rejects_shell_injection_in_args_before_tool_check(tmp_path: Path, monkeypatch):
    from perfxpert.tools._safety import ShellMetacharError
    require_tool = mock.MagicMock(side_effect=AssertionError("require_tool reached"))
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", require_tool)

    with pytest.raises(ShellMetacharError):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", "--", "./app;rm -rf ~"],
            cwd=tmp_path,
        )
    require_tool.assert_not_called()


def test_run_strips_api_keys_from_env(tmp_path: Path, monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("perfxpert.tools.profile_runner.subprocess.run", fake_run)
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", mock.MagicMock())
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
    monkeypatch.setattr("perfxpert.tools.profile_runner.require_tool", mock.MagicMock())
    with pytest.raises(_sp.TimeoutExpired):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", "--", "./app"],
            cwd=tmp_path,
            timeout=10,
        )
