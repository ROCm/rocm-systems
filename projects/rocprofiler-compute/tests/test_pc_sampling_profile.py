# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from unittest.mock import Mock

import pytest
from common import patch_console

from pc_sampling.pc_sampling_profile import (
    PCSamplingProfile,
    pc_sampling_interval_limits,
)

MODULE = "pc_sampling.pc_sampling_profile"

# Limits pc_sampling_interval_limits() reports when no agent can be queried.
STOCHASTIC_FALLBACK = {
    "min_interval": 1,
    "max_interval": 1048576,
    "interval_pow2": True,
}


class MockArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_pc_sampling_profile(profiler="rocprofiler-sdk", filter_blocks=("21",)):
    """Build a PCSamplingProfile runner; options are supplied per launch."""
    return PCSamplingProfile(
        args=MockArgs(filter_blocks=list(filter_blocks)),
        profiler=profiler,
    )


def _patch_avail_library(monkeypatch, configs_by_agent):
    """Stub the avail library with canned per-agent configurations.

    Each config is (method, unit, min_interval, max_interval, flags) as
    pc_sample_config() reports it.
    """
    library = Mock()
    library.get_number_of_agents.return_value = len(configs_by_agent)

    def fill_handles(handles, _count):
        for index, handle in enumerate(configs_by_agent):
            handles[index] = handle

    library.agent_handles.side_effect = fill_handles
    monkeypatch.setattr(f"{MODULE}._load_avail_library", lambda _path: library)
    monkeypatch.setattr(
        f"{MODULE}._query_agent_configs",
        lambda _library, handle: configs_by_agent[handle],
    )


# ---------------------------------------------------------------------------
# sdk backend launch
# ---------------------------------------------------------------------------
def test_sdk_forwards_options_env_and_runs_app_cmd(monkeypatch):
    """sdk launch overlays the options env, pops APP_CMD, and runs it."""
    options = {
        "APP_CMD": "my_app --arg",
        "LD_PRELOAD": "/sdk/tool.so",
        "ROCPROF_PC_SAMPLING_METHOD": "host_trap",
    }

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    _make_pc_sampling_profile()._launch(options)

    assert mock_capture.call_args.args[0] == "my_app --arg"
    called_env = mock_capture.call_args.kwargs.get("new_env", {})
    assert called_env["LD_PRELOAD"] == "/sdk/tool.so"
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == "host_trap"
    mock_error.assert_not_called()


def test_sdk_missing_app_cmd_errors(monkeypatch):
    """sdk non-live-attach without APP_CMD errors before launching."""
    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    _make_pc_sampling_profile()._launch({"LD_PRELOAD": "x"})

    assert mock_error.called
    assert "APP_CMD" in mock_error.call_args.args[0]
    mock_capture.assert_not_called()


def test_sdk_subprocess_failure_errors(monkeypatch):
    """A failed subprocess reports the standard PC sampling failure."""
    mock_capture = Mock(return_value=(False, "Error output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    _make_pc_sampling_profile()._launch({"APP_CMD": "my_app"})

    assert mock_error.called
    assert "PC sampling failed." in mock_error.call_args.args[0]


def test_sdk_env_log_excludes_user_env(monkeypatch):
    """The debug env log records the overlaid vars but never the user's env."""
    monkeypatch.setenv("LEAK_CANARY_TOKEN", "SHOULD_NOT_APPEAR")
    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    _mocks = patch_console(monkeypatch, MODULE, "debug", "error")
    mock_debug, mock_error = _mocks["debug"], _mocks["error"]

    _make_pc_sampling_profile()._launch({
        "APP_CMD": "my_app",
        "ROCPROF_PC_SAMPLING_METHOD": "host_trap",
    })

    logs = [str(call.args[0]) for call in mock_debug.call_args_list]
    env_log_lines = [m for m in logs if "env vars" in m]
    assert env_log_lines
    assert any("ROCPROF_PC_SAMPLING_METHOD" in m for m in env_log_lines)
    assert not any("SHOULD_NOT_APPEAR" in m for m in logs)
    mock_error.assert_not_called()


def test_sdk_live_attach_performs_attach_detach(monkeypatch):
    """sdk live-attach calls perform_attach_detach and returns before launching."""
    options = {
        "ROCPROF_ATTACH_PID": "1234",
        "ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC": "1",
    }

    mock_capture = Mock()
    mock_attach = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.perform_attach_detach", mock_attach)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    _make_pc_sampling_profile()._launch(options)

    mock_attach.assert_called_once()
    new_env, attach_options = mock_attach.call_args.args
    assert new_env["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    assert attach_options["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    mock_capture.assert_not_called()
    mock_error.assert_not_called()


# ---------------------------------------------------------------------------
# v3 backend launch
# ---------------------------------------------------------------------------
def test_v3_runs_rocprof_command(monkeypatch):
    """v3 launch runs the rocprof CLI with the supplied flag list."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_tool")
    options = ["--kernel-trace", "--", "./myapp", "arg1"]

    mock_capture = Mock(return_value=(True, "Success"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    _make_pc_sampling_profile(profiler="rocprofv3")._launch(options)

    assert mock_capture.call_args.args[0] == [
        "rocprof_cli_tool",
        "--kernel-trace",
        "--",
        "./myapp",
        "arg1",
    ]
    mock_error.assert_not_called()


def test_v3_subprocess_failure_errors(monkeypatch):
    """A failed v3 subprocess reports the standard PC sampling failure."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_tool")
    mock_capture = Mock(return_value=(False, "Error"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(profiler="rocprofv3")
    profiler._launch(["--kernel-trace", "--", "x"])

    assert mock_error.called
    assert "PC sampling failed." in mock_error.call_args.args[0]


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def test_is_requested():
    for blocks in (["21"], ["pc_sampling"], ["2", "21"]):
        assert _make_pc_sampling_profile(filter_blocks=blocks).is_requested() is True
    assert _make_pc_sampling_profile(filter_blocks=["2"]).is_requested() is False


def test_run_launches_and_logs(monkeypatch):
    """run() launches the subprocess and emits the run header and a timing debug."""
    mock_capture = Mock(return_value=(True, ""))
    mock_log = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.console_log", mock_log)
    _mocks = patch_console(monkeypatch, MODULE, "debug", "error")
    mock_debug, mock_error = _mocks["debug"], _mocks["error"]

    _make_pc_sampling_profile().run({"APP_CMD": "my_app"}, prior_run_count=0)

    assert mock_capture.called
    mock_error.assert_not_called()
    mock_log.assert_any_call("[Run 1/1][PC sampling profile run]")
    assert any(
        call.args and call.args[0] == "profiling" for call in mock_debug.call_args_list
    )


# ---------------------------------------------------------------------------
# device interval limits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "configs_by_agent, method, expected",
    [
        pytest.param(
            {0: [(1, 2, 256, 1048576, 1)]},
            "stochastic",
            {"min_interval": 256, "max_interval": 1048576, "interval_pow2": True},
            id="stochastic_cycles_decoded",
        ),
        pytest.param(
            {0: [(2, 3, 1, 1048576, 0)]},
            "host_trap",
            {"min_interval": 1, "max_interval": 1048576, "interval_pow2": False},
            id="host_trap_time_decoded",
        ),
        pytest.param(
            {0: [(1, 2, 512, 65536, 1)], 1: [(1, 2, 256, 1048576, 0)]},
            "stochastic",
            {"min_interval": 256, "max_interval": 1048576, "interval_pow2": True},
            id="range_unioned_across_agents",
        ),
        pytest.param(
            {0: [(1, 3, 256, 4096, 0)]},
            "stochastic",
            STOCHASTIC_FALLBACK,
            id="mismatched_unit_ignored",
        ),
    ],
)
def test_interval_limits_from_agent_configs(
    monkeypatch, configs_by_agent, method, expected
):
    """Agent configurations are decoded and merged into per-method limits."""
    _patch_avail_library(monkeypatch, configs_by_agent)

    assert pc_sampling_interval_limits(method) == expected


def test_interval_limits_fall_back_without_avail_library(monkeypatch):
    """An unloadable avail library yields the SDK fallback limits."""
    monkeypatch.setattr(f"{MODULE}._load_avail_library", lambda _path: None)

    assert pc_sampling_interval_limits("stochastic") == STOCHASTIC_FALLBACK
