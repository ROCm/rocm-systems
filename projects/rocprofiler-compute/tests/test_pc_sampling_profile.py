# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from unittest.mock import Mock

from common import patch_console

from pc_sampling.pc_sampling_profile import (
    PC_SAMPLING_STATIC_INTERVAL_LIMITS,
    PCSamplingProfile,
    _merge_agent_interval_limits,
    _resolve_avail_library_path,
    pc_sampling_interval_limits,
    query_pc_sampling_configs,
)

MODULE = "pc_sampling.pc_sampling_profile"


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
# device interval limit query
# ---------------------------------------------------------------------------
def make_config(method, unit, min_interval, max_interval, flags=0):
    """Build one agent config as _query_agent_pc_sampling_configs returns it."""
    return {
        "method": method,
        "unit": unit,
        "min_interval": min_interval,
        "max_interval": max_interval,
        "flags": flags,
    }


def patch_agent_configs(monkeypatch, configs_by_agent):
    """Stub the per-agent ctypes queries with canned configurations."""
    monkeypatch.setattr(
        f"{MODULE}._query_agent_handles",
        lambda _library: list(configs_by_agent),
    )
    monkeypatch.setattr(
        f"{MODULE}._query_agent_pc_sampling_configs",
        lambda _library, agent_handle: configs_by_agent[agent_handle],
    )


def test_merge_agent_interval_limits_decodes_both_methods(monkeypatch):
    """Stochastic and host_trap configs are decoded with their pow2 flag."""
    patch_agent_configs(
        monkeypatch,
        {
            0: [
                make_config(1, 2, 256, 1048576, flags=1),
                make_config(2, 3, 1, 1048576),
            ]
        },
    )

    limits = _merge_agent_interval_limits(Mock())

    assert limits["stochastic"] == {
        "min_interval": 256,
        "max_interval": 1048576,
        "interval_pow2": True,
    }
    assert limits["host_trap"] == {
        "min_interval": 1,
        "max_interval": 1048576,
        "interval_pow2": False,
    }


def test_merge_agent_interval_limits_widens_range_across_agents(monkeypatch):
    """The range is the union across agents; pow2 sticks if any agent needs it."""
    patch_agent_configs(
        monkeypatch,
        {
            0: [make_config(1, 2, 512, 65536, flags=1)],
            1: [make_config(1, 2, 256, 1048576)],
        },
    )

    limits = _merge_agent_interval_limits(Mock())

    assert limits["stochastic"] == {
        "min_interval": 256,
        "max_interval": 1048576,
        "interval_pow2": True,
    }


def test_merge_agent_interval_limits_skips_mismatched_unit(monkeypatch):
    """A stochastic config reported in time units is not a cycles config."""
    patch_agent_configs(monkeypatch, {0: [make_config(1, 3, 256, 1048576)]})

    assert _merge_agent_interval_limits(Mock()) == {}


def test_query_returns_empty_when_library_missing(monkeypatch):
    """A missing avail library yields no limits rather than an error."""
    monkeypatch.setattr(f"{MODULE}._load_avail_library", lambda _path: None)

    assert query_pc_sampling_configs("/nonexistent/tool.so") == {}


def test_interval_limits_falls_back_to_static_limits(monkeypatch):
    """An unqueryable device falls back to the static limits."""
    monkeypatch.setattr(f"{MODULE}.query_pc_sampling_configs", lambda _path: {})

    limits = pc_sampling_interval_limits("stochastic")

    assert limits == PC_SAMPLING_STATIC_INTERVAL_LIMITS["stochastic"]
    assert limits["max_interval"] == 1048576


def test_interval_limits_prefers_queried_limits(monkeypatch):
    """A queried device wins over the static limits."""
    queried = {"min_interval": 256, "max_interval": 4096, "interval_pow2": True}
    monkeypatch.setattr(
        f"{MODULE}.query_pc_sampling_configs",
        lambda _path: {"stochastic": queried},
    )

    assert pc_sampling_interval_limits("stochastic") == queried


def test_avail_library_resolved_next_to_sdk_tool(monkeypatch):
    """The avail library is looked up beside the configured SDK tool."""
    monkeypatch.setattr(f"{MODULE}.resolve_rocm_library_path", lambda path: path)

    resolved = _resolve_avail_library_path("/opt/rocm/lib/rocprofiler-sdk/tool.so")

    assert resolved == ("/opt/rocm/lib/rocprofiler-sdk/librocprofv3-list-avail.so")


def test_avail_library_falls_back_to_rocm_path(monkeypatch):
    """Without an SDK tool path, ROCM_PATH locates the avail library."""
    monkeypatch.setattr(f"{MODULE}.resolve_rocm_library_path", lambda path: path)
    monkeypatch.setenv("ROCM_PATH", "/custom/rocm")

    resolved = _resolve_avail_library_path(None)

    assert resolved == "/custom/rocm/lib/rocprofiler-sdk/librocprofv3-list-avail.so"
