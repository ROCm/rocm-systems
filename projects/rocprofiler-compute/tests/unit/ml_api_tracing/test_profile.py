# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path

import pytest

from ml_api_tracing import profile
from utils.utils_exceptions import PythonScriptNotFoundError


def launcher_path():
    return Path(profile.__file__).parent / "recording" / "launch.py"


def test_profile_import_preserves_profile_dependency_boundaries(assert_profile_import):
    assert_profile_import("ml_api_tracing.profile", Path(profile.__file__).parents[1])


@pytest.mark.parametrize(
    "flags, expected",
    [
        ({}, set()),
        ({"torch_trace": True}, {"torch"}),
        ({"triton_trace": True}, {"triton"}),
        ({"ml_api_trace": True}, {"torch", "triton"}),
        ({"torch_trace": True, "triton_trace": True}, {"torch", "triton"}),
        ({"torch_trace": True, "ml_api_trace": True}, {"torch", "triton"}),
    ],
)
def test_compute_selected_frameworks(flags, expected):
    assert profile.compute_selected_frameworks(argparse.Namespace(**flags)) == expected


def test_injection_preserves_interpreter_flags_and_workload_arguments(tmp_path):
    script = tmp_path / "workload with spaces.py"
    script.write_text("pass\n", encoding="utf-8")
    interpreter = [sys.executable, "-u", "-W", "ignore", "-X", "dev"]
    workload = [str(script), "--name", "two words", "--"]
    command = interpreter + workload
    frameworks = {"triton", "torch"}

    profile.prepare_ml_api_trace_injection(
        command,
        Path(sys.executable),
        is_python=True,
        script_index=len(interpreter),
        skip_flag=None,
        frameworks=frameworks,
    )

    assert command == interpreter + [
        str(launcher_path()),
        "--frameworks",
        "torch",
        "triton",
        "--",
        *workload,
    ]
    assert frameworks == {"torch", "triton"}


@pytest.mark.parametrize("suffix", [".py", ".pyw", ".pyc", ".pyo"])
def test_injection_prepends_current_interpreter_to_direct_script(tmp_path, suffix):
    script = tmp_path / f"workload{suffix}"
    command = [str(script), "argument with spaces"]

    profile.prepare_ml_api_trace_injection(
        command,
        script,
        is_python=False,
        script_index=None,
        skip_flag=None,
        frameworks={"triton"},
    )

    assert command == [
        sys.executable,
        str(launcher_path()),
        "--frameworks",
        "triton",
        "--",
        str(script),
        "argument with spaces",
    ]


@pytest.mark.parametrize("skip_flag", ["-c", "-m"])
def test_injection_warns_and_preserves_inline_python_command(skip_flag, caplog):
    command = [sys.executable, "-u", skip_flag, "workload", "--flag"]
    original = command.copy()

    profile.prepare_ml_api_trace_injection(
        command,
        Path(sys.executable),
        is_python=True,
        script_index=None,
        skip_flag=skip_flag,
        frameworks={"torch"},
    )

    assert command == original
    assert len(caplog.messages) == 1
    assert (
        f"Cannot inject ROCTX markers into 'python {skip_flag}'" in caplog.messages[0]
    )


def test_injection_missing_script_preserves_command(tmp_path):
    script = str(tmp_path / "missing.py")
    command = [sys.executable, "-u", script]
    original = command.copy()

    with pytest.raises(PythonScriptNotFoundError) as error:
        profile.prepare_ml_api_trace_injection(
            command,
            Path(sys.executable),
            is_python=True,
            script_index=2,
            skip_flag=None,
            frameworks={"torch"},
        )

    assert error.value.script == script
    assert command == original


@pytest.mark.parametrize("skip_flag", [None, "-c"])
def test_missing_launcher_errors_before_script_validation_or_skip_warning(
    tmp_path, monkeypatch, caplog, skip_flag
):
    monkeypatch.setattr(profile, "__file__", str(tmp_path / "owner" / "profile.py"))
    command = [sys.executable, skip_flag or "missing.py"]
    original = command.copy()

    with pytest.raises(SystemExit) as error:
        profile.prepare_ml_api_trace_injection(
            command,
            Path(sys.executable),
            is_python=True,
            script_index=1,
            skip_flag=skip_flag,
            frameworks={"torch"},
        )

    assert error.value.code == 1
    assert command == original
    assert caplog.messages == [
        f"Cannot find inject_roctx launcher at {launcher_path()}. "
        "Please verify your installation."
    ]


@pytest.mark.parametrize("is_bundle", [False, True])
def test_binary_warns_in_order_without_command_mutation(tmp_path, caplog, is_bundle):
    binary = tmp_path / "application"
    if is_bundle:
        (tmp_path / "_internal").mkdir()
    command = [str(binary), "--flag"]

    profile.prepare_ml_api_trace_injection(
        command,
        binary,
        is_python=False,
        script_index=None,
        skip_flag=None,
        frameworks={"torch"},
    )

    assert command == [str(binary), "--flag"]
    assert len(caplog.messages) == (2 if is_bundle else 1)
    assert caplog.messages[0].startswith(
        "Command does not look like a Python entry point"
    )
    if is_bundle:
        assert caplog.messages[1].startswith(
            "Workload appears to be a self-contained binary"
        )
