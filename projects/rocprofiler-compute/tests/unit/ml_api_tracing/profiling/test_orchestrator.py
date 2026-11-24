# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ML API tracing profile options and workload injection."""

import sys
from pathlib import Path

import pytest

from ml_api_tracing.profiling import orchestrator
from utils.utils_exceptions import PythonScriptNotFoundError


class MessageRecorder:
    """Stand in for a console function: keep the messages, exit on request."""

    def __init__(self, exit_code=None):
        self.messages = []
        self.exit_code = exit_code

    def __call__(self, *argv):
        self.messages.append(str(argv[-1]))
        if self.exit_code is not None:
            raise SystemExit(self.exit_code)


def launcher_path():
    return Path(orchestrator.__file__).parent.parent / "inject_roctx" / "launch.py"


def test_injection_preserves_interpreter_flags_and_workload_arguments(tmp_path):
    script = tmp_path / "workload with spaces.py"
    script.write_text("pass\n", encoding="utf-8")
    interpreter = [sys.executable, "-u", "-W", "ignore", "-X", "dev"]
    workload = [str(script), "--name", "two words", "--"]
    command = interpreter + workload
    frameworks = ["triton", "torch"]

    orchestrator.prepare_ml_api_trace_injection(
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


@pytest.mark.parametrize("suffix", [".py", ".pyw", ".pyc", ".pyo"])
def test_injection_prepends_current_interpreter_to_direct_script(tmp_path, suffix):
    script = tmp_path / f"workload{suffix}"
    command = [str(script), "argument with spaces"]

    orchestrator.prepare_ml_api_trace_injection(
        command,
        script,
        is_python=False,
        script_index=None,
        skip_flag=None,
        frameworks=["triton"],
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
def test_injection_warns_and_preserves_inline_python_command(skip_flag, monkeypatch):
    warnings = MessageRecorder()
    monkeypatch.setattr(orchestrator, "console_warning", warnings)
    command = [sys.executable, "-u", skip_flag, "workload", "--flag"]
    original = command.copy()

    orchestrator.prepare_ml_api_trace_injection(
        command,
        Path(sys.executable),
        is_python=True,
        script_index=None,
        skip_flag=skip_flag,
        frameworks=["torch"],
    )

    assert command == original
    assert len(warnings.messages) == 1
    assert (
        f"Cannot inject ROCTX markers into 'python {skip_flag}'" in warnings.messages[0]
    )


def test_injection_missing_script_preserves_command(tmp_path):
    script = str(tmp_path / "missing.py")
    command = [sys.executable, "-u", script]
    original = command.copy()

    with pytest.raises(PythonScriptNotFoundError) as error:
        orchestrator.prepare_ml_api_trace_injection(
            command,
            Path(sys.executable),
            is_python=True,
            script_index=2,
            skip_flag=None,
            frameworks=["torch"],
        )

    assert error.value.script == script
    assert command == original


@pytest.mark.parametrize("is_bundle", [False, True])
def test_binary_warns_in_order_without_command_mutation(
    tmp_path, monkeypatch, is_bundle
):
    warnings = MessageRecorder()
    monkeypatch.setattr(orchestrator, "console_warning", warnings)
    binary = tmp_path / "application"
    if is_bundle:
        (tmp_path / "_internal").mkdir()
    command = [str(binary), "--flag"]

    orchestrator.prepare_ml_api_trace_injection(
        command,
        binary,
        is_python=False,
        script_index=None,
        skip_flag=None,
        frameworks=["torch"],
    )

    assert command == [str(binary), "--flag"]
    assert len(warnings.messages) == (2 if is_bundle else 1)
    assert warnings.messages[0].startswith(
        "Command does not look like a Python entry point"
    )
    if is_bundle:
        assert warnings.messages[1].startswith(
            "Workload appears to be a self-contained binary"
        )
