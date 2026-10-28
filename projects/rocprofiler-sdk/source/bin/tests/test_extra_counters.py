# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free tests for extra-counter file path forwarding."""

import errno
import os
import pickle

import pytest


@pytest.fixture
def launch(rocprofv3, monkeypatch):
    environments = []
    monkeypatch.delenv("ROCPROF_EXTRA_COUNTERS_FILE", raising=False)
    monkeypatch.delenv("ROCPROF_EXTRA_COUNTERS_CONTENTS", raising=False)
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)
    monkeypatch.setattr(
        rocprofv3.os,
        "execvpe",
        lambda executable, argv, env: environments.append(env),
    )
    monkeypatch.setattr(
        rocprofv3.subprocess,
        "check_call",
        lambda argv, env: environments.append(env) or 0,
    )

    def run(*args):
        previous_count = len(environments)
        rocprofv3.main(["--kernel-trace", *args, "--", "/bin/true"])
        assert len(environments) == previous_count + 1
        return environments[-1]

    return run


def test_extra_counter_file_is_forwarded_as_absolute_path(launch, monkeypatch, tmp_path):
    definitions_dir = tmp_path / "definitions with spaces"
    definitions_dir.mkdir()
    definitions_file = definitions_dir / "extra counters.yaml"
    definitions_file.write_text("rocprofiler-sdk:\n  counters: []\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROCPROF_EXTRA_COUNTERS_CONTENTS", "legacy inline contents")
    relative_path = os.path.relpath(definitions_file, tmp_path)

    env = launch("-E", relative_path)

    assert env["ROCPROF_EXTRA_COUNTERS_FILE"] == os.path.abspath(relative_path)
    assert "ROCPROF_EXTRA_COUNTERS_CONTENTS" not in env


def test_large_extra_counter_file_is_not_copied_into_environment(launch, tmp_path):
    definitions_file = tmp_path / "large.yaml"
    definitions_file.write_text("#" + ("x" * (256 * 1024)))

    env = launch("-E", str(definitions_file))

    assert env["ROCPROF_EXTRA_COUNTERS_FILE"] == os.path.abspath(definitions_file)
    assert "ROCPROF_EXTRA_COUNTERS_CONTENTS" not in env


def test_attach_with_application_preserves_args_and_large_path_only(
    rocprofv3, monkeypatch, tmp_path
):
    definitions_file = tmp_path / "large.yaml"
    definitions_file.write_text("#" + ("x" * (256 * 1024)))
    pid = 987654321
    cache_file = tmp_path / "attach.pkl"
    executions = []

    monkeypatch.delenv("ROCPROF_EXTRA_COUNTERS_FILE", raising=False)
    monkeypatch.delenv("ROCPROF_EXTRA_COUNTERS_CONTENTS", raising=False)
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, args: path)

    real_open = os.open

    def redirected_open(path, flags, mode=0o777):
        if path == f"/tmp/rocprofv3_attach_{pid}.pkl":
            path = cache_file
        return real_open(path, flags, mode)

    def capture_exec(executable, argv, env):
        executions.append(("exec", argv, dict(env)))

    def capture_subprocess(argv, env):
        executions.append(("subprocess", argv, dict(env)))
        return 0

    monkeypatch.setattr(rocprofv3.os, "open", redirected_open)
    monkeypatch.setattr(rocprofv3.os, "execvpe", capture_exec)
    monkeypatch.setattr(rocprofv3.subprocess, "check_call", capture_subprocess)
    monkeypatch.setattr(pickle, "dump", lambda args, output: None)

    # Preserve the existing attach-then-application behavior while ensuring both executions
    # receive only the path after all profiler environment variables have been configured.
    rocprofv3.main(
        [
            "--kernel-trace",
            "--pid",
            str(pid),
            "-E",
            str(definitions_file),
            "--",
            "/bin/ignored",
        ]
    )

    assert len(executions) == 2
    attach_kind, attach_argv, attach_env = executions[0]
    app_kind, app_argv, app_env = executions[1]
    assert attach_kind == "subprocess"
    assert os.path.basename(attach_argv[1]) == "rocprof-attach"
    assert app_kind == "exec"
    assert app_argv == ["/bin/ignored"]
    for env in (attach_env, app_env):
        assert env["ROCPROF_EXTRA_COUNTERS_FILE"] == os.path.abspath(definitions_file)
        assert "ROCPROF_EXTRA_COUNTERS_CONTENTS" not in env


def test_legacy_inline_contents_are_preserved_without_extra_counter_option(
    launch, monkeypatch
):
    monkeypatch.setenv("ROCPROF_EXTRA_COUNTERS_CONTENTS", "legacy inline contents")

    env = launch()

    assert env["ROCPROF_EXTRA_COUNTERS_CONTENTS"] == "legacy inline contents"
    assert "ROCPROF_EXTRA_COUNTERS_FILE" not in env


def test_missing_extra_counter_file_reports_absolute_path(rocprofv3, tmp_path, capsys):
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as error:
        rocprofv3.resolve_extra_counters_file(str(missing_file))

    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "Failed to access extra counter definition file" in message
    assert os.path.abspath(missing_file) in message


def test_unreadable_extra_counter_file_reports_os_error(
    rocprofv3, monkeypatch, tmp_path, capsys
):
    definitions_file = tmp_path / "unreadable.yaml"
    definitions_file.touch()

    def permission_denied(*args, **kwargs):
        raise PermissionError(
            errno.EACCES,
            os.strerror(errno.EACCES),
            str(definitions_file),
        )

    monkeypatch.setattr("builtins.open", permission_denied)

    with pytest.raises(SystemExit) as error:
        rocprofv3.resolve_extra_counters_file(str(definitions_file))

    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "Failed to open extra counter definition file" in message
    assert os.path.abspath(definitions_file) in message
    assert os.strerror(errno.EACCES) in message


def test_extra_counter_path_must_be_regular_file(rocprofv3, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        rocprofv3.resolve_extra_counters_file(str(tmp_path))

    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "is not a regular file" in message
    assert os.path.abspath(tmp_path) in message


def test_empty_extra_counter_file_warns(rocprofv3, tmp_path, capsys):
    definitions_file = tmp_path / "empty.yaml"
    definitions_file.touch()

    resolved_path = rocprofv3.resolve_extra_counters_file(str(definitions_file))

    assert resolved_path == os.path.abspath(definitions_file)
    assert f"Extra counter file '{resolved_path}' was empty." in capsys.readouterr().err
