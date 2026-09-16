#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

"""System-boundary tests: the real SystemAccessor against real processes and
files, no GPU required.

The fake accessor used elsewhere encodes assumptions about what the system
does; these tests check those assumptions where getting them wrong would be
destructive (the write probe) or would take the doctor down (hung or crashing
children, native imports).
"""

import os
import sys
import time

import pytest


@pytest.fixture
def env(rocprofv3_package):
    from rocprofv3 import doctor_env

    return doctor_env


@pytest.fixture
def accessor(env, tmp_path):
    return env.SystemAccessor(rocm_root=str(tmp_path))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie still answers kill(0); it is dead for our purposes
    try:
        with open("/proc/{}/stat".format(pid)) as handle:
            return handle.read().split(")")[-1].split()[0] != "Z"
    except (IOError, OSError):
        return False


# ----------------------------------------------------------------------
# write probe
# ----------------------------------------------------------------------
def test_doctor_system_touch_probe_never_writes_through_symlink(accessor, tmp_path):
    """Review P1: the old fixed-name probe truncated a symlink's target."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    for pid in (os.getpid(), os.getpid() + 1):
        os.symlink(str(victim), str(probe_dir / ".rocprofv3-doctor-probe-{}".format(pid)))

    assert accessor.touch_probe(str(probe_dir)) is True
    assert victim.read_text() == "precious"


def test_doctor_system_touch_probe_leaves_existing_files_and_no_residue(
    accessor, tmp_path
):
    keep = tmp_path / ".rocprofv3-doctor-probe-{}".format(os.getpid())
    keep.write_text("mine")
    before = sorted(os.listdir(str(tmp_path)))

    assert accessor.touch_probe(str(tmp_path)) is True
    assert keep.read_text() == "mine"
    assert sorted(os.listdir(str(tmp_path))) == before


def test_doctor_system_touch_probe_unwritable(accessor, tmp_path):
    if os.getuid() == 0:
        pytest.skip("root can write anywhere")
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(str(locked), 0o500)
    try:
        assert accessor.touch_probe(str(locked)) is False
    finally:
        os.chmod(str(locked), 0o700)


# ----------------------------------------------------------------------
# run(): timeouts, signals, process trees
# ----------------------------------------------------------------------
def test_doctor_system_run_timeout_kills_the_whole_process_tree(env, accessor, tmp_path):
    """Review P2: killing only the immediate child left grandchildren running."""
    pidfile = tmp_path / "grandchild.pid"
    script = "sleep 60 & echo $! > {}; wait".format(pidfile)
    returncode, _, stderr = accessor.run(["sh", "-c", script], timeout=1)

    assert returncode == env.RUN_TIMEOUT
    assert "timed out" in stderr
    grandchild = int(pidfile.read_text())
    deadline = time.time() + 5
    while _alive(grandchild) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_doctor_system_run_signal_is_not_a_timeout(env, accessor):
    """Review P2: -1 used to mean both "timed out" and "killed by SIGHUP"."""
    returncode, _, _ = accessor.run(["sh", "-c", "kill -HUP $$"], timeout=10)
    assert returncode == -1
    assert returncode != env.RUN_TIMEOUT
    assert env.describe_returncode(returncode) == "crashed (SIGHUP)"


def test_doctor_system_run_missing_executable(env, accessor):
    returncode, _, _ = accessor.run(["/nonexistent/rocprofv3-doctor-test"])
    assert returncode == env.RUN_SPAWN_FAILED


def test_doctor_system_run_output_is_bounded(env, accessor):
    count = env.MAX_OUTPUT_CHARS * 2
    _, stdout, _ = accessor.run(
        [sys.executable, "-c", "print('x' * {} + 'END')".format(count)], timeout=30
    )
    assert len(stdout) == env.MAX_OUTPUT_CHARS
    assert stdout.rstrip().endswith("END")


# ----------------------------------------------------------------------
# imports run in a child
# ----------------------------------------------------------------------
@pytest.fixture
def module_dir(tmp_path, monkeypatch):
    path = tmp_path / "modules"
    path.mkdir()
    monkeypatch.syspath_prepend(str(path))
    return path


def test_doctor_system_import_that_aborts_does_not_kill_the_doctor(accessor, module_dir):
    """Review P2: a native extension aborting at import used to abort the
    whole doctor. It is now one failed import."""
    (module_dir / "doctor_test_aborts.py").write_text("import os\nos.abort()\n")
    ok, detail = accessor.can_import("doctor_test_aborts")
    assert ok is False
    assert "SIGABRT" in detail


def test_doctor_system_import_sees_the_doctors_sys_path(accessor, module_dir):
    """The child resolves modules exactly as an in-process import would."""
    (module_dir / "doctor_test_ok.py").write_text("__version__ = '9.9'\n")
    assert accessor.can_import("doctor_test_ok") == (True, "9.9")
    assert accessor.module_file("doctor_test_ok").endswith("doctor_test_ok.py")


def test_doctor_system_import_failure_reports_the_exception(accessor, module_dir):
    (module_dir / "doctor_test_raises.py").write_text("raise ImportError('nope')\n")
    ok, detail = accessor.can_import("doctor_test_raises")
    assert ok is False
    assert "ImportError: nope" in detail


def test_doctor_system_import_output_noise_is_tolerated(accessor, module_dir):
    (module_dir / "doctor_test_chatty.py").write_text("print('hello from import')\n")
    assert accessor.can_import("doctor_test_chatty")[0] is True


# ----------------------------------------------------------------------
# temporary directories
# ----------------------------------------------------------------------
def test_doctor_system_temp_dir_is_private_and_removable(accessor):
    path = accessor.make_temp_dir()
    try:
        assert os.path.isdir(path)
        assert os.stat(path).st_mode & 0o077 == 0
    finally:
        accessor.remove_tree(path)
    assert not os.path.exists(path)
