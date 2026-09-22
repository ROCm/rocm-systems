#!/usr/bin/env python3

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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the LD_PRELOAD ordering the rocprofv3 launcher builds."""

import os
import sys

import pytest

TOOL_LIBRARY = "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
SDK_LIBRARY = "lib/librocprofiler-sdk.so"
ROCTX_LIBRARY = "lib/librocprofiler-sdk-roctx.so"

# resolve_library_path() only checks that these exist, so empty files are enough.
STUB_LIBRARIES = (
    TOOL_LIBRARY,
    SDK_LIBRARY,
    ROCTX_LIBRARY,
    "lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so",
    "lib/rocprofiler-sdk/librocprofv3-list-avail.so",
)

ATTACH_CACHE_PREFIX = "/tmp/rocprofv3_attach_"


class _Launched(Exception):
    """Raised in place of the exec that would replace the test process."""


@pytest.fixture(autouse=True)
def clean_preload_env(monkeypatch):
    # A sanitizer build injects LD_PRELOAD into the test environment, which would
    # otherwise be indistinguishable from the value under test.
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("ROCPROF_PRELOAD", raising=False)


@pytest.fixture
def rocm_root(tmp_path):
    root = tmp_path / "rocm"
    for relpath in STUB_LIBRARIES:
        library = root / relpath
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    return root


@pytest.fixture
def launch(rocprofv3, rocm_root, tmp_path, monkeypatch):
    """Run the launcher and return the environment it would hand to the application."""
    cache = tmp_path / "attach-config"
    cache.mkdir()
    real_open = os.open

    # Attach mode writes its reattach configuration to a hardcoded /tmp path.
    def redirected_open(path, *args, **kwargs):
        name = os.fspath(path)
        if name.startswith(ATTACH_CACHE_PREFIX):
            name = str(cache / os.path.basename(name))
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(rocprofv3.os, "open", redirected_open)

    def _launch(*argv, application=True):
        captured = {}

        def execvpe(file, args, env):
            captured.update(env)
            raise _Launched

        monkeypatch.setattr(rocprofv3.os, "execvpe", execvpe)

        command = ["--rocm-root", str(rocm_root), *argv]
        if application:
            command += ["--", "/bin/true"]

        with pytest.raises(_Launched):
            rocprofv3.main(command)

        return captured

    return _launch


def expected_preload(rocm_root, *user_libraries, roctx=False):
    entries = [
        *user_libraries,
        str(rocm_root / TOOL_LIBRARY),
        str(rocm_root / SDK_LIBRARY),
    ]
    if roctx:
        entries.append(str(rocm_root / ROCTX_LIBRARY))
    return ":".join(entries)


def test_multiple_preloads_keep_order(launch, rocm_root):
    env = launch("--preload", "/opt/libA.so", "/opt/libB.so", "--kernel-trace")
    assert env["LD_PRELOAD"] == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libB.so"
    )


def test_preload_prepended_before_existing(launch, rocm_root, monkeypatch):
    # A pre-existing value is what distinguishes prepending from appending; without
    # one both produce the same string.
    monkeypatch.setenv("LD_PRELOAD", "/opt/libX.so")
    env = launch("--preload", "/opt/libA.so", "--kernel-trace")
    assert env["LD_PRELOAD"] == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libX.so"
    )


def test_rocprof_preload_env_default(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("ROCPROF_PRELOAD", "/opt/libA.so:/opt/libB.so")
    env = launch("--kernel-trace")
    assert env["LD_PRELOAD"] == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libB.so"
    )


def test_empty_entries_are_dropped(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("ROCPROF_PRELOAD", ":/opt/libA.so:")
    env = launch("--kernel-trace")
    assert env["LD_PRELOAD"] == expected_preload(rocm_root, "/opt/libA.so")


def test_marker_trace_appends_roctx_last(launch, rocm_root):
    env = launch("--marker-trace")
    assert env["LD_PRELOAD"] == expected_preload(rocm_root, roctx=True)


def test_suppress_marker_preload_drops_roctx(launch, rocm_root):
    env = launch("--marker-trace", "--suppress-marker-preload")
    assert env["LD_PRELOAD"] == expected_preload(rocm_root)


def test_preload_order_with_marker_trace(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/opt/libX.so")
    env = launch("--preload", "/opt/libA.so", "--marker-trace")
    assert env["LD_PRELOAD"] == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libX.so", roctx=True
    )


def test_attach_mode_skips_preload(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/opt/libX.so")
    env = launch("--pid", "12345", "--preload", "/opt/libA.so", application=False)
    assert env["LD_PRELOAD"] == "/opt/libX.so"
    assert env["ROCPROF_ATTACH_PID"] == "12345"


def test_attach_mode_appends_roctx(launch, rocm_root, monkeypatch):
    # The marker-trace append is not covered by the attach-mode guard, so the roctx
    # library is the one entry rocprofv3 adds to an attached process.
    monkeypatch.setenv("LD_PRELOAD", "/opt/libX.so")
    env = launch("--pid", "12345", "--marker-trace", application=False)
    assert env["LD_PRELOAD"] == f"/opt/libX.so:{rocm_root / ROCTX_LIBRARY}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
