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

"""GPU-free unit tests for the rocprofv3 reattachment configuration cache.

Reattaching to a PID must reuse the data-collection options of the first
attachment; ``using-rocprofv3-process-attachment.rst`` documents a ``RuntimeError``
when an option protected by the comparison filter changes.
"""

import os
import pickle
import sys

import pytest

# resolve_library_path() only checks that these exist, so empty files are enough.
STUB_LIBRARIES = (
    "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
    "lib/librocprofiler-sdk.so",
    "lib/librocprofiler-sdk-roctx.so",
    "lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so",
    "lib/rocprofiler-sdk/librocprofv3-list-avail.so",
)

ATTACH_CACHE_PREFIX = "/tmp/rocprofv3_attach_"
TARGET_PID = "12345"
CACHE_NAME = f"rocprofv3_attach_{TARGET_PID}.pkl"


class _Launched(Exception):
    """Raised in place of the exec that would replace the test process."""


@pytest.fixture
def rocm_root(tmp_path):
    root = tmp_path / "rocm"
    for relpath in STUB_LIBRARIES:
        library = root / relpath
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    return root


@pytest.fixture
def cache_dir(rocprofv3, tmp_path, monkeypatch):
    """Redirect the launcher's hardcoded /tmp configuration cache into tmp_path."""
    cache = tmp_path / "attach-config"
    cache.mkdir()
    real_open = os.open

    # The launcher resolves os.open through the os module, so this replaces it
    # process-wide; the prefix check keeps every other caller on the real one.
    def redirected_open(path, *args, **kwargs):
        name = os.fspath(path)
        if name.startswith(ATTACH_CACHE_PREFIX):
            name = str(cache / os.path.basename(name))
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(rocprofv3.os, "open", redirected_open)
    return cache


@pytest.fixture
def captured_exec(rocprofv3, monkeypatch):
    """Stand in for the exec that would otherwise replace the test process.

    Tests that expect a fatal error need this too: without it, a regression that
    stops rejecting the option runs on and execs over the test runner instead of
    reporting a failure.
    """
    captured = {}

    def execvpe(file, args, env):
        captured.clear()
        captured.update(env)
        raise _Launched

    monkeypatch.setattr(rocprofv3.os, "execvpe", execvpe)
    return captured


@pytest.fixture
def attach(rocprofv3, rocm_root, cache_dir, captured_exec):
    """Attach to a fixed PID and return the environment built for rocprof-attach."""

    def _attach(*argv):
        with pytest.raises(_Launched):
            rocprofv3.main(["--rocm-root", str(rocm_root), "--attach", TARGET_PID, *argv])

        return dict(captured_exec)

    return _attach


def test_first_attach_saves_config(attach, cache_dir):
    attach("--marker-trace", "--attach-duration-msec", "500")

    # The file is created before anything is written, so its presence alone would
    # not show that the configuration was saved.
    with open(cache_dir / CACHE_NAME, "rb") as cache:
        saved = pickle.load(cache)

    assert saved.pid == int(TARGET_PID)
    assert saved.marker_trace


def test_identical_reattach_is_accepted(attach):
    options = ("--marker-trace", "--attach-duration-msec", "500")

    attach(*options)
    env = attach(*options)

    assert env["ROCPROF_ATTACH_PID"] == TARGET_PID


def test_changed_trace_option_is_rejected(attach):
    attach("--marker-trace", "--attach-duration-msec", "500")

    # Exactly one protected option is added, so the reported option is deterministic:
    # get_args() walks an unordered set, and naming is arbitrary when several differ.
    with pytest.raises(RuntimeError, match="kernel_trace"):
        attach("--marker-trace", "--kernel-trace", "--attach-duration-msec", "500")


def test_config_survives_rejected_reattach(attach, cache_dir):
    options = ("--marker-trace", "--attach-duration-msec", "500")
    attach(*options)
    original = (cache_dir / CACHE_NAME).read_bytes()

    with pytest.raises(RuntimeError):
        attach("--marker-trace", "--kernel-trace", "--attach-duration-msec", "500")

    assert (cache_dir / CACHE_NAME).read_bytes() == original
    # The original configuration is still the one a matching reattach compares against.
    assert attach(*options)["ROCPROF_ATTACH_PID"] == TARGET_PID


def test_collection_period_is_rejected(
    rocprofv3, rocm_root, cache_dir, captured_exec, capsys
):
    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(
            [
                "--rocm-root",
                str(rocm_root),
                "--attach",
                TARGET_PID,
                "--collection-period",
                "0:100:1",
                "--marker-trace",
            ]
        )

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        "[rocprofv3] Fatal error: --collection-period is not compatible with "
        "attach mode\n"
    )
    assert not (cache_dir / CACHE_NAME).exists()


def test_symlinked_config_is_refused(
    rocprofv3, rocm_root, cache_dir, captured_exec, tmp_path, capsys
):
    # O_NOFOLLOW must reject a cache path redirected through a symlink.
    planted = tmp_path / "planted.pkl"
    planted.touch()
    (cache_dir / CACHE_NAME).symlink_to(planted)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(
            [
                "--rocm-root",
                str(rocm_root),
                "--attach",
                TARGET_PID,
                "--marker-trace",
            ]
        )

    assert exc_info.value.code == 1
    assert "Could not open attach configuration file" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
