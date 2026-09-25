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
import subprocess
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

# Stands in for the profiled application and reports the LD_PRELOAD it receives.
REPORT_PREFIX = "LD_PRELOAD="
REPORT_LD_PRELOAD = [
    sys.executable,
    "-c",
    f"import os; print('{REPORT_PREFIX}' + os.environ.get('LD_PRELOAD', ''))",
]


@pytest.fixture
def rocm_root(tmp_path):
    root = tmp_path / "rocm"
    for relpath in STUB_LIBRARIES:
        library = root / relpath
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    return root


@pytest.fixture
def launch(rocprofv3, rocm_root):
    """Run the launcher and return the LD_PRELOAD its application receives."""

    def _launch(*argv, env=None):
        # A sanitizer build injects LD_PRELOAD into the test environment, which would
        # otherwise be indistinguishable from the value under test.
        environ = {
            key: value
            for key, value in os.environ.items()
            if key not in ("LD_PRELOAD", "ROCPROF_PRELOAD")
        }
        environ.update(env or {})

        result = subprocess.run(
            [
                sys.executable,
                rocprofv3.__file__,
                "--rocm-root",
                str(rocm_root),
                *argv,
                "--",
                *REPORT_LD_PRELOAD,
            ],
            env=environ,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        reports = [
            line[len(REPORT_PREFIX) :]
            for line in result.stdout.splitlines()
            if line.startswith(REPORT_PREFIX)
        ]
        assert len(reports) == 1, result.stdout
        return reports[0]

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
    assert launch(
        "--preload", "/opt/libA.so", "/opt/libB.so", "--kernel-trace"
    ) == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libB.so"
    ), "--preload entries must keep the order they were given on the command line"


def test_preload_prepended_before_existing(launch, rocm_root):
    # A pre-existing value is what distinguishes prepending from appending; without
    # one both produce the same string.
    assert launch(
        "--preload", "/opt/libA.so", "--kernel-trace", env={"LD_PRELOAD": "/opt/libX.so"}
    ) == expected_preload(rocm_root, "/opt/libA.so", "/opt/libX.so"), (
        "--preload must precede a pre-existing LD_PRELOAD, which must precede the "
        "tool and SDK libraries"
    )


def test_rocprof_preload_env_default(launch, rocm_root):
    assert launch(
        "--kernel-trace", env={"ROCPROF_PRELOAD": "/opt/libA.so:/opt/libB.so"}
    ) == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libB.so"
    ), "ROCPROF_PRELOAD must supply the --preload default, split on colons"


def test_cli_preload_overrides_env_default(launch, rocm_root):
    assert launch(
        "--preload",
        "/opt/libA.so",
        "--kernel-trace",
        env={"ROCPROF_PRELOAD": "/opt/libEnv.so"},
    ) == expected_preload(
        rocm_root, "/opt/libA.so"
    ), "an explicit --preload must replace the ROCPROF_PRELOAD default"


def test_empty_entries_are_dropped(launch, rocm_root):
    assert launch(
        "--kernel-trace", env={"ROCPROF_PRELOAD": ":/opt/libA.so:"}
    ) == expected_preload(
        rocm_root, "/opt/libA.so"
    ), "empty preload entries must be dropped rather than left as bare separators"


def test_marker_trace_appends_roctx_last(launch, rocm_root):
    assert launch("--marker-trace") == expected_preload(
        rocm_root, roctx=True
    ), "marker tracing must append the roctx library after the tool and SDK libraries"


def test_suppress_marker_preload_drops_roctx(launch, rocm_root):
    assert launch("--marker-trace", "--suppress-marker-preload") == expected_preload(
        rocm_root
    ), "--suppress-marker-preload must drop the roctx library"


def test_preload_order_with_marker_trace(launch, rocm_root):
    assert launch(
        "--preload", "/opt/libA.so", "--marker-trace", env={"LD_PRELOAD": "/opt/libX.so"}
    ) == expected_preload(
        rocm_root, "/opt/libA.so", "/opt/libX.so", roctx=True
    ), "the full order is user, pre-existing, tool, SDK, then roctx"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
