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
#
# SPM runner preflight for rocprofiler-sdk CI. The pinned profiler runner is
# expected to satisfy these requirements; this script fails fast with a clear
# message when it does not (see source/docs/how-to/using-spm.rst).

import sys
from pathlib import Path
from typing import Optional, Tuple

# Single source of truth for SPM CI driver gating. CMake parses this assignment;
# TheRock CI runs this script from the installed tests tree.
SPM_MIN_AMDGPU_DRIVER_VERSION = "6.19.14.31400000"

AMDGPU_VERSION_PATH = Path("/sys/module/amdgpu/version")


def parse_dot_version(version):
    # type: (str) -> Tuple[int, ...]
    return tuple(int(part) for part in version.strip().split("."))


def read_amdgpu_driver_version():
    # type: () -> Optional[str]
    if not AMDGPU_VERSION_PATH.exists():
        return None
    return AMDGPU_VERSION_PATH.read_text(encoding="utf-8").strip()


def version_ge(version_a, version_b):
    # type: (str, str) -> bool
    return parse_dot_version(version_a) >= parse_dot_version(version_b)


def check_spm_runner_requirements(min_driver=SPM_MIN_AMDGPU_DRIVER_VERSION):
    # type: (str) -> None
    current = read_amdgpu_driver_version()
    if not current:
        raise SystemExit(
            "SPM runner preflight failed: /sys/module/amdgpu/version is unavailable. "
            "SPM tests require an amdgpu kernel module."
        )
    if not version_ge(current, min_driver):
        raise SystemExit(
            "SPM runner preflight failed: "
            "amdgpu driver {} < required {}.".format(current, min_driver)
        )


def main():
    # type: () -> int
    check_spm_runner_requirements()
    return 0


if __name__ == "__main__":
    sys.exit(main())
