#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
run_amdsmi_pkg_conflict_test.py
================================

Sanity-check that the system-package shared object and the pip wheel's
bundled shared object ship under DIFFERENT SONAMEs, so a process that
somehow has both installed cannot accidentally double-load the same
library and crash during static C++ initialisation.

This is the only multi-package invariant we still validate. amd-smi
follows an "either you install the rpm/deb OR you install the wheel"
model -- the loader in py-interface/amdsmi_wrapper.py picks one .so
based solely on whether libamd_smi_python.so sits next to the wrapper
file, with no path-walking, no env-var ladders, and no .pth tricks.

Run it from inside a freshly-built tree where both artifacts exist:
    /opt/rocm/lib/libamd_smi.so.<MAJOR>           (system rpm/deb)
    <build>/.../libamd_smi_python.so.<MAJOR>      (wheel)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def soname(lib_path: Path) -> str:
    out = subprocess.check_output(
        ["readelf", "-d", str(lib_path)], text=True, stderr=subprocess.DEVNULL
    )
    for line in out.splitlines():
        if "(SONAME)" in line:
            return line.split("[", 1)[1].rstrip("]\n ")
    return ""


def find_system_lib() -> Path:
    candidates = [c for c in Path("/opt/rocm/lib").glob("libamd_smi.so.*") if not c.is_symlink()]
    if not candidates:
        sys.exit("System libamd_smi.so not found under /opt/rocm/lib; install the rpm/deb first.")
    return candidates[0]


def find_wheel_lib(repo_root: Path) -> Path:
    candidates = [c for c in repo_root.glob("**/libamd_smi_python.so.*") if not c.is_symlink()]
    if not candidates:
        sys.exit(
            "libamd_smi_python.so not found under the build tree; "
            "build with -DBUILD_PYTHON_WHEEL=ON first."
        )
    return candidates[0]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys_lib = find_system_lib()
    wheel_lib = find_wheel_lib(repo_root)

    sys_son = soname(sys_lib)
    wheel_son = soname(wheel_lib)

    print(f"system  {sys_lib}  SONAME={sys_son}")
    print(f"wheel   {wheel_lib}  SONAME={wheel_son}")

    if not sys_son or not wheel_son:
        sys.exit("could not read SONAMEs from both libraries")
    if sys_son == wheel_son:
        sys.exit(
            f"SONAME collision: {sys_son} vs {wheel_son} -- "
            "the two-library design (libamd_smi.so vs libamd_smi_python.so) is broken."
        )

    print("PASS: distinct SONAMEs; system and wheel cannot accidentally double-load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
