#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    $ROCM_PATH/lib/libamd_smi.so.<MAJOR>          (system rpm/deb; /opt/rocm default)
    <build>/.../libamd_smi_python.so.<MAJOR>      (wheel)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _version_key(lib_path: Path) -> tuple:
    """Sort key ordering libfoo.so.<N>[.<M>...] by NUMERIC version components.

    A plain string sort is wrong here: lexically "libfoo.so.24" precedes
    "libfoo.so.9", so ``sorted(...)[-1]`` would pick the older major. Parse
    the trailing dotted-integer version and sort on the integer tuple so the
    newest library is unambiguously last.
    """
    digits = re.findall(r"\d+", lib_path.name.split(".so.", 1)[-1])
    return tuple(int(d) for d in digits)


def soname(lib_path: Path) -> str:
    if shutil.which("readelf") is None:
        sys.exit("readelf not found; install binutils to run this check.")
    try:
        out = subprocess.check_output(["readelf", "-d", str(lib_path)], text=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"readelf failed on {lib_path} (exit {exc.returncode}): {exc}")
    for line in out.splitlines():
        if "(SONAME)" in line:
            return line.split("[", 1)[1].rstrip("]\n ")
    return ""


def _rocm_lib_dir() -> Path:
    """System library directory, honoring ROCM_PATH / ROCM_HOME over /opt/rocm."""
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    return Path(rocm_path) / "lib"


def find_system_lib(build_root: Path) -> Path:
    # Prefer the installed system library (rpm/deb layout). When it is absent
    # -- e.g. the manylinux wheel job builds both libraries but installs no
    # system package -- fall back to the build tree, which holds the same
    # libamd_smi.so.<MAJOR> the package would ship.
    lib_dir = _rocm_lib_dir()
    candidates = sorted(
        (c for c in lib_dir.glob("libamd_smi.so.*") if not c.is_symlink()), key=_version_key
    )
    if not candidates:
        candidates = sorted(
            (c for c in build_root.glob("**/libamd_smi.so.*") if not c.is_symlink()),
            key=_version_key,
        )
    if not candidates:
        sys.exit(
            f"System libamd_smi.so not found under {lib_dir} or the build tree "
            f"({build_root}); install the rpm/deb or build first."
        )
    return candidates[-1]


def find_wheel_lib(build_root: Path) -> Path:
    candidates = sorted(
        (c for c in build_root.glob("**/libamd_smi_python.so.*") if not c.is_symlink()),
        key=_version_key,
    )
    if not candidates:
        sys.exit(
            f"libamd_smi_python.so not found under {build_root}; "
            "build with -DBUILD_PYTHON_WHEEL=ON first."
        )
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory to search for the built libraries "
        "(default: the amdsmi project root; use e.g. /tmp/amdsmi-build when the "
        "wheel was built out-of-tree).",
    )
    args = parser.parse_args()
    build_root = args.build_root.resolve()
    sys_lib = find_system_lib(build_root)
    wheel_lib = find_wheel_lib(build_root)

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
