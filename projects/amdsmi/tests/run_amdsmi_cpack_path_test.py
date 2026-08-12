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
run_amdsmi_cpack_path_test.py
=============================

The packaging rework installs the amdsmi Python module to an absolute system
path (site-packages / dist-packages) outside the /opt/rocm prefix. That path is
detected at configure time, so a packaging regression can silently ship the
module where no interpreter looks. This test inspects a BUILT .deb/.rpm and
asserts the amdsmi module files appear under a site-packages or dist-packages
directory, before the package is ever installed.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# amdsmi_interface.py is a stable module file the package must ship; matching it
# under a site/dist-packages path proves the module landed on an import path.
MODULE_MARKER = re.compile(r"(site-packages|dist-packages)/amdsmi/amdsmi_interface\.py$")


def _list_deb(pkg: Path) -> list:
    out = subprocess.run(
        ["dpkg", "-c", str(pkg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    # dpkg -c lines end with the path (may be "./usr/lib/..."); take the last field.
    paths = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts:
            paths.append(parts[-1].lstrip("."))
    return paths


def _list_rpm(pkg: Path) -> list:
    out = subprocess.run(
        ["rpm", "-qpl", str(pkg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def list_package(pkg: Path) -> list:
    suffix = pkg.suffix.lower()
    if suffix == ".deb":
        return _list_deb(pkg)
    if suffix == ".rpm":
        return _list_rpm(pkg)
    sys.exit(f"unsupported package type: {pkg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Built .deb or .rpm to inspect.")
    args = parser.parse_args()
    pkg = args.package
    if not pkg.is_file():
        sys.exit(f"package not found: {pkg}")

    paths = list_package(pkg)
    matches = [p for p in paths if MODULE_MARKER.search(p)]
    if not matches:
        sample = "\n  ".join(p for p in paths if "amdsmi" in p) or "(no amdsmi paths at all)"
        sys.exit(
            "amdsmi module not found under a site-packages/dist-packages path in "
            f"{pkg.name}.\namdsmi-related entries:\n  {sample}"
        )

    print(f"PASS: {pkg.name} ships the amdsmi module at:")
    for m in matches:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
