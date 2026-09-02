#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath


def parse_dpkg_listing(text: str) -> list:
    paths = []
    for line in text.splitlines():
        member = line.split(" -> ", 1)[0]
        fields = member.split(None, 5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if path.startswith("./"):
            path = path[1:]
        paths.append(path)
    return paths


def _list_deb(pkg: Path) -> list:
    out = subprocess.run(
        ["dpkg", "-c", str(pkg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return parse_dpkg_listing(out.stdout)


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


def select_module_paths(paths, site_dirs) -> list:
    expected = {
        PurePosixPath(
            os.path.normpath(str(PurePosixPath(site_dir) / "amdsmi" / "amdsmi_interface.py"))
        )
        for site_dir in site_dirs
    }
    return [path for path in paths if PurePosixPath(os.path.normpath(path)) in expected]


def default_python() -> str:
    """Pick the interpreter the package install path was derived from.

    py-interface/CMakeLists.txt detects the module destination with
    /usr/libexec/platform-python when it exists (the stable RHEL/Fedora handle
    on the default interpreter) and /usr/bin/python3 otherwise. Mirror that
    order rather than using sys.executable: on the RHEL family the two can be
    different minors, and checking the wrong one reports a correctly packaged
    module as misplaced.
    """
    for candidate in ("/usr/libexec/platform-python", "/usr/bin/python3"):
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def discover_site_dirs(python_exe) -> list:
    command = "import site, json; print(json.dumps(site.getsitepackages()))"
    try:
        out = subprocess.run(
            [python_exe, "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        site_dirs = json.loads(out.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        sys.exit(
            "failed to discover site directories with {}: {}".format(python_exe, detail.strip())
        )
    if not isinstance(site_dirs, list):
        sys.exit(
            "failed to discover site directories with {}: expected a JSON list".format(python_exe)
        )
    return site_dirs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Built .deb or .rpm to inspect.")
    parser.add_argument(
        "--site-dir",
        action="append",
        dest="site_dirs",
        help="Expected interpreter site directory (repeatable; skips discovery).",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used to discover site directories "
        "(default: the interpreter the package install path was derived from).",
    )
    args = parser.parse_args(argv)
    pkg = args.package
    if not pkg.is_file():
        sys.exit(f"package not found: {pkg}")

    site_dirs = args.site_dirs or discover_site_dirs(args.python or default_python())
    paths = list_package(pkg)
    matches = select_module_paths(paths, site_dirs)
    if not matches:
        sample = "\n  ".join(p for p in paths if "amdsmi" in p) or "(no amdsmi paths at all)"
        expected = "\n  ".join(site_dirs) or "(no site directories reported)"
        sys.exit(
            f"amdsmi module not found under an expected site directory in {pkg.name}.\n"
            f"expected site directories:\n  {expected}\namdsmi-related entries:\n  {sample}"
        )

    print(f"PASS: {pkg.name} ships the amdsmi module at:")
    for m in matches:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
