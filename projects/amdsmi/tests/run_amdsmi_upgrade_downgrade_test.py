#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_upgrade_downgrade_test.py
====================================

The packaging rework changed how the amdsmi Python module is delivered (direct
site-packages install instead of a postinst pip install). This test exercises
the deb/rpm upgrade and downgrade paths to prove the module, the ld.so.conf.d
entry, and the CLI stay consistent when moving between two package builds.

Given an OLD and a NEW package it:
  1. installs OLD, verifies import + CLI,
  2. installs NEW on top (upgrade), verifies again,
  3. reinstalls OLD (downgrade), verifies again,
so a broken pre/postinst transition fails here instead of on a user's machine.

Requires root (installs/removes system packages). Skips cleanly if the package
manager is unavailable. Not run by default in the build harness; wire it into
CI where a prior-version artifact is available.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_cli_version(amd_smi: str) -> None:
    # The CLI initializes the driver for every subcommand, so on a GPU-less CI
    # runner `amd-smi version` exits non-zero with "Drivers not loaded" instead
    # of printing a version. That still proves the CLI is installed and wired
    # up (the point of this packaging test), so accept it. Fail only when the
    # binary cannot be executed at all (missing/broken install).
    cmd = [amd_smi, "version"]
    print("+ " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
    except OSError as e:
        sys.exit(f"CLI {amd_smi} could not be executed: {e}")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0 and "Drivers not loaded" not in result.stderr:
        sys.exit(f"CLI {amd_smi} version failed with exit {result.returncode}")


def install_command(manager: str, pkg: str, downgrade: bool = False) -> list:
    if manager == "apt":
        return ["apt-get", "install", "-y", "--allow-downgrades", "--reinstall", pkg]
    if manager == "dnf":
        # dnf install refuses to downgrade; use dnf downgrade for that direction.
        # apt and zypper handle both directions with their existing flags.
        if downgrade:
            return ["dnf", "downgrade", "-y", pkg]
        return ["dnf", "install", "-y", pkg]
    if manager == "zypper":
        return ["zypper", "--non-interactive", "install", "--allow-downgrade", pkg]
    raise ValueError(f"unsupported package manager: {manager}")


def _install(pkg: Path, manager: str, downgrade: bool = False) -> None:
    _run(install_command(manager, str(pkg), downgrade))


def read_package_name(pkg: Path, manager: str) -> str:
    if manager == "apt":
        cmd = ["dpkg-deb", "-f", str(pkg), "Package"]
    elif manager in ("dnf", "zypper"):
        cmd = ["rpm", "-qp", "--qf", "%{NAME}", str(pkg)]
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    return subprocess.check_output(cmd, universal_newlines=True).strip()


def read_package_identity(pkg: Path, manager: str) -> str:
    if manager == "apt":
        cmd = ["dpkg-deb", "-f", str(pkg), "Package", "Version"]
    elif manager in ("dnf", "zypper"):
        cmd = ["rpm", "-qp", "--qf", "%{NAME}-%{EVR}", str(pkg)]
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    return subprocess.check_output(cmd, universal_newlines=True).strip()


def assert_distinct_packages(old_id: str, new_id: str) -> None:
    if old_id == new_id:
        sys.exit(f"OLD and NEW packages have the same identity: {old_id}")


def owner_package_name(owner_query_output: str, manager: str) -> str:
    """Exact name of the package owning a file, or "" when nothing owns it."""
    text = owner_query_output.strip()
    lowered = text.lower()
    if not text or "no path found" in lowered or "not owned by any package" in lowered:
        return ""
    first = text.splitlines()[0].strip()
    # `dpkg -S` answers "<package>: <path>"; the rpm query below asks for the
    # bare %{NAME}, so it needs no further parsing.
    return first.split(":", 1)[0].strip() if manager == "apt" else first


def import_is_package_owned(owner_query_output: str, manager: str, expected_package: str) -> bool:
    """Whether the imported module belongs to the package under test.

    Compares the owning package name exactly: a substring test would accept a
    sibling such as amd-smi-lib-tests, whose name contains the main package's.
    """
    return owner_package_name(owner_query_output, manager) == expected_package


def _verify(manager: str, expected_package: str) -> None:
    # Module imports and resolves a library path.
    import_cmd = ["python3", "-c", "import amdsmi; print(amdsmi.__file__)"]
    print("+ " + " ".join(import_cmd))
    module_file = subprocess.check_output(import_cmd, universal_newlines=True).strip()
    print(f"import OK from {module_file}")
    if manager == "apt":
        owner_cmd = ["dpkg", "-S", module_file]
    elif manager in ("dnf", "zypper"):
        owner_cmd = ["rpm", "-qf", "--qf", "%{NAME}", module_file]
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    print("+ " + " ".join(owner_cmd))
    owner = subprocess.run(
        owner_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    ).stdout
    print(owner, end="")
    if not import_is_package_owned(owner, manager, expected_package):
        sys.exit(
            f"amdsmi import resolved to {module_file}, owned by "
            f"'{owner_package_name(owner, manager) or 'nothing'}' rather than "
            f"'{expected_package}' (for example, a pip wheel under /usr/local)"
        )
    # CLI runs. The package installs amd-smi to <ROCM_PATH>/bin, which is not
    # on PATH in the CI container, so resolve it there directly rather than via
    # `which` (which would silently skip the check and let a broken CLI pass).
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    amd_smi = Path(rocm_path) / "bin" / "amd-smi"
    if not amd_smi.is_file():
        sys.exit(f"CLI {amd_smi} is missing after install/upgrade")
    _run_cli_version(str(amd_smi))
    # The ld.so.conf.d entry the postinst writes must survive install AND
    # upgrade. A prior package's %postun/prerm running after the new %post can
    # delete it (RPM transaction ordering), leaving the new package without its
    # linker registration -- so require the file rather than skipping when it is
    # absent, which is exactly the broken-upgrade state this test must catch.
    conf = Path("/etc/ld.so.conf.d/x86_64-libamd_smi_lib.conf")
    if not conf.is_file():
        sys.exit(f"linker registration {conf} is missing after install/upgrade")
    conf_lines = [line.strip() for line in conf.read_text().splitlines() if line.strip()]
    if not conf_lines:
        sys.exit("linker registration {} is empty; postinst wrote no library path".format(conf))
    libdir = conf_lines[0]
    if not list(Path(libdir).glob("libamd_smi.so*")):
        sys.exit(f"ld.so.conf.d points at {libdir} but no libamd_smi.so is there")


def _detect_manager() -> str:
    for mgr in ("apt-get", "dnf", "zypper"):
        if shutil.which(mgr):
            return {"apt-get": "apt", "dnf": "dnf", "zypper": "zypper"}[mgr]
    sys.exit("no supported package manager found")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-package", type=Path, required=True, help="Prior-version .deb/.rpm.")
    parser.add_argument("--new-package", type=Path, required=True, help="New .deb/.rpm to test.")
    parser.add_argument(
        "--package-manager", default=None, help="apt / dnf / zypper (default: auto)."
    )
    args = parser.parse_args(argv)

    for p in (args.old_package, args.new_package):
        if not p.is_file():
            sys.exit(f"package not found: {p}")

    manager = args.package_manager or _detect_manager()
    assert_distinct_packages(
        read_package_identity(args.old_package, manager),
        read_package_identity(args.new_package, manager),
    )

    expected_package = read_package_name(args.new_package, manager)

    print("=== 1. install OLD ===")
    _install(args.old_package, manager)
    _verify(manager, expected_package)

    print("=== 2. upgrade to NEW ===")
    _install(args.new_package, manager)
    _verify(manager, expected_package)

    print("=== 3. downgrade to OLD ===")
    _install(args.old_package, manager, downgrade=True)
    _verify(manager, expected_package)

    print("PASS: upgrade and downgrade keep the module, CLI, and linker config consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
