#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_component_removal_test.py
====================================

The amd-smi-lib-tests package depends on amd-smi-lib, but removing that optional
component must not run the main package's removal scripts. A packaging regression
did exactly that, deleting the main package's linker registration and, on RPM
systems, its log directory and logrotate configuration while amd-smi-lib remained
installed.

This test installs the main and tests packages, snapshots the main installation,
removes only the tests package, and proves the main package, Python module, linker
configuration, and any pre-existing logging paths survive intact.

Requires root and is intended to run only in the CI distro container.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


LINKER_CONFIG = Path("/etc/ld.so.conf.d/x86_64-libamd_smi_lib.conf")
OPTIONAL_PATHS = (Path("/var/log/amd_smi_lib"), Path("/etc/logrotate.d/amd_smi.conf"))


def _run(cmd: list) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_command(manager: str, package: str) -> list:
    if manager == "apt":
        return ["apt-get", "install", "-y", "--reinstall", package]
    if manager == "dnf":
        return ["dnf", "install", "-y", package]
    if manager == "zypper":
        return ["zypper", "--non-interactive", "install", package]
    raise ValueError(f"unsupported package manager: {manager}")


def remove_command(manager: str, package_name: str) -> list:
    if manager == "apt":
        return ["apt-get", "remove", "-y", package_name]
    if manager == "dnf":
        return ["dnf", "remove", "-y", package_name]
    if manager == "zypper":
        return ["zypper", "--non-interactive", "remove", package_name]
    raise ValueError(f"unsupported package manager: {manager}")


def read_package_name(package: Path, manager: str) -> str:
    if manager == "apt":
        cmd = ["dpkg-deb", "-f", str(package), "Package"]
    elif manager in ("dnf", "zypper"):
        cmd = ["rpm", "-qp", "--qf", "%{NAME}", str(package)]
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    return subprocess.check_output(cmd, universal_newlines=True).strip()


def package_is_installed(manager: str, package_name: str) -> bool:
    if manager == "apt":
        cmd = ["dpkg-query", "-W", "-f=${Status}", package_name]
        expected = "install ok installed"
    elif manager in ("dnf", "zypper"):
        cmd = ["rpm", "-q", package_name]
        expected = None
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True
    )
    return result.returncode == 0 and (expected is None or result.stdout.strip() == expected)


def module_owner(manager: str, module_file: str) -> str:
    if manager == "apt":
        cmd = ["dpkg", "-S", module_file]
    elif manager in ("dnf", "zypper"):
        cmd = ["rpm", "-qf", "--qf", "%{NAME}", module_file]
    else:
        raise ValueError(f"unsupported package manager: {manager}")
    print("+ " + " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        return ""
    if manager == "apt":
        return result.stdout.partition(":")[0].strip()
    return result.stdout.strip()


def import_module_file() -> str:
    cmd = ["python3", "-c", "import amdsmi; print(amdsmi.__file__)"]
    print("+ " + " ".join(cmd))
    return subprocess.check_output(cmd, universal_newlines=True).strip()


def linker_library_directory(config_text: str) -> Path:
    lines = [line.strip() for line in config_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("linker registration is empty")
    return Path(lines[0])


def _detect_manager() -> str:
    for mgr in ("apt-get", "dnf", "zypper"):
        if shutil.which(mgr):
            return {"apt-get": "apt", "dnf": "dnf", "zypper": "zypper"}[mgr]
    sys.exit("no supported package manager found")


def snapshot_main_state(manager: str, main_package_name: str) -> dict:
    if not package_is_installed(manager, main_package_name):
        sys.exit(f"main package {main_package_name} is not installed before component removal")
    if not LINKER_CONFIG.is_file():
        sys.exit(f"main package linker registration {LINKER_CONFIG} is missing before removal")
    config_text = LINKER_CONFIG.read_text()
    try:
        libdir = linker_library_directory(config_text)
    except ValueError as error:
        sys.exit(f"main package {error} before component removal")
    if not list(libdir.glob("libamd_smi.so*")):
        sys.exit(f"main package linker registration points at {libdir} without libamd_smi.so")
    try:
        module_file = import_module_file()
    except subprocess.CalledProcessError:
        sys.exit("main package Python module does not import before component removal")
    if module_owner(manager, module_file) != main_package_name:
        sys.exit(
            f"main package Python module {module_file} is not owned by {main_package_name} "
            "before component removal"
        )
    return {
        "config_text": config_text,
        "module_file": module_file,
        "optional_paths": {path: path.exists() for path in OPTIONAL_PATHS},
    }


def verify_main_survived(manager: str, main_package_name: str, snapshot: dict) -> None:
    if not LINKER_CONFIG.is_file():
        sys.exit(f"tests-package removal destroyed linker registration {LINKER_CONFIG}")
    config_text = LINKER_CONFIG.read_text()
    if config_text != snapshot["config_text"]:
        sys.exit(f"tests-package removal changed linker registration {LINKER_CONFIG}")
    try:
        libdir = linker_library_directory(config_text)
    except ValueError:
        sys.exit(f"tests-package removal emptied linker registration {LINKER_CONFIG}")
    if not list(libdir.glob("libamd_smi.so*")):
        sys.exit(
            f"tests-package removal left linker registration pointing at {libdir}, "
            "but destroyed libamd_smi.so"
        )
    if not package_is_installed(manager, main_package_name):
        sys.exit(f"tests-package removal uninstalled main package {main_package_name}")
    try:
        module_file = import_module_file()
    except subprocess.CalledProcessError:
        sys.exit("tests-package removal destroyed the main package's Python module import")
    if module_file != snapshot["module_file"]:
        sys.exit(
            f"tests-package removal changed the main package Python module from "
            f"{snapshot['module_file']} to {module_file}"
        )
    if module_owner(manager, module_file) != main_package_name:
        sys.exit(
            f"tests-package removal left Python module {module_file} no longer owned by "
            f"main package {main_package_name}"
        )
    for path, existed in snapshot["optional_paths"].items():
        if existed and not path.exists():
            sys.exit(f"tests-package removal destroyed main package path {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-package", type=Path, required=True, help="Main .deb/.rpm package.")
    parser.add_argument(
        "--tests-package", type=Path, required=True, help="Matching tests .deb/.rpm package."
    )
    parser.add_argument(
        "--package-manager",
        choices=("apt", "dnf", "zypper"),
        default=None,
        help="apt / dnf / zypper (default: auto).",
    )
    args = parser.parse_args(argv)

    if os.geteuid() != 0:
        sys.exit("this package installation test requires root")
    for package in (args.main_package, args.tests_package):
        if not package.is_file():
            sys.exit(f"package not found: {package}")

    manager = args.package_manager or _detect_manager()
    main_package_name = read_package_name(args.main_package, manager)
    tests_package_name = read_package_name(args.tests_package, manager)

    print("=== 1. install main package ===")
    _run(install_command(manager, str(args.main_package)))
    print("=== 2. install tests package ===")
    _run(install_command(manager, str(args.tests_package)))
    snapshot = snapshot_main_state(manager, main_package_name)
    print("=== 3. remove tests package only ===")
    _run(remove_command(manager, tests_package_name))
    verify_main_survived(manager, main_package_name, snapshot)

    print("PASS: removing the tests package left the main package intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
