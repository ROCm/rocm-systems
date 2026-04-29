#!/usr/bin/env python3
"""
build_wheel_rpm.py
==================

Build AMDSMI Python wheels on RPM-based hosts (RHEL, AlmaLinux, SLES,
AzureLinux) or inside a manylinux_2_28 container.

This script automates the full wheel-build pipeline that the CI workflow
(.github/workflows/manylinux-build.yml) performs:

    1. (optional) Install build prerequisites via dnf / zypper
    2. cmake configure  (-DBUILD_PYTHON_WHEEL=ON)
    3. make -j$(nproc)
    4. pip wheel --no-deps --no-build-isolation  (per interpreter)
    5. auditwheel repair  (optional, for manylinux tags)

Requirements
------------
* cmake, make, gcc/g++, git, python3, pip
* (optional) auditwheel -- installed automatically when repair is enabled
* (optional) /opt/python/cpXX-cpXX interpreters -- present in manylinux images

Supported OS Variants
---------------------
* RHEL 8 / 9 / 10  (dnf)
* AlmaLinux 8      (dnf)
* AzureLinux 3     (dnf)
* SLES             (zypper)

Examples
--------
Single-interpreter build on a bare RHEL host::

    python3 build_wheel_rpm.py --project-dir /path/to/amdsmi --skip-install

Multi-interpreter manylinux build inside quay.io/pypa/manylinux_2_28_x86_64::

    python3 build_wheel_rpm.py --project-dir /src/amdsmi

Specify an OS variant explicitly::

    python3 build_wheel_rpm.py --project-dir /src/amdsmi --os-variant RHEL8

Compatibility
-------------
* Python >= 3.6  (f-strings require 3.6; no 3.7+ APIs are used)
* Tested on RHEL 8/9/10, AlmaLinux 8, SLES 15, AzureLinux 3, and
  manylinux_2_28 containers.
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OS_VARIANTS = ("RHEL8", "RHEL9", "RHEL10", "AlmaLinux8", "SLES", "AzureLinux3")
QA_RPATHS_VARIANTS = ("RHEL10", "AlmaLinux8")

DNF_PREREQS = ["git", "make", "gcc", "gcc-c++", "cmake", "ninja-build", "openssl-devel"]
ZYPPER_PREREQS = ["git", "make", "gcc", "gcc-c++", "cmake", "ninja", "libopenssl-devel"]


# ---------------------------------------------------------------------------
# OS variant detection
# ---------------------------------------------------------------------------


def detect_os_variant():
    """Best-effort detection of the running OS variant from /etc/os-release."""
    os_release = Path("/etc/os-release")
    if os_release.exists():
        info = {}
        for line in os_release.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                info[k.strip()] = v.strip().strip('"')

        name = info.get("NAME", "").lower()
        version_id = info.get("VERSION_ID", "")

        if "azure linux" in name or "mariner" in name:
            return "AzureLinux3"
        if "alma" in name:
            return "AlmaLinux8"
        if "suse" in name or "sles" in name:
            return "SLES"
        if "rhel" in name or "red hat" in name:
            major = version_id.split(".")[0]
            return "RHEL%s" % major

    if shutil.which("zypper"):
        return "SLES"
    return "RHEL9"  # generic fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Generic helpers (run, abort, find_project_dir, best_effort_pip_upgrade,
# mark_safe_git_dir, write_temp_git_config) are shared with
# build_wheel_debian.py via _build_wheel_common.py. Edit them there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_wheel_common import (  # noqa: E402
    run,
    abort,
    find_project_dir,
    best_effort_pip_upgrade,
    mark_safe_git_dir,
    write_temp_git_config,
)


def collect_interpreters(python_bins_csv):
    """Return a list of Python interpreter paths.

    If *python_bins_csv* is given, split on commas.  Otherwise, discover
    /opt/python interpreters or fall back to the system ``python3``.
    """
    if python_bins_csv:
        return [p.strip() for p in python_bins_csv.split(",") if p.strip()]

    found = sorted(glob.glob("/opt/python/cp3*-cp3*/bin/python3"))
    if found:
        return found

    sys_py = shutil.which("python3")
    if sys_py:
        return [sys_py]
    return []


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def install_prerequisites(os_variant, skip):
    """Install distro build tools via dnf or zypper."""
    if skip:
        log.info("Skipping prerequisite installation (--skip-install).")
        return

    if os_variant == "SLES":
        log.info("Installing build prerequisites via zypper ...")
        run(["zypper", "--non-interactive", "refresh"])
        run(["zypper", "--non-interactive", "install", "-y"] + ZYPPER_PREREQS)
    else:
        log.info("Installing build prerequisites via dnf ...")
        run(["dnf", "-y", "install"] + DNF_PREREQS)

    if os_variant == "AzureLinux3":
        log.info("Installing more_itertools for AzureLinux3 ...")
        py3 = shutil.which("python3") or "python3"
        run([py3, "-m", "pip", "install", "more_itertools"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Build AMDSMI Python wheels (RPM-based OS / manylinux).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="AMDSMI project root (auto-detected if omitted).",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/tmp/amdsmi-build"),
        help="CMake build directory  [default: /tmp/amdsmi-build].",
    )
    p.add_argument(
        "--raw-wheels-dir",
        type=Path,
        default=Path("/tmp/raw-wheels"),
        help="Staging dir for raw wheels  [default: /tmp/raw-wheels].",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/amdsmi-wheels"),
        help="Final wheel output dir  [default: /tmp/amdsmi-wheels].",
    )
    p.add_argument(
        "--build-type",
        default="Release",
        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
        help="CMake build type  [default: Release].",
    )
    p.add_argument(
        "--enable-esmi",
        action="store_true",
        default=True,
        help="Enable ESMI library build (default).",
    )
    p.add_argument(
        "--no-esmi", action="store_false", dest="enable_esmi", help="Disable ESMI library build."
    )
    p.add_argument(
        "--python-bins", default=None, help="Comma-separated Python executables (auto-detected)."
    )
    p.add_argument(
        "--os-variant",
        choices=list(OS_VARIANTS),
        default=None,
        help="Target OS variant (auto-detected from /etc/os-release).",
    )
    p.add_argument(
        "--skip-install", action="store_true", help="Skip dnf/zypper prerequisite installation."
    )
    p.add_argument(
        "--repair",
        action="store_true",
        default=True,
        help="Run auditwheel repair for manylinux tags (default).",
    )
    p.add_argument(
        "--skip-repair",
        action="store_false",
        dest="repair",
        help="Skip the auditwheel repair step.",
    )
    p.add_argument("--build-tests", action="store_true", help="Build C/C++ tests  [default: OFF].")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- OS variant ------------------------------------------------------
    os_variant = args.os_variant or detect_os_variant()
    log.info("OS variant   : %s", os_variant)

    # --- project dir -----------------------------------------------------
    if args.project_dir is None:
        args.project_dir = find_project_dir(Path(__file__).resolve().parent.parent)
    args.project_dir = args.project_dir.resolve()

    # --- git safe.directory (avoid "dubious ownership" / +unknown VERSION_HASH)
    # CMake's get_package_version_number() runs `git rev-parse --short HEAD`;
    # in containers with bind-mounted repos, git refuses unless the path is
    # registered as safe. If the global config write fails (read-only HOME,
    # nobody user, etc.), fall back to a per-build GIT_CONFIG_GLOBAL.
    git_safe_paths = []
    if not mark_safe_git_dir(args.project_dir):
        git_safe_paths.append(args.project_dir)

    esmi_repo = args.project_dir / "esmi_ib_library"
    if esmi_repo.exists() and not mark_safe_git_dir(esmi_repo):
        git_safe_paths.append(esmi_repo)

    # --- interpreters ----------------------------------------------------
    python_interpreters = collect_interpreters(args.python_bins)
    if not python_interpreters:
        abort("No Python interpreters found.  Pass --python-bins explicitly.")
    cmake_python = python_interpreters[0]

    log.info("Project dir  : %s", args.project_dir)
    log.info("Build dir    : %s", args.build_dir)
    log.info("Raw wheels   : %s", args.raw_wheels_dir)
    log.info("Output dir   : %s", args.output_dir)
    log.info("Python (cmake): %s", cmake_python)
    log.info("Python targets: %s", ", ".join(python_interpreters))

    # --- prerequisites ---------------------------------------------------
    install_prerequisites(os_variant, args.skip_install)

    # --- prepare build directory -----------------------------------------
    args.build_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("CMakeCache.txt", "CMakeFiles"):
        p = args.build_dir / stale
        if p.exists():
            log.info("Removing stale %s", p)
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    # --- cmake configure & build -----------------------------------------
    extra_env = {}
    if os_variant in QA_RPATHS_VARIANTS:
        qa_rpaths = hex(0x0010 | 0x0002)
        extra_env["QA_RPATHS"] = qa_rpaths
        log.info("Setting QA_RPATHS=%s for %s", qa_rpaths, os_variant)

    if git_safe_paths:
        extra_env.update(write_temp_git_config(args.build_dir / "git-safe.config", git_safe_paths))

    run(
        [
            "cmake",
            str(args.project_dir),
            "-DBUILD_TESTS=" + ("ON" if args.build_tests else "OFF"),
            "-DENABLE_ESMI_LIB=" + ("ON" if args.enable_esmi else "OFF"),
            "-DBUILD_PYTHON_WHEEL=ON",
            "-DCMAKE_BUILD_TYPE=" + args.build_type,
            "-DPython3_EXECUTABLE=" + cmake_python,
        ],
        cwd=str(args.build_dir),
        env=extra_env or None,
    )

    run(["make", "-j" + str(os.cpu_count() or 4)], cwd=str(args.build_dir), env=extra_env or None)

    pkg_dir = args.build_dir / "py-interface" / "python_package"
    if not pkg_dir.exists():
        abort("Python package directory not found: " + str(pkg_dir))

    # --- prepare wheel directories ---------------------------------------
    raw_wheels_dir = args.raw_wheels_dir
    output_dir = args.output_dir

    if raw_wheels_dir.exists():
        log.info("Removing existing raw wheels dir %s", raw_wheels_dir)
        shutil.rmtree(raw_wheels_dir)
    raw_wheels_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.whl"):
        log.info("Removing existing wheel %s", existing)
        existing.unlink()

    # --- build wheels ----------------------------------------------------
    best_effort_pip_upgrade(
        cmake_python, ["pip", "setuptools", "wheel"] + (["auditwheel"] if args.repair else [])
    )

    build_interpreters = [cmake_python]
    if len(python_interpreters) > 1:
        skipped = ", ".join(python_interpreters[1:])
        log.info(
            "Wheel is pure Python; building once with %s (skipping: %s)", cmake_python, skipped
        )
    log.info("Building wheels for %d interpreter(s) ...", len(build_interpreters))

    for py in build_interpreters:
        if not os.path.isfile(py) or not os.access(py, os.X_OK):
            log.warning("Skipping missing/non-executable interpreter: %s", py)
            continue

        log.info("--- Building wheel with %s ---", py)
        best_effort_pip_upgrade(py, ["pip", "setuptools", "wheel"])

        for pattern in ("*.whl", "*.egg-info", "build", "dist"):
            for path in pkg_dir.glob(pattern):
                shutil.rmtree(path) if path.is_dir() else path.unlink()

        run(
            [
                py,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "-w",
                str(raw_wheels_dir),
                ".",
            ],
            cwd=str(pkg_dir),
        )

    raw_wheels = sorted(raw_wheels_dir.glob("*.whl"))
    if not raw_wheels:
        abort("No wheels found in " + str(raw_wheels_dir))

    log.info("=== Raw wheel(s) built ===")
    for w in raw_wheels:
        log.info("  %s  (%d KB)", w.name, w.stat().st_size // 1024)

    # --- auditwheel repair (optional) ------------------------------------
    if args.repair:
        auditwheel_ok = (
            subprocess.run(
                [cmake_python, "-m", "auditwheel", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).returncode
            == 0
        )

        if auditwheel_ok:
            log.info("Repairing wheels with auditwheel ...")
            for whl in raw_wheels:
                result = run(
                    [
                        cmake_python,
                        "-m",
                        "auditwheel",
                        "repair",
                        str(whl),
                        "--wheel-dir",
                        str(output_dir),
                    ],
                    check=False,
                )
                if result.returncode != 0:
                    log.warning("auditwheel repair failed for %s; copying raw wheel.", whl.name)
                    shutil.copy2(whl, output_dir)
        else:
            log.warning("auditwheel not available; copying raw wheels.")
            for whl in raw_wheels:
                shutil.copy2(whl, output_dir)
    else:
        for whl in raw_wheels:
            shutil.copy2(whl, output_dir)

    # --- summary ---------------------------------------------------------
    final_wheels = sorted(output_dir.glob("*.whl"))
    log.info("=== Final wheel(s) ===")
    for w in final_wheels:
        log.info("  %s  (%d KB)", w.name, w.stat().st_size // 1024)
    log.info("Done.  Wheels written to %s", output_dir)


if __name__ == "__main__":
    main()
