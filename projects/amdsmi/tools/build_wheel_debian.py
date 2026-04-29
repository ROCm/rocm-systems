#!/usr/bin/env python3
"""
build_wheel_debian.py
=====================

Build AMDSMI Python wheels on Debian / Ubuntu hosts or inside a
manylinux_2_28 container.

This script automates the full wheel-build pipeline that the CI workflow
(.github/workflows/manylinux-build.yml) performs:

    1. cmake configure  (-DBUILD_PYTHON_WHEEL=ON)
    2. make -j$(nproc)
    3. pip wheel --no-deps --no-build-isolation  (per interpreter)
    4. auditwheel repair  (optional, for manylinux tags)

Requirements
------------
* cmake, make, gcc/g++, git, python3, pip
* (optional) auditwheel -- installed automatically when --repair is used
* (optional) /opt/python/cpXX-cpXX interpreters -- present in manylinux images

Examples
--------
Single-interpreter build on a bare Debian/Ubuntu host::

    python3 build_wheel_debian.py --project-dir /path/to/amdsmi

Multi-interpreter manylinux build inside quay.io/pypa/manylinux_2_28_x86_64 docker container::

    python3 build_wheel_debian.py --project-dir /src/amdsmi --all-pythons

Compatibility
-------------
* Python >= 3.6  (uses ``universal_newlines`` instead of ``text``,
  ``stdout/stderr=PIPE`` instead of ``capture_output``)
* Tested on Ubuntu 20.04 / 22.04 / 24.04 and manylinux_2_28 containers.
"""

import argparse
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
DEFAULT_BUILD_DIR = Path("/tmp/amdsmi-build")
DEFAULT_RAW_WHEELS_DIR = Path("/tmp/raw-wheels")
DEFAULT_OUTPUT_NAME = "wheels"

# Ordered to match .github/workflows/manylinux-build.yml
MANYLINUX_PYTHONS = [
    "/opt/python/cp38-cp38/bin/python3",
    "/opt/python/cp39-cp39/bin/python3",
    "/opt/python/cp310-cp310/bin/python3",
    "/opt/python/cp311-cp311/bin/python3",
    "/opt/python/cp312-cp312/bin/python3",
    "/opt/python/cp313-cp313/bin/python3",
]


# ---------------------------------------------------------------------------
# Helpers  (all Python 3.6-safe)
# ---------------------------------------------------------------------------
# Generic helpers (run, abort, find_project_dir, best_effort_pip_upgrade,
# mark_safe_git_dir, write_temp_git_config) are shared with build_wheel_rpm.py
# via _build_wheel_common.py. Edit them there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_wheel_common import (  # noqa: E402
    run,
    abort,
    find_project_dir,
    best_effort_pip_upgrade,
    mark_safe_git_dir,
    write_temp_git_config,
)


def ensure_writable_dir(path, label):
    """Return True if *path* is writable, attempting ``sudo chown`` if not."""
    if os.access(path, os.W_OK):
        return True
    log.warning("%s not writable; attempting sudo chown ...", label)
    result = subprocess.run(
        ["sudo", "chown", "-R", "%d:%d" % (os.getuid(), os.getgid()), str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode == 0 and os.access(path, os.W_OK):
        return True
    log.error(
        "Could not make %s writable (rc=%s): %s", path, result.returncode, result.stderr.strip()
    )
    return False


def collect_interpreters(all_pythons, preferred_python):
    """Return a list of Python interpreter paths to build wheels for."""
    if not all_pythons:
        return [preferred_python]
    found = []
    for candidate in MANYLINUX_PYTHONS:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            found.append(candidate)
        else:
            log.debug("Skipping missing interpreter %s", candidate)
    if not found:
        log.warning("No /opt/python interpreters found; falling back to %s", preferred_python)
        return [preferred_python]
    return found


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Build AMDSMI Python wheels (Debian / Ubuntu / manylinux).",
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
        default=None,
        help="CMake build directory  [default: /tmp/amdsmi-build].",
    )
    p.add_argument(
        "--raw-wheels-dir",
        type=Path,
        default=None,
        help="Staging dir for raw wheels  [default: /tmp/raw-wheels].",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final wheel output dir  [default: <project-dir>/wheels].",
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
        "--python-bin", default="python3", help="Python interpreter for cmake  [default: python3]."
    )
    p.add_argument(
        "--all-pythons",
        action="store_true",
        help="Build for all /opt/python interpreters (manylinux).",
    )
    p.add_argument(
        "--repair", action="store_true", help="Run auditwheel repair for manylinux tags."
    )
    p.add_argument("--build-tests", action="store_true", help="Build C/C++ tests  [default: OFF].")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- project dir & git safety ----------------------------------------
    if args.project_dir is None:
        args.project_dir = find_project_dir(Path(__file__).resolve().parent.parent)
    args.project_dir = args.project_dir.resolve()

    git_safe_paths = []
    if not mark_safe_git_dir(args.project_dir):
        git_safe_paths.append(args.project_dir)

    esmi_repo = args.project_dir / "esmi_ib_library"
    if esmi_repo.exists() and not mark_safe_git_dir(esmi_repo):
        git_safe_paths.append(esmi_repo)

    esmi_temp = args.project_dir / "esmi_ib_library_temp"
    if esmi_temp.exists():
        log.info("Removing stale esmi_ib_library_temp at %s", esmi_temp)
        shutil.rmtree(esmi_temp)

    if esmi_repo.exists() and not ensure_writable_dir(esmi_repo, "esmi_ib_library"):
        abort(str(esmi_repo) + " is not writable; fix permissions or rerun with sudo.")

    # --- directories -----------------------------------------------------
    if args.build_dir is None:
        args.build_dir = DEFAULT_BUILD_DIR
    args.build_dir = args.build_dir.resolve()

    raw_wheels_dir = (
        args.raw_wheels_dir.resolve() if args.raw_wheels_dir else DEFAULT_RAW_WHEELS_DIR
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir else args.project_dir / DEFAULT_OUTPUT_NAME
    )
    pkg_dir = args.build_dir / "py-interface" / "python_package"

    # --- interpreters ----------------------------------------------------
    python_bin = shutil.which(args.python_bin) or args.python_bin
    python_interpreters = collect_interpreters(args.all_pythons, python_bin)
    cmake_python = python_interpreters[0]

    if args.all_pythons and not args.repair:
        log.info("Enabling --repair because --all-pythons was requested.")
        args.repair = True

    log.info("Project dir  : %s", args.project_dir)
    log.info("Build dir    : %s", args.build_dir)
    log.info("Raw wheels   : %s", raw_wheels_dir)
    log.info("Output dir   : %s", output_dir)
    log.info("Python (cmake): %s", cmake_python)
    log.info("Python targets: %s", ", ".join(python_interpreters))

    # --- prepare build directory -----------------------------------------
    if args.build_dir == DEFAULT_BUILD_DIR:
        if args.build_dir.exists():
            log.info("Removing existing default build dir %s", args.build_dir)
            shutil.rmtree(args.build_dir)
        args.build_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.build_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(args.build_dir, os.W_OK):
            log.warning("Build dir not writable; reclaiming with sudo chown ...")
            subprocess.run(
                ["sudo", "chown", "-R", "%d:%d" % (os.getuid(), os.getgid()), str(args.build_dir)],
                check=True,
            )
        for stale in ("CMakeCache.txt", "CMakeFiles"):
            p = args.build_dir / stale
            if p.exists():
                log.info("Removing stale %s", p)
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    git_env = {}
    if git_safe_paths:
        git_env = write_temp_git_config(args.build_dir / "git-safe.config", git_safe_paths)

    # --- prepare wheel directories ---------------------------------------
    if raw_wheels_dir.exists():
        log.info("Removing existing raw wheels dir %s", raw_wheels_dir)
        shutil.rmtree(raw_wheels_dir)
    raw_wheels_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.whl"):
        log.info("Removing existing wheel %s", existing)
        existing.unlink()

    # --- cmake configure & build -----------------------------------------
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
        env=git_env or None,
    )

    run(["make", "-j" + str(os.cpu_count() or 4)], cwd=str(args.build_dir), env=git_env or None)

    if not pkg_dir.exists():
        abort("Python package directory not found: " + str(pkg_dir))

    # --- build wheels ----------------------------------------------------
    best_effort_pip_upgrade(
        cmake_python, ["pip", "setuptools", "wheel"] + (["auditwheel"] if args.repair else [])
    )

    log.info("Building wheels for %d interpreter(s) ...", len(python_interpreters))

    for py in python_interpreters:
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
