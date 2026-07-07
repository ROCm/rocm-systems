#!/usr/bin/env python3
"""
build_wheel.py
==============

Build AMDSMI Python wheels on any supported host (Debian/Ubuntu, RHEL,
AlmaLinux, SLES, AzureLinux) or inside a ``quay.io/pypa/manylinux_2_28``
container. Replaces the previous ``build_wheel_debian.py`` /
``build_wheel_rpm.py`` pair -- distro-level prerequisite installation is
done in the calling workflow / by the operator before invoking this
script.

Pipeline
--------
    1. (optional) Register the project as a git safe.directory
    2. cmake configure  (-DBUILD_PYTHON_WHEEL=ON)
    3. make -j$(nproc)
    4. pip wheel --no-deps --no-build-isolation  (per interpreter)
    5. auditwheel repair  (optional, for manylinux tags)

Requirements
------------
* cmake, make, gcc/g++, git, python3, pip already installed
* (optional) auditwheel -- pip-installed automatically when --repair is set
* (optional) /opt/python/cpXX-cpXX interpreters -- present in manylinux images

Examples
--------
Single-interpreter local build::

    python3 build_wheel.py --project-dir /path/to/amdsmi

Multi-interpreter manylinux build inside the manylinux_2_28 container::

    /opt/python/cp38-cp38/bin/python3 build_wheel.py \
        --project-dir /src/amdsmi \
        --output-dir /src/amdsmi/wheels \
        --os-variant AlmaLinux8 \
        --repair

Compatibility
-------------
* Python >= 3.6
* Tested on Ubuntu 20.04/22.04/24.04, RHEL 8/9/10, AlmaLinux 8, SLES 15,
  AzureLinux 3, and manylinux_2_28 containers.
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
# Shared helpers
# ---------------------------------------------------------------------------
# These helpers exist (rather than being inlined into the workflow YAML) so a
# developer can reproduce a CI wheel-build failure locally with the exact
# command CI runs -- for example inside a ``quay.io/pypa/manylinux_2_28_x86_64``
# container. The two whose semantics do not survive translation to shell are:
#
# * ``best_effort_pip_upgrade`` -- warns on failure rather than silently
#   swallowing all errors the way ``pip install ... || true`` would.
# * ``mark_safe_git_dir`` / ``write_temp_git_config`` -- falls back to a
#   per-build ``GIT_CONFIG_GLOBAL`` when the global gitconfig is not writable
#   (read-only ``HOME``, ``nobody`` user, etc.).
#
# Python 3.6-safe (uses ``universal_newlines``, not ``text``).


def run(cmd, cwd=None, env=None, check=True):
    """Execute *cmd* and stream output to the console."""
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, check=check, cwd=cwd, env=merged_env)


def abort(message, code=1):
    log.error(message)
    sys.exit(code)


def best_effort_pip_upgrade(py, packages):
    """Try to upgrade *packages* via pip; log a warning on failure."""
    result = subprocess.run(
        [py, "-m", "pip", "install", "--upgrade"] + list(packages),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        log.warning(
            "pip upgrade failed with %s (packages: %s)\nstdout: %s\nstderr: %s",
            py,
            ", ".join(packages),
            result.stdout.strip(),
            result.stderr.strip(),
        )
    return result.returncode == 0


def mark_safe_git_dir(path):
    """Register *path* as a safe git directory (avoids dubious-ownership errors)."""
    git_bin = shutil.which("git")
    if not git_bin or not Path(path).exists():
        return True
    result = subprocess.run(
        [git_bin, "config", "--global", "--add", "safe.directory", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        log.warning(
            "git safe.directory add failed for %s (rc=%s): %s",
            path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def write_temp_git_config(config_path, safe_paths):
    """Write a temporary gitconfig that marks *safe_paths* as safe directories."""
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for p in safe_paths:
            lines.append("[safe]\n\tdirectory = %s\n" % p)
        Path(config_path).write_text("\n".join(lines))
        log.info("Temporary git config for safe.directory: %s", config_path)
        return {"GIT_CONFIG_GLOBAL": str(config_path)}
    except (OSError, PermissionError) as exc:
        log.warning("Failed to create temporary git config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OS_VARIANTS = (
    "RHEL8",
    "RHEL9",
    "RHEL10",
    "AlmaLinux8",
    "SLES",
    "AzureLinux3",
    "Ubuntu20",
    "Ubuntu22",
    "Ubuntu24",
    "Debian10",
    "Debian11",
    "Debian12",
)

# Variants that need QA_RPATHS to silence rpath-related QA failures.
QA_RPATHS_VARIANTS = ("RHEL10", "AlmaLinux8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_interpreters(python_bins_csv):
    """Return the list of Python interpreters to build wheels for.

    Priority:
      1. ``--python-bins`` (comma-separated), if given
      2. ``/opt/python/cp3*-cp3*/bin/python3`` (manylinux image)
      3. The system ``python3``
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


def prepare_build_dir(build_dir, clean):
    """Create *build_dir*; if *clean* is true, wipe it; otherwise drop stale CMake state only."""
    if clean and build_dir.exists():
        log.info("Removing build dir %s (--clean)", build_dir)
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(build_dir, os.W_OK):
        abort(str(build_dir) + " is not writable; fix permissions and retry.")
    for stale in ("CMakeCache.txt", "CMakeFiles"):
        p = build_dir / stale
        if p.exists():
            log.info("Removing stale %s", p)
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def prepare_wheel_dirs(raw_wheels_dir, output_dir):
    """Reset *raw_wheels_dir* and clear any pre-existing wheels in *output_dir*."""
    if raw_wheels_dir.exists():
        log.info("Removing existing raw wheels dir %s", raw_wheels_dir)
        shutil.rmtree(raw_wheels_dir)
    raw_wheels_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.whl"):
        log.info("Removing existing wheel %s", existing)
        existing.unlink()


def configure_git_safety(project_dir, build_dir):
    """Mark *project_dir* (and its bundled esmi tree) as git-safe.

    Returns an environment dict suitable for passing to subsequent ``run()``
    calls. Empty if the global gitconfig accepted the entries directly.
    """
    safe_paths = []
    if not mark_safe_git_dir(project_dir):
        safe_paths.append(project_dir)

    esmi_repo = project_dir / "esmi_ib_library"
    if esmi_repo.exists() and not mark_safe_git_dir(esmi_repo):
        safe_paths.append(esmi_repo)

    if not safe_paths:
        return {}
    return write_temp_git_config(build_dir / "git-safe.config", safe_paths)


def cmake_and_build(project_dir, build_dir, cmake_python, args, extra_env):
    """Run cmake configure + make."""
    run(
        [
            "cmake",
            str(project_dir),
            "-DBUILD_TESTS=" + ("ON" if args.build_tests else "OFF"),
            "-DENABLE_ESMI_LIB=" + ("ON" if args.enable_esmi else "OFF"),
            "-DBUILD_PYTHON_WHEEL=ON",
            "-DCMAKE_BUILD_TYPE=" + args.build_type,
            "-DPython3_EXECUTABLE=" + cmake_python,
        ],
        cwd=str(build_dir),
        env=extra_env or None,
    )
    run(["make", "-j" + str(os.cpu_count() or 4)], cwd=str(build_dir), env=extra_env or None)


def build_wheels(pkg_dir, raw_wheels_dir, interpreters):
    """Run ``pip wheel`` for each interpreter into *raw_wheels_dir*."""
    log.info("Building wheels for %d interpreter(s) ...", len(interpreters))
    for py in interpreters:
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


def repair_or_copy(raw_wheels, output_dir, cmake_python, repair):
    """Run ``auditwheel repair`` if requested; otherwise just copy raw wheels."""
    if not repair:
        for whl in raw_wheels:
            shutil.copy2(whl, output_dir)
        return

    auditwheel_ok = (
        subprocess.run(
            [cmake_python, "-m", "auditwheel", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).returncode
        == 0
    )

    if not auditwheel_ok:
        log.warning("auditwheel not available; copying raw wheels.")
        for whl in raw_wheels:
            shutil.copy2(whl, output_dir)
        return

    log.info("Repairing wheels with auditwheel ...")
    for whl in raw_wheels:
        result = run(
            [cmake_python, "-m", "auditwheel", "repair", str(whl), "--wheel-dir", str(output_dir)],
            check=False,
        )
        if result.returncode != 0:
            log.warning("auditwheel repair failed for %s; copying raw wheel.", whl.name)
            shutil.copy2(whl, output_dir)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Build AMDSMI Python wheels (Linux / manylinux).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="AMDSMI project root (the directory with CMakeLists.txt).",
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
        dest="enable_esmi",
        action="store_true",
        default=True,
        help="Enable ESMI library build (default).",
    )
    p.add_argument(
        "--no-esmi", dest="enable_esmi", action="store_false", help="Disable ESMI library build."
    )
    p.add_argument(
        "--python-bins",
        default=None,
        help="Comma-separated Python executables  "
        "(default: /opt/python/cp3*/bin/python3 if present, "
        "else system python3).",
    )
    p.add_argument(
        "--os-variant",
        choices=list(OS_VARIANTS),
        default=None,
        help="Target OS variant (only affects QA_RPATHS handling).",
    )
    p.add_argument(
        "--repair",
        dest="repair",
        action="store_true",
        default=False,
        help="Run auditwheel repair for manylinux tags.",
    )
    p.add_argument(
        "--no-repair", dest="repair", action="store_false", help="Skip the auditwheel repair step."
    )
    p.add_argument("--build-tests", action="store_true", help="Build C/C++ tests  [default: OFF].")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Wipe --build-dir entirely before configuring  "
        "[default: surgical clean of CMakeCache.txt only].",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    project_dir = args.project_dir.resolve()
    if not (project_dir / "CMakeLists.txt").exists():
        abort("No CMakeLists.txt in --project-dir: " + str(project_dir))

    build_dir = args.build_dir.resolve()
    raw_wheels_dir = args.raw_wheels_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else project_dir / "wheels"

    interpreters = collect_interpreters(args.python_bins)
    if not interpreters:
        abort("No Python interpreters found.  Pass --python-bins explicitly.")
    cmake_python = interpreters[0]

    log.info("OS variant    : %s", args.os_variant or "(unspecified)")
    log.info("Project dir   : %s", project_dir)
    log.info("Build dir     : %s", build_dir)
    log.info("Raw wheels    : %s", raw_wheels_dir)
    log.info("Output dir    : %s", output_dir)
    log.info("Python (cmake): %s", cmake_python)
    log.info("Python targets: %s", ", ".join(interpreters))

    # esmi_ib_library_temp is generated by a stale build; remove it before
    # reconfiguring so cmake won't pick it up.
    esmi_temp = project_dir / "esmi_ib_library_temp"
    if esmi_temp.exists():
        log.info("Removing stale esmi_ib_library_temp at %s", esmi_temp)
        shutil.rmtree(esmi_temp)

    prepare_build_dir(build_dir, args.clean)

    # Configure git safety *after* the build dir exists (the per-build gitconfig
    # fallback writes inside build_dir).
    extra_env = configure_git_safety(project_dir, build_dir)

    if args.os_variant in QA_RPATHS_VARIANTS:
        qa_rpaths = hex(0x0010 | 0x0002)
        extra_env["QA_RPATHS"] = qa_rpaths
        log.info("Setting QA_RPATHS=%s for %s", qa_rpaths, args.os_variant)

    cmake_and_build(project_dir, build_dir, cmake_python, args, extra_env)

    pkg_dir = build_dir / "py-interface" / "python_package"
    if not pkg_dir.exists():
        abort("Python package directory not found: " + str(pkg_dir))

    prepare_wheel_dirs(raw_wheels_dir, output_dir)

    best_effort_pip_upgrade(
        cmake_python, ["pip", "setuptools", "wheel"] + (["auditwheel"] if args.repair else [])
    )

    build_wheels(pkg_dir, raw_wheels_dir, interpreters)

    raw_wheels = sorted(raw_wheels_dir.glob("*.whl"))
    if not raw_wheels:
        abort("No wheels found in " + str(raw_wheels_dir))

    log.info("=== Raw wheel(s) built ===")
    for w in raw_wheels:
        log.info("  %s  (%d KB)", w.name, w.stat().st_size // 1024)

    repair_or_copy(raw_wheels, output_dir, cmake_python, args.repair)

    final_wheels = sorted(output_dir.glob("*.whl"))
    log.info("=== Final wheel(s) ===")
    for w in final_wheels:
        log.info("  %s  (%d KB)", w.name, w.stat().st_size // 1024)
    log.info("Done.  Wheels written to %s", output_dir)


if __name__ == "__main__":
    main()
