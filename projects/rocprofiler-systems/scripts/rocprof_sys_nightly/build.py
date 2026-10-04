# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure and build examples or a full rocprofiler-systems tree."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import die, log, run, step
from .constants import TRACE_PROCESSOR_SHELL_SHA256, TRACE_PROCESSOR_SHELL_URL
from .tarball import _require_https, _sha256_file


def mpi_available(workdir: Path, env: dict, extra_cmake_args: list[str] | None) -> bool:
    """Probe whether ``find_package(MPI)`` succeeds with this toolchain."""
    probe = workdir / ".mpi-probe"
    # never reuse the cache: stale results would ignore newly passed MPI hints
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(rocprofsys_mpi_probe LANGUAGES C CXX)\n"
        "find_package(MPI)\n"
        "if(NOT (MPI_C_FOUND AND MPI_CXX_FOUND))\n"
        '    message(FATAL_ERROR "MPI_C/MPI_CXX not found")\n'
        "endif()\n",
        encoding="utf-8",
    )
    cmd = [
        shutil.which("cmake", path=env.get("PATH")) or "cmake",
        "-S",
        str(probe),
        "-B",
        str(probe / "build"),
    ] + (extra_cmake_args or [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def detect_mpi(workdir: Path, env: dict, cmake_args: list[str], facts: dict) -> bool:
    """Probe MPI, record the outcome, and explain it (see ``mpi_available``).

    Called immediately before a CMake configure rather than once up front, so a run
    that reuses up-to-date examples does not pay for a probe it cannot act on.
    """
    step("Detect MPI support")
    use_mpi = mpi_available(workdir, env, cmake_args)
    facts["mpi"] = "available" if use_mpi else "unavailable (find_package(MPI) failed)"
    if use_mpi:
        log("find_package(MPI) succeeded; MPI is enabled for the build below.")
        log("MPI tests stay excluded until you pass --run-labels mpi.")
    else:
        log(
            "find_package(MPI) failed; MPI is disabled, so MPI tests would skip as "
            "'binary not found'. For a vendor MPI with no mpicc wrapper, pass hints "
            "via --cmake-arg (see --help)."
        )
    return use_mpi


def build_from_source(
    src_dir: Path,
    workdir: Path,
    rocm_dir: Path,
    env: dict,
    venv_py: Path,
    jobs: int,
    use_mpi: bool,
    disable_examples: list[str] | None = None,
    extra_cmake_args: list[str] | None = None,
) -> Path:
    """Configure + build the full rocprofiler-systems project from source.

    Builds the rocprof-sys binaries, examples and test suite in-tree (like CI/QA),
    using the ROCm runtime + toolchain from the extracted tarball. Returns the
    build directory (its binaries are validated in pytest 'build mode').
    """
    build_dir = workdir / "build-from-source"
    cmake = shutil.which("cmake", path=env.get("PATH")) or "cmake"

    # The in-build pytest CTest-generation step needs pytest + python from our venv,
    # so put the venv on PATH and point CMake's Python discovery at it.
    build_env = dict(env)
    build_env["PATH"] = os.pathsep.join(
        [str(venv_py.parent), build_env.get("PATH", "")]
    ).rstrip(os.pathsep)

    cfg = [
        cmake,
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={workdir / 'install-from-source'}",
        f"-DPython3_EXECUTABLE={venv_py}",
        "-DROCPROFSYS_BUILD_DYNINST=ON",
        "-DROCPROFSYS_BUILD_TBB=ON",
        "-DROCPROFSYS_BUILD_ELFUTILS=ON",
        "-DROCPROFSYS_BUILD_LIBIBERTY=ON",
        "-DROCPROFSYS_BUILD_CI=ON",
        "-DROCPROFSYS_BUILD_EXAMPLES=ON",
        "-DROCPROFSYS_BUILD_TESTING=ON",
        "-DROCPROFSYS_USE_PYTHON=ON",
        f"-DROCPROFSYS_USE_MPI={'ON' if use_mpi else 'OFF'}",
    ]
    if disable_examples:
        cfg.append("-DROCPROFSYS_DISABLE_EXAMPLES=" + ";".join(disable_examples))
    if extra_cmake_args:
        cfg += extra_cmake_args
    log("Configuring full source build (this pulls/builds dyninst, elfutils, etc.)")
    run(cfg, env=build_env)
    log(f"Building rocprofiler-systems from source with {jobs} jobs (can take ~1-2h)")
    run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], env=build_env)
    return build_dir


def build_examples(
    src_dir: Path,
    workdir: Path,
    rocm_dir: Path,
    env: dict,
    jobs: int,
    use_mpi: bool,
    disable_examples: list[str],
    extra_cmake_args: list[str] | None = None,
) -> None:
    """Configure/build the standalone examples and install into the ROCm prefix."""
    build_dir = workdir / "build-examples"
    cmake = shutil.which("cmake", path=env.get("PATH")) or "cmake"
    cfg = [
        cmake,
        "-S",
        str(src_dir / "examples"),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={rocm_dir}",
        "-DROCPROFSYS_INSTALL_EXAMPLES=ON",
        f"-DROCPROFSYS_USE_MPI={'ON' if use_mpi else 'OFF'}",
    ]
    if disable_examples:
        # semicolon-separated CMake list; skips add_subdirectory() for these
        cfg.append("-DROCPROFSYS_DISABLE_EXAMPLES=" + ";".join(disable_examples))
        log("Skipping examples: " + ", ".join(disable_examples))
    if extra_cmake_args:
        cfg += extra_cmake_args
    run(cfg, env=env)
    run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], env=env)
    run([cmake, "--install", str(build_dir)], env=env)

    examples_out = rocm_dir / "share" / "rocprofiler-systems" / "examples"
    n = len(list(examples_out.glob("*"))) if examples_out.is_dir() else 0
    log(f"Installed {n} example artifact(s) into {examples_out}")


def _download_binary(
    url: str,
    dest: Path,
    *,
    expected_sha256: str,
    skip_download: bool,
) -> bool:
    """Download a pinned binary into ``dest``, or reuse a verified cached copy.

    Returns True when ``dest`` is present and executable afterward.
    """
    _require_https(url)
    if dest.is_file() and os.access(dest, os.X_OK):
        if _sha256_file(dest).lower() == expected_sha256.lower():
            return True
        if skip_download:
            log(
                f"WARNING: cached {dest.name} checksum mismatch; "
                "Perfetto validation may fail offline."
            )
            return True
        log(f"Removing invalid cached binary: {dest.name}")
        dest.unlink(missing_ok=True)

    if skip_download:
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    if wget:
        cmd = [wget, "--tries=3", "-q", "-O", str(tmp), url]
    elif curl:
        cmd = [curl, "-fL", "--retry", "3", "-o", str(tmp), url]
    else:
        log(
            "WARNING: neither wget nor curl available; "
            "cannot download trace_processor_shell."
        )
        return False

    log(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode  # noqa: S603
    if rc != 0:
        tmp.unlink(missing_ok=True)
        log(f"WARNING: download failed (exit {rc}): {url}")
        return False

    actual = _sha256_file(tmp)
    if actual.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        die(
            "trace_processor_shell SHA-256 mismatch:\n"
            f"       expected: {expected_sha256}\n"
            f"       actual:   {actual}"
        )
    tmp.rename(dest)
    dest.chmod(0o755)
    return True


def ensure_trace_processor_shell(workdir: Path, skip_download: bool) -> Path | None:
    """Cache the pinned perfetto trace_processor_shell under ``workdir``.

    Mirrors the binary staged by tests/CMakeLists.txt so Perfetto validation works
    on air-gapped compute nodes after a networked ``--prepare-only`` run.
    """
    if platform.machine() not in ("x86_64", "AMD64"):
        log(
            "WARNING: trace_processor_shell is built for x86-64 only; "
            f"skipping on {platform.machine()}."
        )
        return None

    dest = workdir / "trace_processor_shell"
    if _download_binary(
        TRACE_PROCESSOR_SHELL_URL,
        dest,
        expected_sha256=TRACE_PROCESSOR_SHELL_SHA256,
        skip_download=skip_download,
    ):
        log(f"trace_processor_shell ready: {dest}")
        return dest

    if skip_download:
        log(
            "WARNING: trace_processor_shell is not cached; Perfetto validation "
            "may fail offline."
        )
    return None


def stage_tests(
    src_dir: Path,
    rocm_dir: Path,
    workdir: Path,
    env: dict,
    *,
    skip_download: bool = False,
) -> Path:
    """Copy the pytest suite + helpers into the ROCm prefix (install-mode layout).

    Mirrors tests/CMakeLists.txt + tests/pytest/CMakeLists.txt copy rules.
    Returns the installed pytest directory.
    """
    tests_src = src_dir / "tests"
    tests_dst = rocm_dir / "share" / "rocprofiler-systems" / "tests"
    pytest_dst = tests_dst / "pytest"
    (pytest_dst / "rocprofsys").mkdir(parents=True, exist_ok=True)

    # pytest package + conftest + test_*.py
    pytest_src = tests_src / "pytest"
    for item in pytest_src.glob("test_*.py"):
        shutil.copy2(item, pytest_dst / item.name)
    shutil.copy2(pytest_src / "conftest.py", pytest_dst / "conftest.py")
    for item in (pytest_src / "rocprofsys").glob("*.py"):
        shutil.copy2(item, pytest_dst / "rocprofsys" / item.name)

    # top-level test helpers / validators / scripts
    top_level_files = [
        "check_amd_smi_metrics.py",
        "validate-causal-json.py",
        "validate-perfetto-proto.py",
        "validate-rocpd.py",
        "validate-timemory-json.py",
        "validate-unified-memory.py",
        "get_default_nic.sh",
        "generate_papi_nic_events.sh",
        "run_rocprofiler_systems.py",
        "test_categories.yaml",
        "README.md",
        "run_if_shmem_ok.sh",
        "shmem_validation_check.sh",
    ]
    for name in top_level_files:
        srcf = tests_src / name
        if srcf.is_file():
            shutil.copy2(srcf, tests_dst / name)

    # requirements.txt (some helpers reference it)
    req = src_dir / "requirements.txt"
    if req.is_file():
        shutil.copy2(req, tests_dst / "requirements.txt")

    # rocpd validation rule directory
    rules_src = tests_src / "rocpd-validation-rules"
    if rules_src.is_dir():
        rules_dst = tests_dst / "rocpd-validation-rules"
        if rules_dst.exists():
            shutil.rmtree(rules_dst)
        shutil.copytree(rules_src, rules_dst)

    # capability-check helper (standalone C++; needed by several tests)
    _build_capchk(tests_src, tests_dst, env)

    shell = ensure_trace_processor_shell(workdir, skip_download)
    if shell is not None:
        staged_shell = tests_dst / "trace_processor_shell"
        shutil.copy2(shell, staged_shell)
        staged_shell.chmod(0o755)
        log(f"Installed trace_processor_shell into {tests_dst}")

    log(f"Staged test suite into {tests_dst}")
    return pytest_dst


def _build_capchk(tests_src: Path, tests_dst: Path, env: dict) -> None:
    capchk_cpp = tests_src / "rocprof-sys-capchk.cpp"
    if not capchk_cpp.is_file():
        return
    cxx = (
        env.get("CXX")
        or shutil.which("amdclang++", path=env.get("PATH"))
        or shutil.which("g++")
        or shutil.which("c++")
    )
    if not cxx:
        log("WARNING: no C++ compiler found; skipping rocprof-sys-capchk build.")
        return
    out = tests_dst / "rocprof-sys-capchk"
    run(
        [cxx, "-std=c++17", "-O2", str(capchk_cpp), "-o", str(out)],
        env=env,
        check=False,
    )
