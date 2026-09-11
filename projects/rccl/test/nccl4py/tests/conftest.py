# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Shared fixtures for the nccl4py build-smoke pytest harness.

This validates the **uv / CMake build path** that NCCL 2.29.7 shipped as
"Added CMake for NCCL4Py build" -- i.e. the ``BUILD_NCCL4PY`` option and the
``nccl4py`` CMake target (backed by ``uv build``), rather than a bare
``pip install``. It mirrors test/ir-device: the target is driven on demand
from a session-scoped fixture and the whole suite is skipped with a clear
reason when prerequisites are missing.

Flow:

  1. ``cmake -DBUILD_NCCL4PY=ON <RCCL_BUILD>`` -- (re)configure the existing
     RCCL build tree so the nccl4py CMake target is available (exercises the
     root-CMakeLists BUILD_NCCL4PY wiring).
  2. ``cmake --build <RCCL_BUILD> --target nccl4py`` -- build the wheel via
     ``uv build`` into ``<RCCL_BUILD>/dist``.
  3. ``uv venv`` + ``uv pip install <wheel> pytest`` -- stand up an isolated
     Python 3.13 environment matching the wheel's ABI.
  4. Run the CPU/GPU smoke modules under ``bindings/nccl4py/tests`` with that
     environment's interpreter.

Prerequisites (environment variables, with sensible defaults):

  RCCL_DIR      RCCL source root         (default: repo root, derived)
  RCCL_BUILD    RCCL CMake build dir     (default: $RCCL_DIR/build/release)
  ROCM_PATH     ROCm install root        (default: /opt/rocm)
  NCCL4PY_DIR   nccl4py source tree      (default: $RCCL_DIR/bindings/nccl4py)
  NCCL4PY_VENV  uv venv for the wheel    (default: <workdir>/nccl4py_uvenv)
  UV_PYTHON     Python for the uv venv   (default: 3.13, matches the target)
  CMAKE         cmake executable         (default: cmake)
  UV            uv executable            (default: uv)

RCCL must have been built (and configured) at least once so ``librccl.so`` is
discoverable and ``$RCCL_BUILD`` is a usable CMake build tree.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

logger = logging.getLogger(__name__)

WORKDIR = os.getcwd()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RCCL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

RCCL_DIR = os.path.abspath(os.environ.get("RCCL_DIR", _DEFAULT_RCCL_DIR))
RCCL_BUILD = os.path.abspath(
    os.environ.get("RCCL_BUILD", os.path.join(RCCL_DIR, "build", "release"))
)
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
NCCL4PY_DIR = os.path.abspath(
    os.environ.get("NCCL4PY_DIR", os.path.join(RCCL_DIR, "bindings", "nccl4py"))
)
NCCL4PY_VENV = os.path.abspath(
    os.environ.get("NCCL4PY_VENV", os.path.join(WORKDIR, "nccl4py_uvenv"))
)
UV_PYTHON = os.environ.get("UV_PYTHON", "3.13")
CMAKE = os.environ.get("CMAKE", "cmake")
UV = os.environ.get("UV", "uv")

DIST_DIR = os.path.join(RCCL_BUILD, "dist")

LOGDIR = os.path.join(WORKDIR, "logs")
os.makedirs(LOGDIR, exist_ok=True)

# CPU smoke module: accepts NotImplementedError or NCCLError depending on
# which RCCL symbols are present in the loaded librccl.so.
CPU_SMOKE_TESTS = ("tests/test_rocm_extensions.py",)

# Optional GPU-backed shim surface; the module self-skips without HIP devices.
GPU_SMOKE_TESTS = ("tests/test_shim_surface.py",)


def _librccl_candidates():
    yield os.path.join(RCCL_BUILD, "librccl.so")
    yield os.path.join(ROCM_PATH, "lib", "librccl.so")
    yield os.path.join(ROCM_PATH, "lib64", "librccl.so")


def _find_librccl() -> str | None:
    explicit = os.environ.get("NCCL_LIBRARY")
    if explicit and os.path.isfile(explicit):
        return explicit
    for path in _librccl_candidates():
        if os.path.isfile(path):
            return path
    return None


def _missing_prerequisite() -> str | None:
    if not os.path.isfile(os.path.join(NCCL4PY_DIR, "pyproject.toml")):
        return f"nccl4py source not found at {NCCL4PY_DIR}"
    if not os.path.isfile(os.path.join(RCCL_BUILD, "CMakeCache.txt")):
        return (
            f"RCCL build tree not configured at RCCL_BUILD={RCCL_BUILD} "
            f"(configure/build RCCL once so the nccl4py CMake target is reachable)"
        )
    if shutil.which(CMAKE) is None:
        return f"cmake ('{CMAKE}') not found on PATH"
    if shutil.which(UV) is None:
        return (
            f"uv ('{UV}') not found on PATH -- the CMake nccl4py target is "
            f"uv-driven (https://docs.astral.sh/uv/getting-started/installation/)"
        )
    if _find_librccl() is None:
        return (
            f"librccl.so not found under RCCL_BUILD={RCCL_BUILD} or "
            f"ROCM_PATH={ROCM_PATH} (build RCCL once or set NCCL_LIBRARY)"
        )
    return None


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    librccl = _find_librccl()
    lib_dirs = []
    if librccl:
        env["NCCL_LIBRARY"] = librccl
        lib_dirs.append(os.path.dirname(librccl))
    for d in (RCCL_BUILD, os.path.join(ROCM_PATH, "lib"), os.path.join(ROCM_PATH, "lib64")):
        if os.path.isdir(d) and d not in lib_dirs:
            lib_dirs.append(d)
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


def _run_logged(args: list[str], log_path: str, cwd: str, header: str = "") -> subprocess.CompletedProcess:
    with open(log_path, "a") as log:
        if header:
            log.write(header + "\n")
        log.write("$ " + " ".join(args) + f"  (cwd={cwd})\n\n")
        log.flush()
        return subprocess.run(
            args,
            cwd=cwd,
            env=_runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )


def _build_nccl4py_already_enabled() -> bool:
    """True if the existing CMake cache already has BUILD_NCCL4PY=ON."""
    cache = os.path.join(RCCL_BUILD, "CMakeCache.txt")
    try:
        with open(cache) as fh:
            for line in fh:
                if line.startswith("BUILD_NCCL4PY:") and line.rstrip().endswith("=ON"):
                    return True
    except OSError:
        pass
    return False


def _cmake_build_wheel() -> str:
    """Drive the uv/CMake nccl4py target; return the built wheel path."""
    build_log = os.path.join(LOGDIR, "nccl4py_cmake_build.log")
    open(build_log, "w").close()

    # Only (re)configure when BUILD_NCCL4PY isn't already enabled. Re-running
    # configure regenerates device sources (updating mtimes) and would trigger
    # an expensive RCCL rebuild via the nccl4py -> rccl dependency, so we avoid
    # it when the option is already set in the cache.
    if not _build_nccl4py_already_enabled():
        proc = _run_logged(
            [CMAKE, "-DBUILD_NCCL4PY=ON", RCCL_BUILD],
            build_log,
            cwd=RCCL_DIR,
            header="### configure: enable BUILD_NCCL4PY",
        )
        assert proc.returncode == 0, (
            f"cmake configure with BUILD_NCCL4PY=ON failed (see {build_log})"
        )
    else:
        with open(build_log, "a") as log:
            log.write("### configure: BUILD_NCCL4PY already ON in cache, skipping\n\n")

    # Build the wheel via the uv-backed custom target.
    proc = _run_logged(
        [CMAKE, "--build", RCCL_BUILD, "--target", "nccl4py"],
        build_log,
        cwd=RCCL_DIR,
        header="\n### build: cmake --build --target nccl4py",
    )
    assert proc.returncode == 0, f"cmake --build --target nccl4py failed (see {build_log})"

    wheels = []
    if os.path.isdir(DIST_DIR):
        wheels = sorted(
            os.path.join(DIST_DIR, f) for f in os.listdir(DIST_DIR) if f.endswith(".whl")
        )
    assert wheels, f"nccl4py target produced no wheel in {DIST_DIR} (see {build_log})"
    return wheels[-1]


def _uv_env_python(wheel: str) -> str:
    """Create a uv venv matching the wheel ABI and install it; return python path."""
    env_log = os.path.join(LOGDIR, "nccl4py_uv_env.log")
    open(env_log, "w").close()

    proc = _run_logged(
        [UV, "venv", "--python", UV_PYTHON, NCCL4PY_VENV],
        env_log,
        cwd=WORKDIR,
        header="### uv venv",
    )
    assert proc.returncode == 0, f"uv venv creation failed (see {env_log})"

    venv_python = os.path.join(NCCL4PY_VENV, "bin", "python")
    proc = _run_logged(
        [UV, "pip", "install", "--python", venv_python, wheel, "pytest"],
        env_log,
        cwd=WORKDIR,
        header="\n### uv pip install <wheel> pytest",
    )
    assert proc.returncode == 0, f"uv pip install of nccl4py wheel failed (see {env_log})"
    return venv_python


@pytest.fixture(scope="session")
def nccl4py_wheel():
    """Build the nccl4py wheel via the uv/CMake target once per session."""
    reason = _missing_prerequisite()
    if reason:
        pytest.skip(f"nccl4py build-smoke skipped: {reason}")

    logger.info("cmake --build --target nccl4py (uv/CMake path)...")
    wheel = _cmake_build_wheel()
    logger.info("nccl4py wheel built: %s", wheel)
    return wheel


@pytest.fixture(scope="session")
def nccl4py_env(nccl4py_wheel):
    """Provision a uv venv with the built wheel; return its python interpreter."""
    venv_python = _uv_env_python(nccl4py_wheel)
    logger.info("nccl4py uv env ready: %s", venv_python)
    return venv_python


@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        WORKDIR=WORKDIR,
        RCCL_DIR=RCCL_DIR,
        RCCL_BUILD=RCCL_BUILD,
        ROCM_PATH=ROCM_PATH,
        NCCL4PY_DIR=NCCL4PY_DIR,
        NCCL4PY_VENV=NCCL4PY_VENV,
        DIST_DIR=DIST_DIR,
        LOGDIR=LOGDIR,
        LIBRCCL=_find_librccl(),
    )


@pytest.fixture(scope="session")
def run_nccl4py_pytest(nccl4py_env):
    """Run a pytest module under bindings/nccl4py/tests using the uv env python."""

    def _run(relative_test_path: str, log_name: str) -> tuple[subprocess.CompletedProcess, str]:
        target = os.path.join(NCCL4PY_DIR, relative_test_path)
        if not os.path.isfile(target):
            raise FileNotFoundError(target)
        args = [
            nccl4py_env,
            "-m",
            "pytest",
            relative_test_path,
            "-q",
            "--tb=short",
            "--color=no",
        ]
        log_file = os.path.join(LOGDIR, log_name)
        with open(log_file, "w") as log:
            log.write("$ " + " ".join(args) + f"  (cwd={NCCL4PY_DIR})\n\n")
            log.flush()
            proc = subprocess.run(
                args,
                cwd=NCCL4PY_DIR,
                env=_runtime_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=300,
            )
        return proc, log_file

    return _run
