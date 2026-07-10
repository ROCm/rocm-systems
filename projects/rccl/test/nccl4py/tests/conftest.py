# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Shared fixtures for the nccl4py build-smoke pytest harness.

This mirrors test/ir-device: the supported ROCm install path (``pip install``)
is exercised on demand from a session-scoped fixture, and the whole suite is
skipped with a clear reason when prerequisites are missing.

Prerequisites (environment variables, with sensible defaults):

  RCCL_DIR      RCCL source root             (default: repo root, derived)
  RCCL_BUILD    RCCL CMake build dir         (default: $RCCL_DIR/build/release)
  ROCM_PATH     ROCm install root            (default: /opt/rocm)
  NCCL4PY_DIR   nccl4py source tree          (default: $RCCL_DIR/bindings/nccl4py)
  NCCL4PY_OUT   editable-install staging dir (default: <workdir>/nccl4py_build)

RCCL must have been built at least once so ``librccl.so`` is discoverable from
``$RCCL_BUILD`` or ``$ROCM_PATH/lib``.
"""

from __future__ import annotations

import logging
import os
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
NCCL4PY_OUT = os.environ.get("NCCL4PY_OUT", os.path.join(WORKDIR, "nccl4py_build"))

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
    for d in (os.path.join(RCCL_BUILD), os.path.join(ROCM_PATH, "lib"), os.path.join(ROCM_PATH, "lib64")):
        if os.path.isdir(d) and d not in lib_dirs:
            lib_dirs.append(d)
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


def _refresh_site_packages() -> None:
    """Pick up editable-install .pth files added by a subprocess pip install.

    Python only processes site-packages .pth files at interpreter startup, so
    an in-process import after ``pip install -e`` in a child process needs this.
    """
    import importlib
    import site

    for d in site.getsitepackages():
        if os.path.isdir(d):
            site.addsitedir(d)
    importlib.invalidate_caches()


def _pip_install_editable() -> str:
    """``pip install -e`` the nccl4py tree; return the build log path."""
    os.makedirs(NCCL4PY_OUT, exist_ok=True)
    build_log = os.path.join(LOGDIR, "nccl4py_pip_install.log")
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        "Cython>=3.1",
    ]
    with open(build_log, "w") as log:
        log.write("$ " + " ".join(args) + "\n\n")
        log.flush()
        proc = subprocess.run(
            args,
            cwd=NCCL4PY_DIR,
            env=_runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(f"Failed to install build deps (see {build_log})")

        log.write("\n$ pip install -e .\n\n")
        log.flush()
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=NCCL4PY_DIR,
            env=_runtime_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
    assert proc.returncode == 0, f"Failed to pip install nccl4py (see {build_log})"
    _refresh_site_packages()
    return build_log


@pytest.fixture(scope="session")
def nccl4py_installed():
    """Build/install nccl4py via the supported ROCm path once per session."""
    reason = _missing_prerequisite()
    if reason:
        pytest.skip(f"nccl4py build-smoke skipped: {reason}")

    logger.info("pip install -e %s (ROCm path)...", NCCL4PY_DIR)
    build_log = _pip_install_editable()
    logger.info("nccl4py editable install complete (log: %s)", build_log)


@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        WORKDIR=WORKDIR,
        RCCL_DIR=RCCL_DIR,
        RCCL_BUILD=RCCL_BUILD,
        ROCM_PATH=ROCM_PATH,
        NCCL4PY_DIR=NCCL4PY_DIR,
        NCCL4PY_OUT=NCCL4PY_OUT,
        LOGDIR=LOGDIR,
        LIBRCCL=_find_librccl(),
    )


@pytest.fixture(scope="session")
def run_nccl4py_pytest(nccl4py_installed):
    """Run a pytest module under bindings/nccl4py/tests and return (proc, log)."""

    def _run(relative_test_path: str, log_name: str) -> tuple[subprocess.CompletedProcess, str]:
        target = os.path.join(NCCL4PY_DIR, relative_test_path)
        if not os.path.isfile(target):
            raise FileNotFoundError(target)
        args = [
            sys.executable,
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
