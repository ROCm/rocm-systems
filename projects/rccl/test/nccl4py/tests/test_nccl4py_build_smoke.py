# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Build and runtime smoke tests for the nccl4py uv/CMake build path.

These validate the NCCL 2.29.7 "Added CMake for NCCL4Py build" feature as
adopted by RCCL: the ``BUILD_NCCL4PY`` CMake option and the ``nccl4py``
target (backed by ``uv build``) produce an importable wheel, and the
CPU-only pytest modules under ``bindings/nccl4py/tests`` pass against it.
GPU-backed shim tests are included as an optional case that self-skips
when no HIP devices are visible.
"""

import os
import subprocess

import pytest

CPU_SMOKE_TESTS = ("tests/test_rocm_extensions.py",)

GPU_SMOKE_TESTS = ("tests/test_shim_surface.py",)


@pytest.mark.nccl4py
@pytest.mark.nccl4py_cpu
def test_cmake_target_builds_wheel(nccl4py_wheel):
    """The uv/CMake nccl4py target produced a wheel artifact."""
    assert nccl4py_wheel.endswith(".whl"), nccl4py_wheel
    assert os.path.isfile(nccl4py_wheel), nccl4py_wheel
    assert os.path.getsize(nccl4py_wheel) > 0, nccl4py_wheel


@pytest.mark.nccl4py
@pytest.mark.nccl4py_cpu
def test_import_nccl_bindings(nccl4py_env):
    """The wheel installed into the uv env exposes an importable nccl.bindings."""
    proc = subprocess.run(
        [nccl4py_env, "-c", "import nccl.bindings"],
        capture_output=True,
        universal_newlines=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.nccl4py
@pytest.mark.nccl4py_cpu
@pytest.mark.parametrize(
    "relative_test_path",
    CPU_SMOKE_TESTS,
    ids=[p.split("/")[-1] for p in CPU_SMOKE_TESTS],
)
def test_cpu_smoke_modules(relative_test_path, run_nccl4py_pytest):
    """Run CPU-only nccl4py smoke modules against the uv/CMake-built wheel."""
    log_name = relative_test_path.replace("/", "_").replace(".py", ".log")
    proc, log = run_nccl4py_pytest(relative_test_path, log_name)
    assert proc.returncode == 0, f"{relative_test_path} failed, see {log}"


@pytest.mark.nccl4py
@pytest.mark.nccl4py_gpu
@pytest.mark.parametrize(
    "relative_test_path",
    GPU_SMOKE_TESTS,
    ids=[p.split("/")[-1] for p in GPU_SMOKE_TESTS],
)
def test_gpu_smoke_modules(relative_test_path, run_nccl4py_pytest):
    """Run optional GPU shim tests (module skips when no HIP devices)."""
    log_name = relative_test_path.replace("/", "_").replace(".py", ".log")
    proc, log = run_nccl4py_pytest(relative_test_path, log_name)
    # Exit 0 means pass or skip-all; 5 is pytest's "no tests collected" which
    # we treat as skip when the module bails at import time on CPU-only hosts.
    assert proc.returncode in (0, 5), f"{relative_test_path} failed, see {log}"
