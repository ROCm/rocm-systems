# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Build and runtime smoke tests for the ROCm nccl4py pip install path.

Each case validates that ``pip install -e .`` succeeds (via the session
fixture in conftest.py) and that the CPU-only pytest modules under
``bindings/nccl4py/tests`` pass against the installed package. GPU-backed
shim tests are included as an optional case that self-skips when no HIP
devices are visible.
"""

import pytest

CPU_SMOKE_TESTS = ("tests/test_rocm_extensions.py",)

GPU_SMOKE_TESTS = ("tests/test_shim_surface.py",)


@pytest.mark.nccl4py
@pytest.mark.nccl4py_cpu
def test_import_nccl_bindings(nccl4py_installed):
    """Editable pip install produced an importable nccl.bindings module."""
    import nccl.bindings  # noqa: F401


@pytest.mark.nccl4py
@pytest.mark.nccl4py_cpu
@pytest.mark.parametrize(
    "relative_test_path",
    CPU_SMOKE_TESTS,
    ids=[p.split("/")[-1] for p in CPU_SMOKE_TESTS],
)
def test_cpu_smoke_modules(relative_test_path, run_nccl4py_pytest):
    """Run CPU-only nccl4py smoke modules after editable pip install."""
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
