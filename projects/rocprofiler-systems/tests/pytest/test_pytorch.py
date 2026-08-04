# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the PyTorch function-fit example.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest
from pathlib import Path

pytestmark = [
    pytest.mark.pytorch,
    pytest.mark.gpu,
]

# =============================================================================
# PyTorch fixtures
# =============================================================================


@pytest.fixture
def pytorch_env() -> dict[str, str]:
    """Tracing settings for the example.

    Must be used with ``rocprof-sys-python`` and not ``python -m rocprofsys``
    """
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation"
    }


@pytest.fixture
def pytorch_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "pytorch"
    return [
        validation_rules_dir / "default-rules.json",
        rules_dir / "validation-rules.json",
    ]


# =============================================================================
# Tests
# =============================================================================

# Notes for anyone reading the trace:
#
# 1. The first `train_one_epoch` is far longer than the rest: batch 1 pays for
#    first-touch initialization -- rocBLAS lazily loads a Tensile kernel object
#    per GEMM shape, AdamW allocates its state tensors, `mse_loss` pulls in its
#    reduction kernel, and the caching allocator carves out its segments.
# 2. The blank region before `main` is the module-level `import torch`: the
#    import machinery is not instrumented, and most of the cost is native work
#    with no Python frame at all (dlopen of libtorch and the ROCm runtime, plus
#    their static initializers).
# 3. The backward pass runs on its own thread, so kernel dispatches are split
#    between the main thread and autograd's.


@pytest.mark.timeout(120)
@pytest.mark.rocpd("pytorch_env")
@pytest.mark.python_versions
@pytest.mark.pytorch_available
class TestPytorch(RocprofsysTest):
    # Changing this requires updating the rocpd validation rules
    FUNCTION_FIT_EPOCHS = 200

    def test_function_fit(self, python_version, pytorch_env, pytorch_rocpd_rules):
        result = self.run_test(
            "python",
            target="pytorch_function_fit.py",
            env=pytorch_env,
            python_version=python_version,
            run_args=["-e", str(self.FUNCTION_FIT_EPOCHS)],
        )
        self.assert_regex(result)
        # Exact counts only for names the torch import tree cannot also define:
        # `sinc`, `evaluate` and `sample` all collide with something torch or sympy
        # declares during import, which would inflate the count.
        self.assert_perfetto(
            result,
            subtest_name="Perfetto python region validation",
            categories=["python"],
            labels=["train_one_epoch", "require_gpu", "print_plot"],
            counts=[self.FUNCTION_FIT_EPOCHS, 1, 1],
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API validation",
            categories=["rocm_hip_api"],
            label_substrings=["hipLaunchKernel", "hipMemcpyWithStream"],
        )
        # Substrings, because the torch kernel names are mangled C++ templates.
        # 'Cijk' is a rocBLAS GEMM (one per nn.Linear), 'tanh' an activation, and
        # 'copyBuffer' the blit kernel that stages each batch onto the device.
        self.assert_perfetto(
            result,
            subtest_name="Perfetto kernel dispatch validation",
            categories=["rocm_kernel_dispatch"],
            label_substrings=["Cijk", "tanh", "copyBuffer"],
        )
        self.assert_rocpd(result, rules_files=pytorch_rocpd_rules)
