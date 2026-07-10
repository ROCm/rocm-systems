# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for scratch memory.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.scratch_memory,
    pytest.mark.gpu,
    pytest.mark.no_docker,
]

# =============================================================================
# Scratch Memory Fixtures
# =============================================================================


@pytest.fixture
def scratch_memory_env() -> dict[str, str]:
    """Environment variables for scratch memory tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_api,hsa_api,kernel_dispatch,memory_copy,memory_allocation,scratch_memory"
    }


@pytest.fixture
def scratch_memory_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "scratch-memory"
    return [
        validation_rules_dir / "default-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


SCRATCH_SAMPLE_CASES = [
    pytest.param(
        ["--rocm=hip,kernel"],
        {
            "hip_labels": ["hipHostMalloc"],
            "kernel_substrings": ["test_kern_small"],
        },
        id="rocm-hip-kernel",
    ),
]


# =============================================================================
# Scratch Memory Tests
# =============================================================================


@pytest.mark.class_name("scratch-memory")
class TestScratchMemory(RocprofsysTest):
    SCRATCH_MEMORY_PASS_REGEX = [
        "Detected [1-9][0-9]* agents",
        "Running test_primary_then_uso",
        "Running test_gridx",
        "Running Small",
        "Running Medium",
        "Running Large",
    ]
    SCRATCH_MEMORY_FAIL_REGEX = [
        "hip error",
        "HSA error",
    ]
    SCRATCH_MEMORY_LABELS = ["SCRATCH_MEMORY_ALLOC", "SCRATCH_MEMORY_FREE"]
    KERNEL_LABEL_SUBSTRINGS = [
        "test_kern_small",
        "test_kern_medium",
        "test_kern_large",
    ]
    HIP_API_CATEGORY = "rocm_hip_api"
    KERNEL_DISPATCH_CATEGORY = "rocm_kernel_dispatch"
    HIP_API_LABELS = ["hipHostMalloc", "hipDeviceSynchronize"]

    @pytest.mark.rocpd("scratch_memory_env")
    @pytest.mark.parametrize(
        "mode", ["baseline", "sampling", "binary_rewrite", "sys_run"]
    )
    def test(self, mode, scratch_memory_env, scratch_memory_rules):
        result = self.run_test(mode, "scratch-memory", env=scratch_memory_env)
        self.assert_regex(
            result,
            pass_regex=self.SCRATCH_MEMORY_PASS_REGEX,
            fail_regex=self.SCRATCH_MEMORY_FAIL_REGEX,
        )
        if mode == "sampling":
            self.assert_perfetto(
                result,
                subtest_name="Perfetto scratch memory",
                categories=["rocm_scratch_memory"],
                labels=self.SCRATCH_MEMORY_LABELS,
            )
            self.assert_perfetto(
                result,
                subtest_name="Perfetto HIP API",
                categories=[self.HIP_API_CATEGORY],
                labels=self.HIP_API_LABELS,
            )
            self.assert_perfetto(
                result,
                subtest_name="Perfetto kernel dispatch",
                categories=[self.KERNEL_DISPATCH_CATEGORY],
                label_substrings=self.KERNEL_LABEL_SUBSTRINGS,
            )
            self.assert_rocpd(
                result,
                rules_files=scratch_memory_rules,
            )


# =============================================================================
# rocprof-sys-sample CLI on scratch-memory
# =============================================================================


@pytest.mark.timeout(120)
@pytest.mark.sampling
@pytest.mark.class_name("scratch-memory-sample-cli")
class TestScratchMemorySampleCLI(RocprofsysTest):
    @pytest.mark.parametrize("sampling_args,expect", SCRATCH_SAMPLE_CASES)
    def test(self, sampling_args, expect):
        result = self.run_test(
            "sampling",
            target="scratch-memory",
            sampling_args=sampling_args,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            pass_regex=TestScratchMemory.SCRATCH_MEMORY_PASS_REGEX,
            fail_regex=TestScratchMemory.SCRATCH_MEMORY_FAIL_REGEX,
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API",
            categories=[TestScratchMemory.HIP_API_CATEGORY],
            labels=expect["hip_labels"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto kernel dispatch",
            categories=[TestScratchMemory.KERNEL_DISPATCH_CATEGORY],
            label_substrings=expect["kernel_substrings"],
        )
