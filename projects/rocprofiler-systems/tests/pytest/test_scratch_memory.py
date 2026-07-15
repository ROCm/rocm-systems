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
    pytest.mark.rocm,
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
        rules_dir / "sdk-metrics-rules.json",
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
                categories=["rocm_scratch_memory"],
            )
            self.assert_rocpd(
                result,
                rules_files=scratch_memory_rules,
            )
