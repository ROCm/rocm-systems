# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the hipfile example (hipFILE API tracing).
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hipfile,
    pytest.mark.rocm,
    # hipFILE callback tracing domain requires rocprofiler-sdk >= 1.3.5
    pytest.mark.rocprofiler_sdk_min_version("1.3.5"),
]


# =============================================================================
# hipFILE fixtures
# =============================================================================


@pytest.fixture
def hipfile_env() -> dict[str, str]:
    """Environment variables for hipFILE trace tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,hipfile_api",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
    }


@pytest.fixture
def hipfile_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for hipFILE trace tests."""
    rules_dir = validation_rules_dir / "hipfile"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# =============================================================================
# hipFILE tests
# =============================================================================


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("sampling", marks=pytest.mark.rocpd("hipfile_env")),
        "sys_run",
    ],
)
@pytest.mark.class_name("hipfile")
class TestHipFile(RocprofsysTest):
    def test(self, mode, hipfile_env, hipfile_rules):
        result = self.run_test(
            mode,
            "hipfile-trace",
            env=hipfile_env,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                categories=["rocm_hipfile_api"],
                labels=["hipFileGetVersion"],
                counts=[1],
                depths=[1],
            )
            self.assert_rocpd(result, rules_files=hipfile_rules)
