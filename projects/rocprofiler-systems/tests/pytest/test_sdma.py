# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for SDMA (System DMA) usage metrics.

Validates that running sdma-test with ROCPROFSYS_AMD_SMI_METRICS=sdma_usage
produces expected SDMA usage tracks and values in Perfetto and ROCPD output,
as implemented in the amd_smi component.

SDMA usage requires an Instinct GPU, AMD-SMI >= 26.3
(see source/lib/core/sdma_feature.hpp), and amdgpu driver >= 6.19.14;
tests are skipped otherwise.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def sdma_env() -> dict[str, str]:
    """Environment variables for SDMA tests (sdma_usage metric only, legacy trace)."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "sdma_usage",
    }


@pytest.fixture
def sdma_rules(validation_rules_dir: Path) -> list[Path]:
    """Validation rules for SDMA ROCPD (device_sdma_usage in rocpd_info_pmc / rocpd_pmc_event)."""
    rules_dir = validation_rules_dir / "sdma"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "amd-smi-sdma-rules.json",
    ]


@pytest.mark.run_if_gpu_category("instinct")
@pytest.mark.amdsmi_min_version("26.3")
@pytest.mark.amdgpu_min_version("6.19.14")
class TestSDMA(RocprofsysTest):
    """Tests for SDMA usage metrics (Perfetto and ROCPD)."""

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "mode", [pytest.param("sys_run", marks=pytest.mark.rocpd("sdma_env"))]
    )
    def test_usage(self, mode, sdma_env, sdma_rules):
        """Run sdma-test with sdma_usage and validate Perfetto + ROCPD for SDMA tracks/values."""
        # sdma-test is a host-side hipMemcpy benchmark with no device kernels,
        # so the binary embeds no gfx offload bundle; skip the target-arch check
        # (it would always report no supported architectures).
        result = self.run_test(
            mode,
            "sdma-test",
            env=sdma_env,
            run_args=["-n", "2", "-s", "64"],
        )
        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_rocpd(result, rules_files=sdma_rules)
            self.assert_perfetto(
                result,
                counter_names=["SDMA Usage"],
                subtest_name="Perfetto SDMA Usage counter validation",
            )
