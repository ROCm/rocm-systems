# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Integration tests for SDMA (System DMA) metrics collection.

These tests validate that SDMA Usage metrics are correctly collected and
output to all supported formats (Perfetto Standard, Perfetto Legacy, RocPD).
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu, pytest.mark.sdma, pytest.mark.ci_enable]

# =============================================================================
# SDMA test fixtures
# =============================================================================


@pytest.fixture
def sdma_perfetto_env() -> dict[str, str]:
    """Environment variables for SDMA Perfetto Standard test."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "OFF",
        "ROCPROFSYS_AMD_SMI_METRICS": "sdma_usage",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_CPU_FREQ_ENABLED": "OFF",
    }


@pytest.fixture
def sdma_perfetto_legacy_env() -> dict[str, str]:
    """Environment variables for SDMA Perfetto Legacy test."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "ON",
        "ROCPROFSYS_AMD_SMI_METRICS": "sdma_usage",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_CPU_FREQ_ENABLED": "OFF",
    }


@pytest.fixture
def sdma_rocpd_env() -> dict[str, str]:
    """Environment variables for SDMA RocPD test."""
    return {
        "ROCPROFSYS_USE_ROCPD": "ON",
        "ROCPROFSYS_AMD_SMI_METRICS": "sdma_usage",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_CPU_FREQ_ENABLED": "OFF",
    }


@pytest.fixture
def sdma_all_metrics_env() -> dict[str, str]:
    """Environment variables for SDMA test with all AMD SMI metrics."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_AMD_SMI_METRICS": "all",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_CPU_FREQ_ENABLED": "OFF",
    }


# =============================================================================
# SDMA integration tests
# =============================================================================


class TestSDMA(RocprofsysTest):
    """Integration tests for SDMA metrics collection and output."""

    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_sdma_perfetto_standard(self, mode, sdma_perfetto_env):
        """
        Test SDMA metrics in Perfetto Standard format.

        This test validates that:
        1. sdma_test workload runs successfully
        2. SDMA Usage counter tracks appear in Perfetto trace
        3. Counter values are reasonable (> 0% for at least some samples)

        Expected counter tracks:
        - GPU [0] SDMA Usage (S)
        - GPU [1] SDMA Usage (S) (if multiple GPUs)
        """
        result = self.run_test(
            mode,
            "sdma_test",
            env=sdma_perfetto_env,
            check_target_arch=False,  # sdma_test is architecture-independent
            timeout=60,
        )

        # Validate test completed successfully
        self.assert_regex(result)

        # Validate Perfetto output contains SDMA counter tracks
        if mode == "sys_run":
            self.assert_perfetto(
                result,
                counter_names=["SDMA Usage"],
            )

    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_sdma_perfetto_legacy(self, mode, sdma_perfetto_legacy_env):
        """
        Test SDMA metrics in Perfetto Legacy format.

        This test validates that SDMA metrics appear in the legacy Perfetto
        format with the alternate naming convention.

        Expected counter tracks:
        - GPU SDMA Usage [0] (S)
        - GPU SDMA Usage [1] (S) (if multiple GPUs)
        """
        result = self.run_test(
            mode,
            "sdma_test",
            env=sdma_perfetto_legacy_env,
            check_target_arch=False,  # sdma_test and transpose are architecture-independent
            timeout=60,
        )

        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_perfetto(
                result,
                counter_names=["SDMA Usage"],
                            )

    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_sdma_rocpd(self, mode, sdma_rocpd_env):
        """
        Test SDMA metrics in RocPD database format.

        This test validates that SDMA metrics are written to the RocPD
        database with the expected metric names.

        Expected database entries:
        - device_sdma_usage in rocpd_info_pmc table
        """
        result = self.run_test(
            mode,
            "sdma_test",
            env=sdma_rocpd_env,
            check_target_arch=False,  # sdma_test and transpose are architecture-independent
            timeout=60,
        )

        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_rocpd(result)
            # Additional validation: check that device_sdma_usage exists in database
            # This could be enhanced with specific SQL queries in the future

    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_sdma_with_all_metrics(self, mode, sdma_all_metrics_env):
        """
        Test SDMA metrics alongside all other AMD SMI metrics.

        This test validates that SDMA metrics appear correctly when collected
        alongside other GPU metrics (power, temp, busy, etc.).

        Expected counter tracks (subset):
        - GPU [0] SDMA Usage (S)
        - GPU [0] Current Power (S)
        - GPU [0] GFX Busy (S)
        - GPU [0] Temperature (S)
        """
        result = self.run_test(
            mode,
            "sdma_test",
            env=sdma_all_metrics_env,
            check_target_arch=False,  # sdma_test and transpose are architecture-independent
            timeout=60,
        )

        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_perfetto(
                result,
                counter_names=[
                    "SDMA Usage",
                    "Current Power",
                    "GFX Busy",
                    "Temperature",
                ],
                            )

    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_sdma_transpose_workload(self, mode, sdma_perfetto_env):
        """
        Test SDMA metrics with transpose workload.

        The transpose workload also generates DMA activity through HIP memcpy
        operations, so SDMA metrics should be non-zero.
        """
        result = self.run_test(
            mode,
            "transpose",
            env=sdma_perfetto_env,
            check_target_arch=False,  # sdma_test and transpose are architecture-independent
            timeout=60,
        )

        self.assert_regex(result)

        if mode == "sys_run":
            # Note: transpose may have lower SDMA activity than sdma_test
            # We just validate the counter tracks exist
            self.assert_perfetto(
                result,
                counter_names=["SDMA Usage"],
                            )
