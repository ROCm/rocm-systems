# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
NIC tests.
"""

from __future__ import annotations
import pytest
import shutil
from conftest import RocprofsysTest

pytestmark = [pytest.mark.nic, pytest.mark.network]

# =============================================================================
# NIC fixtures
# =============================================================================


@pytest.fixture
def nic_perf_env(rocprof_config) -> dict[str, str]:
    """Environment variables for NIC performance tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "ON",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_ROCM": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,papi_array,network_stats",
        "ROCPROFSYS_NETWORK_INTERFACE": f"{rocprof_config.capabilities.default_nic}",
        "ROCPROFSYS_PAPI_EVENTS": f"{rocprof_config.capabilities.papi_nic_events}",
        "ROCPROFSYS_SAMPLING_DELAY": "0.05",
    }


@pytest.fixture
def nic_perf_download_url_1() -> str:
    """Download URL for the first file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.1/rocprofiler-systems-1.0.1-ubuntu-22.04-ROCm-60400-PAPI-OMPT-Python3.sh"


@pytest.fixture
def nic_perf_download_url_2() -> str:
    """Download URL for the second file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.3/rocprofiler-systems-1.0.2-rhel-9.4-PAPI-OMPT-Python3.sh"


# =============================================================================
# NIC tests
# =============================================================================


# @pytest.mark.ci_disable("assert_perfetto")
class TestNIC(RocprofsysTest):
    """Tests for NIC performance."""

    PERFETTO_PASS_REGEX = [r"perfetto-trace\.proto validated"]
    PERFETTO_FAIL_REGEX = [r"Failure validating.*perfetto-trace\.proto"]

    def test_performance(
        self,
        nic_perf_env,
        nic_perf_download_url_1,
        nic_perf_download_url_2,
        test_output_dir,
    ):
        target = shutil.which("wget")
        if not target:
            pytest.skip("wget not found")

        download_cmd = [
            "--no-check-certificate",
            nic_perf_download_url_1,
            nic_perf_download_url_2,
            "-O",
            str(test_output_dir / "rocprofiler-systems.test.bin"),
        ]
        result = self.run_test(
            "sampling",
            target,
            run_args=download_cmd,
            env=nic_perf_env,
            timeout=300,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            counter_names=["rx:byte", "rx:packet", "tx:byte", "tx:packet"],
            pass_regex=self.PERFETTO_PASS_REGEX,
            fail_regex=self.PERFETTO_FAIL_REGEX,
        )
