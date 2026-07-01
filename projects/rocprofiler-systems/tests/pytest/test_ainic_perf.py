# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AI NIC tests using AMD SMI RDMA metrics (amdsmi_get_nic_rdma_dev_info).

These tests verify that rocprofiler-systems correctly collects all AI NIC
RDMA counters per device and writes them to both the Perfetto (.proto) trace
and the ROCpd (.db) database.

AI NIC devices are discovered at runtime via ``amd-smi static | grep -i netdev``.
The test is skipped automatically when no AI NIC devices are present on the system.
"""

from __future__ import annotations

import os
import pytest
import shutil
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.ainic, pytest.mark.network]

# =============================================================================
# Constants
# =============================================================================

# Substrings used to match the 10 Perfetto counter track names via LIKE.
# Full name format: "NIC [<device_id>] <METRIC> (S)"
AINIC_PERFETTO_COUNTER_NAMES = [
    "RX RDMA Bytes",
    "TX RDMA Bytes",
    "RX RDMA Packets",
    "TX RDMA Packets",
    "RX CNP Packets",
    "TX CNP Packets",
    "TX ACK TIMEOUT",
    "RESP TX PKT SEQ ERR",
    "REQ RX PKT SEQ ERR",
    "REQ RX IMPL NAK SEQ ERR",
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ainic_perf_env() -> dict[str, str]:
    """Environment variables for AI NIC performance tests."""
    env = {
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_AMD_SMI": "ON",
        "ROCPROFSYS_USE_AINIC": "ON",
        "ROCPROFSYS_SAMPLING_AINICS": "all",
        "ROCPROFSYS_SAMPLING_DELAY": "0.05",
    }
    sysfs_root = os.environ.get("SMI_NIC_SYSFS_ROOT", "")
    if sysfs_root:
        env["SMI_NIC_SYSFS_ROOT"] = sysfs_root
    return env


@pytest.fixture
def ainic_download_url_1() -> str:
    """Download URL for the first file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.1/rocprofiler-systems-1.0.1-ubuntu-22.04-ROCm-60400-PAPI-OMPT-Python3.sh"


@pytest.fixture
def ainic_download_url_2() -> str:
    """Download URL for the second file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.3/rocprofiler-systems-1.0.2-rhel-9.4-PAPI-OMPT-Python3.sh"


@pytest.fixture
def ainic_rocpd_rules(validation_rules_dir) -> list[Path]:
    """Validation rules for AI NIC RDMA track presence in ROCpd database."""
    rules_dir = validation_rules_dir / "ainic"
    return [rules_dir / "ainic-rdma-rules.json"]


# =============================================================================
# Tests
# =============================================================================


class TestAINIC(RocprofsysTest):
    """Tests for AI NIC performance using AMD SMI Phase 2 RDMA metrics."""

    PERFETTO_PASS_REGEX = [r"perfetto-trace\.proto validated"]
    PERFETTO_FAIL_REGEX = [r"Failure validating.*perfetto-trace\.proto"]

    @pytest.mark.rocpd("ainic_perf_env")
    def test_performance_tracks(
        self,
        ainic_perf_env,
        ainic_download_url_1,
        ainic_download_url_2,
        ainic_rocpd_rules,
        test_output_dir,
    ):
        target = shutil.which("wget")
        if not target:
            pytest.skip("wget not found")

        download_cmd = [
            "--no-check-certificate",
            ainic_download_url_1,
            ainic_download_url_2,
            "-O",
            str(test_output_dir / "rocprofiler-systems.test.bin"),
        ]
        result = self.run_test(
            "sampling",
            target,
            run_args=download_cmd,
            env=ainic_perf_env,
        )

        self.assert_regex(result)

        # Validate Perfetto .proto: all 10 AI NIC counter track substrings must match
        self.assert_perfetto(
            result,
            counter_names=AINIC_PERFETTO_COUNTER_NAMES,
            pass_regex=self.PERFETTO_PASS_REGEX,
            fail_regex=self.PERFETTO_FAIL_REGEX,
        )

        # Validate ROCpd .db: all 10 AI NIC track names must be present
        self.assert_rocpd(
            result,
            subtest_name="ROCpd AI NIC track validation",
            rules_files=ainic_rocpd_rules,
        )
