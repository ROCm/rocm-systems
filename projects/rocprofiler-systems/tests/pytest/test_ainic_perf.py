# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AI NIC tests using AMD SMI Phase 2 RDMA metrics (amdsmi_get_nic_rdma_dev_info).

These tests verify that rocprofiler-systems correctly collects all 10 AI NIC
RDMA counters per device and writes them to both the Perfetto (.proto) trace
and the ROCpd (.db) database.

AI NIC devices are discovered at runtime via ``amd-smi static | grep -i netdev``.
The test is skipped automatically when no AI NIC devices are present on the system.
"""

from __future__ import annotations

import os
import sqlite3
import pytest
import shutil
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.ainic, pytest.mark.network]

# =============================================================================
# Constants
# =============================================================================

# The 10 AI NIC RDMA track names written to the ROCpd (.db) database.
AINIC_ROCPD_TRACK_NAMES = [
    "ainic_rx_rdma_ucast_bytes",
    "ainic_tx_rdma_ucast_bytes",
    "ainic_rx_rdma_ucast_pkts",
    "ainic_tx_rdma_ucast_pkts",
    "ainic_rx_rdma_cnp_pkts",
    "ainic_tx_rdma_cnp_pkts",
    "ainic_tx_rdma_ack_timeout",
    "ainic_resp_tx_pkt_seq_err",
    "ainic_req_rx_pkt_seq_err",
    "ainic_req_rx_impl_nak_seq_err",
]

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
def ainic_perf_env(rocprof_config) -> dict[str, str]:
    """Environment variables for AI NIC performance tests."""
    env = {
        "ROCPROFSYS_TRACE_LEGACY": "ON",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_AMD_SMI": "ON",
        "ROCPROFSYS_USE_AINIC": "ON",
        "ROCPROFSYS_SAMPLING_AINICS": "all",
        "ROCPROFSYS_USE_ROCPD": "ON",
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


# =============================================================================
# Private helpers
# =============================================================================


def _get_ainic_tracks_from_rocpd(db_path: Path) -> set[str]:
    """Return the set of AI NIC track names found in the ROCpd SQLite database.

    Track names are stored in ``rocpd_string_{upid}`` tables. We query every
    such table for strings that start with ``ainic_``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name LIKE 'rocpd_string_%'"
        )
        string_tables = [row[0] for row in cursor.fetchall()]

        found: set[str] = set()
        for table in string_tables:
            try:
                cursor.execute(
                    f"SELECT DISTINCT string FROM {table} WHERE string LIKE 'ainic_%'"
                )
                found.update(row[0] for row in cursor.fetchall())
            except sqlite3.Error:
                pass
        return found
    finally:
        conn.close()


# =============================================================================
# Tests
# =============================================================================


class TestAINIC(RocprofsysTest):
    """Tests for AI NIC performance using AMD SMI Phase 2 RDMA metrics."""

    PERFETTO_PASS_REGEX = [r"perfetto-trace\.proto validated"]
    PERFETTO_FAIL_REGEX = [r"Failure validating.*perfetto-trace\.proto"]

    def test_performance(
        self,
        rocprof_config,
        ainic_perf_env,
        ainic_download_url_1,
        ainic_download_url_2,
        test_output_dir,
        subtests,
        record_subtest_failure,
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
        subtest_name = "ROCpd AI NIC track validation"
        with subtests.test(subtest_name):
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                record_subtest_failure(subtest_name)
                pytest.fail("ROCpd database (.db) was not created")

            found_tracks = _get_ainic_tracks_from_rocpd(rocpd_file)
            missing = [t for t in AINIC_ROCPD_TRACK_NAMES if t not in found_tracks]
            if missing:
                record_subtest_failure(subtest_name)
                pytest.fail(
                    f"Missing AI NIC tracks in ROCpd database:\n"
                    f"  Missing : {missing}\n"
                    f"  Found   : {sorted(found_tracks)}\n"
                    f"  Database: {rocpd_file}"
                )
