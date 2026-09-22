# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AI NIC tests.

Two independent checks live here:

* ``test_settings_present`` — a hardware-independent build-artifact check that
  verifies AI NIC support was compiled in. The AI NIC settings (e.g.
  ``ROCPROFSYS_USE_AINIC``) are only registered when the binaries are built with
  ``ROCPROFSYS_BUILD_AINIC=ON``, so their presence in ``rocprof-sys-avail
  --settings`` is a direct indicator of compile-time support. No NIC hardware is
  required.

* ``test_performance_tracks`` — verifies that rocprofiler-systems correctly
  collects the AI NIC RDMA counters per device and writes them to both the
  Perfetto (.proto) trace and the ROCpd (.db) database. AI NIC devices are
  discovered at runtime via ``amd-smi static | grep -i netdev``; this test is
  skipped automatically when no AI NIC devices are present on the system.
"""

from __future__ import annotations

import os
import subprocess
import pytest
import shutil
from pathlib import Path
from conftest import RocprofsysTest
from rocprofsys import RocprofsysConfig

# AI NIC support is only compiled in when AMD SMI >= 26.3.
pytestmark = [
    pytest.mark.amdsmi_min_version("26.3"),
    pytest.mark.ainic,
]

# =============================================================================
# Constants
# =============================================================================

# Settings registered only when ROCPROFSYS_BUILD_AINIC=ON (see cmake/Packages.cmake
# and the guarded ROCPROFSYS_CONFIG_SETTING blocks in config.cpp).
AINIC_SETTINGS = [
    "ROCPROFSYS_USE_AINIC",
    "ROCPROFSYS_SAMPLING_AINICS",
]

# Targets whose --help=sampling output should advertise the AI NIC flags.
TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
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
def ainic_download_url() -> str:
    """Download URL for the first file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.1/rocprofiler-systems-1.0.1-ubuntu-22.04-ROCm-60400-PAPI-OMPT-Python3.sh"


@pytest.fixture
def ainic_rocpd_rules(validation_rules_dir) -> list[Path]:
    """Validation rules for AI NIC RDMA track presence in ROCpd database."""
    rules_dir = validation_rules_dir / "ainic"
    return [rules_dir / "ainic-rdma-rules.json"]


@pytest.fixture
def ainic_two_card_env() -> dict[str, str]:
    """Environment variables for the two-card AI NIC RDMA test.

    Identical to ``ainic_perf_env``; kept separate so it can evolve
    independently (e.g., to tune sampling frequency for long RDMA bursts).
    """
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


# =============================================================================
# Tests
# =============================================================================


class TestAINIC(RocprofsysTest):
    """Tests for AI NIC support and performance metric collection."""

    PERFETTO_PASS_REGEX = [r"perfetto-trace\.proto validated"]
    PERFETTO_FAIL_REGEX = [r"Failure validating.*perfetto-trace\.proto"]

    # No NIC hardware required
    @pytest.mark.timeout(30)
    def test_settings_present(self, rocprof_config: RocprofsysConfig):
        """AI NIC settings must be listed by ``rocprof-sys-avail --settings``.

        These settings are only registered when the binaries are compiled with
        ROCPROFSYS_BUILD_AINIC=ON, so their presence proves AI NIC support was
        compiled in. No NIC hardware is required.
        """
        result = subprocess.run(
            [str(rocprof_config.rocprofsys_avail), "--settings"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"rocprof-sys-avail failed: {result.stderr}"

        settings = result.stdout
        missing = [s for s in AINIC_SETTINGS if s not in settings]
        assert not missing, (
            "AI NIC settings not reported by rocprof-sys-avail --settings — "
            "were the binaries built with ROCPROFSYS_BUILD_AINIC=OFF?\n"
            f"Missing: {missing}"
        )

    # No NIC hardware required; the module-level amdsmi_min_version("26.3") gate
    # skips this on builds where AI NIC support is not compiled in.
    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.timeout(30)
    def test_sampling_ainics_help(self, target):
        """--sampling-ainics must appear in the sampling help output.

        The flag is only registered (and therefore only shown by
        ``--help=sampling``) when the binary was built with
        ROCPROFSYS_BUILD_AINIC=ON.
        """
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--help=sampling"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"--sampling-ainics"])

    @pytest.mark.ainic_required
    @pytest.mark.network
    @pytest.mark.rocpd("ainic_perf_env")
    @pytest.mark.timeout(120)
    def test_performance_tracks(
        self,
        ainic_perf_env,
        ainic_download_url,
        ainic_rocpd_rules,
        test_output_dir,
    ):
        target = shutil.which("wget")
        if not target:
            pytest.skip("wget not found")

        download_cmd = [
            "--no-check-certificate",
            ainic_download_url,
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
        # We are only validating the presence: in this test, RDMA counters may
        # legitimately be 0.
        self.assert_perfetto(
            result,
            counter_names=AINIC_PERFETTO_COUNTER_NAMES,
            counter_names_presence_only=True,
            pass_regex=self.PERFETTO_PASS_REGEX,
            fail_regex=self.PERFETTO_FAIL_REGEX,
        )

        # Validate ROCpd .db: all 10 AI NIC track names must be present
        self.assert_rocpd(
            result,
            subtest_name="ROCpd AI NIC track validation",
            rules_files=ainic_rocpd_rules,
        )

    @pytest.mark.ainic_required
    @pytest.mark.ainic_two_card
    @pytest.mark.network
    @pytest.mark.rocpd("ainic_two_card_env")
    @pytest.mark.timeout(180)
    def test_rdma_two_card(
        self,
        ainic_two_card_env,
        ainic_rocpd_rules,
        test_output_dir,
    ):
        """Two-card RDMA test: strict validation of non-zero AI NIC metrics.

        **Disabled by default.**  Enable by exporting the three required
        environment variables before running pytest::

            ROCPROFSYS_AINIC_REMOTE_IP=<remote-host>   # SSH target
            ROCPROFSYS_AINIC_LOCAL_IP=<local-nic-ip>   # IP client connects to
            ROCPROFSYS_AINIC_IB_DEVICE=ionic_0         # optional, default ionic_0

        Hardware requirements:

        * Two machines, each with a Pensando Pollara 400G AI NIC.
        * Both NICs connected via a cable or an Ethernet switch.
        * Both ports in PORT_ACTIVE state (``/sys/class/infiniband/ionic_0/ports/1/state``).
        * ``ib_write_bw`` (from the *perftest* package) installed on **both** machines.
        * Passwordless SSH from this machine to ``ROCPROFSYS_AINIC_REMOTE_IP``.

        Test flow:

        1. Start ``ib_write_bw`` **server** on this machine under
           ``rocprof-sys-sample`` so AI NIC PMC counters are collected.
        2. A background thread SSHes to the remote machine (after a 3-second
           delay to let the server come up) and starts the ``ib_write_bw``
           **client**, which writes 10 × 1 MB messages over RDMA.
        3. The client exits when the transfer is complete; the server exits
           immediately after.
        4. Strict Perfetto validation (``counter_names_presence_only=False``)
           asserts that TX/RX RDMA counter values are **non-zero**.
        5. ROCpd database is validated for AI NIC track presence.
        """
        import time
        import threading

        remote_ip = os.environ["ROCPROFSYS_AINIC_REMOTE_IP"]
        local_ip = os.environ["ROCPROFSYS_AINIC_LOCAL_IP"]
        ib_device = os.environ.get("ROCPROFSYS_AINIC_IB_DEVICE", "ionic_0")

        ib_write_bw = shutil.which("ib_write_bw")
        if not ib_write_bw:
            pytest.skip("ib_write_bw not found on PATH")

        # ib_write_bw server args: use the IB device, run 10 iterations
        server_args = ["-d", ib_device, "-n", "10"]

        # Collect errors from the remote-client thread so they surface in the
        # main test rather than being silently swallowed.
        client_errors: list[str] = []

        def _run_remote_client() -> None:
            # Give the server a moment to open its listen socket.
            time.sleep(3)
            try:
                proc = subprocess.run(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=15",
                        remote_ip,
                        "ib_write_bw",
                        local_ip,
                        "-d", ib_device,
                        "-n", "10",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode != 0:
                    client_errors.append(
                        f"Remote ib_write_bw client exited with rc={proc.returncode}\n"
                        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
                    )
            except subprocess.TimeoutExpired:
                client_errors.append("Remote ib_write_bw client timed out (>120 s)")
            except Exception as exc:  # noqa: BLE001
                client_errors.append(f"Remote ib_write_bw client raised: {exc}")

        client_thread = threading.Thread(target=_run_remote_client, daemon=True)
        client_thread.start()

        # Run the profiled server — blocks until the client finishes and the
        # server exits.
        result = self.run_test(
            "sampling",
            ib_write_bw,
            run_args=server_args,
            env=ainic_two_card_env,
        )

        client_thread.join(timeout=30)

        if client_errors:
            pytest.fail("Remote client side failed:\n" + client_errors[0])

        self.assert_regex(result)

        # Strict validation: real RDMA traffic MUST produce non-zero counter
        # values.  counter_names_presence_only is intentionally False (default).
        self.assert_perfetto(
            result,
            counter_names=AINIC_PERFETTO_COUNTER_NAMES,
            pass_regex=self.PERFETTO_PASS_REGEX,
            fail_regex=self.PERFETTO_FAIL_REGEX,
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd AI NIC two-card RDMA validation",
            rules_files=ainic_rocpd_rules,
        )
