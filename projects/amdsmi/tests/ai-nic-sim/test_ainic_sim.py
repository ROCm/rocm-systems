# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_ainic_sim.py — simulation-based pytest test for the amd-smi AI NIC API.

The test creates a fake sysfs tree that looks like a Pensando Pollara NIC,
starts a background simulator that increments the hw_counter files, then
runs amd_smi_ainic_info (the example binary built from amd_smi_ainic_example.cc)
with SMI_NIC_SYSFS_ROOT pointing at the fake tree.

Assertions:
  1. amd_smi_ainic_info exits with return code 0.
  2. Exactly one AI NIC device is reported.
  3. The expected RDMA hw_counter names appear in the output.
  4. At least one RDMA counter has a non-zero value (proves the simulator's
     updates were picked up during the run).

No real Pensando Pollara hardware is required.

Run:
    pytest projects/amdsmi/tests/ai-nic-sim/test_ainic_sim.py -v

The amd_smi_ainic_info binary must be built first:
    cmake -B build && cmake --build build --target amd_smi_ainic_info
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_FAKE_SYSFS_PY = _HERE / "fake_sysfs.py"
_NIC_SIM_PY = _HERE / "nic_simulator.py"

sys.path.insert(0, str(_HERE))
import fake_sysfs  # noqa: E402


def _find_ainic_info_binary() -> Path | None:
    """Search common build locations for the amd_smi_ainic_info binary."""
    from shutil import which

    found = which("amd_smi_ainic_info")
    if found:
        return Path(found)

    # projects/amdsmi/tests/ai-nic-sim  →  parents[4] = repo root
    repo_root = _HERE.parents[3]
    candidates = [
        repo_root / "build" / "projects" / "amdsmi" / "example" / "amd_smi_ainic_info",
        repo_root / "build-release" / "projects" / "amdsmi" / "example" / "amd_smi_ainic_info",
        repo_root / "projects" / "amdsmi" / "example" / "build" / "amd_smi_ainic_info",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


_BINARY = _find_ainic_info_binary()

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_sysfs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the fake sysfs tree once per test session."""
    root = tmp_path_factory.mktemp("fake-sysfs")
    fake_sysfs.create(root)
    return root


@pytest.fixture(scope="session")
def hw_counters_dir(fake_sysfs_root: Path) -> Path:
    """Return the hw_counters directory inside the fake sysfs tree."""
    return (
        fake_sysfs_root
        / "sys/devices/pci0000:e0"
        / fake_sysfs.BRIDGE_BDF
        / fake_sysfs.PORT_BDF
        / "infiniband"
        / fake_sysfs.IB_DEV
        / "ports"
        / fake_sysfs.IB_PORT
        / "hw_counters"
    )


@pytest.fixture(scope="session")
def nic_simulator(hw_counters_dir: Path):
    """Start the NIC simulator subprocess; stop it after the session."""
    proc = subprocess.Popen(
        [sys.executable, str(_NIC_SIM_PY), str(hw_counters_dir), "--interval", "0.05"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    assert proc.poll() is None, "nic_simulator failed to start"
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAINICSim:
    """Simulation-based tests for the amd-smi AI NIC API."""

    def test_fake_sysfs_structure(self, fake_sysfs_root: Path, hw_counters_dir: Path) -> None:
        """Verify the fake sysfs tree has the expected structure."""
        # PCI bus device symlinks
        assert (fake_sysfs_root / "sys/bus/pci/devices" / fake_sysfs.BRIDGE_BDF).is_symlink()
        assert (fake_sysfs_root / "sys/bus/pci/devices" / fake_sysfs.PORT_BDF).is_symlink()

        # Bridge PCI IDs
        bridge_dev = fake_sysfs_root / "sys/devices/pci0000:e0" / fake_sysfs.BRIDGE_BDF
        assert (bridge_dev / "vendor").read_text().strip() == fake_sysfs.VENDOR_ID_HEX
        assert (bridge_dev / "device").read_text().strip() == fake_sysfs.BRIDGE_DEV_HEX

        # Port PCI IDs
        port_dev = bridge_dev / fake_sysfs.PORT_BDF
        assert (port_dev / "vendor").read_text().strip() == fake_sysfs.VENDOR_ID_HEX
        assert (port_dev / "device").read_text().strip() == fake_sysfs.PORT_DEV_HEX

        # All hw_counter files present
        assert hw_counters_dir.is_dir()
        for counter in fake_sysfs.HW_COUNTERS:
            assert (hw_counters_dir / counter).exists(), f"Missing counter: {counter}"

        # Net interface and device symlink
        net_iface = fake_sysfs_root / "sys/class/net" / fake_sysfs.IFACE
        assert net_iface.is_dir()
        assert (net_iface / "device").is_symlink()

        # downstream_port() check: canonical port path must contain BRIDGE_BDF
        port_link = fake_sysfs_root / "sys/bus/pci/devices" / fake_sysfs.PORT_BDF
        assert f"/{fake_sysfs.BRIDGE_BDF}/" in str(port_link.resolve()), (
            f"downstream_port() check would fail: '{fake_sysfs.BRIDGE_BDF}' "
            f"not in '{port_link.resolve()}'"
        )

    def test_nic_simulator_increments_counters(
        self, nic_simulator: subprocess.Popen, hw_counters_dir: Path
    ) -> None:
        """Verify the NIC simulator is actually incrementing hw_counter values."""
        counter_path = hw_counters_dir / "rx_rdma_ucast_bytes"
        before = int(counter_path.read_text().strip())
        time.sleep(0.3)
        after = int(counter_path.read_text().strip())
        assert after > before, (
            f"rx_rdma_ucast_bytes did not increase: before={before}, after={after}"
        )

    @pytest.mark.skipif(
        _BINARY is None,
        reason="amd_smi_ainic_info not found — build amdsmi first "
        "(cmake --build <build_dir> --target amd_smi_ainic_info)",
    )
    def test_amdsmi_reads_simulated_nic(
        self, fake_sysfs_root: Path, nic_simulator: subprocess.Popen
    ) -> None:
        """
        Run amd_smi_ainic_info with the fake sysfs and verify:
          - exits 0
          - reports exactly 1 AI NIC device
          - lists RDMA hw_counter names
          - at least one RDMA counter value is non-zero
        """
        env = os.environ.copy()
        env["SMI_NIC_SYSFS_ROOT"] = str(fake_sysfs_root)

        result = subprocess.run([str(_BINARY)], env=env, capture_output=True, text=True, timeout=15)

        assert result.returncode == 0, (
            f"amd_smi_ainic_info failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        output = result.stdout

        # Exactly 1 AI NIC device discovered
        assert "Total AI NIC devices found: 1" in output, (
            f"Expected exactly 1 AI NIC device.\nstdout:\n{output}"
        )

        # RDMA counter names appear in the output
        expected_counters = [
            "rx_rdma_ucast_bytes",
            "tx_rdma_ucast_bytes",
            "rx_rdma_ucast_pkts",
            "tx_rdma_ucast_pkts",
            "rx_rdma_cnp_pkts",
            "tx_rdma_cnp_pkts",
            "tx_rdma_ack_timeout",
            "resp_tx_pkt_seq_err",
            "req_rx_pkt_seq_err",
            "req_rx_impl_nak_seq_err",
        ]
        missing = [c for c in expected_counters if c not in output]
        assert not missing, f"RDMA counter names missing from output: {missing}\nstdout:\n{output}"

        # At least one counter value is non-zero
        nonzero = False
        for line in output.splitlines():
            for counter in expected_counters:
                if counter in line and ":" in line:
                    try:
                        value = int(line.split(":")[-1].strip())
                        if value > 0:
                            nonzero = True
                            break
                    except ValueError:
                        pass
            if nonzero:
                break

        assert nonzero, (
            "All RDMA counter values are zero — the simulator may not have "
            "been running long enough or its updates were not picked up.\n"
            f"stdout:\n{output}"
        )
