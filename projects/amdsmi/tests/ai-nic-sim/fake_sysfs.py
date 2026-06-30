# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
fake_sysfs.py — build a minimal fake sysfs tree that fools amdsmi into
thinking one AMD Pensando Pollara (AI NIC) card is present.

Directory layout mirrors real Linux sysfs.  Symlinks are created to
satisfy every traversal that amdsmi_unified and amd_smi.cc perform:

  - SmiNicSubsystemPensando::discover()   reads pci_path for vendor/device IDs
  - SmiNicSubsystem::resolve_bdf()        follows class/net/<iface>/device symlink
  - SmiNicSubsystemPensando::downstream_port()  canonicalises pci symlink
  - SmiNicSubsystemPensando::driver_loaded()    checks pci/drivers/ionic and
                                                auxiliary/drivers/ionic_rdma.rdma
  - SmiNicPort::discover_infiniband()     reads <pci_bdf>/infiniband/...
  - SmiInfiniBandPort::collect_hw_counters()  reads ports/<N>/hw_counters/*
  - amdsmi_get_nic_rdma_port_statistics() re-reads hw_counters on every call
    via: class/net/<iface>/device/infiniband/<ib>/subsystem/<ib>/subsystem/<ib>/ports/<N>/hw_counters/

Usage (command line):
    python3 fake_sysfs.py <SIM_ROOT>

Prints the hw_counters directory path to stdout so callers can capture it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulated hardware identifiers
# ---------------------------------------------------------------------------

BRIDGE_BDF = "0000:e2:00.0"  # Pensando PCIe bridge (vendor=0x1dd8, device=0x0008)
PORT_BDF = "0000:e2:00.1"  # Pensando Ethernet/IB port (vendor=0x1dd8, device=0x1002)
IFACE = "enp226s0"  # Linux net interface name
IB_DEV = "rocep226s0"  # InfiniBand device name
IB_PORT = "1"  # InfiniBand port number (directory name under ports/)

VENDOR_ID_HEX = "0x1dd8"
BRIDGE_DEV_HEX = "0x0008"
PORT_DEV_HEX = "0x1002"

HW_COUNTERS = [
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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _symlink(link: Path, target: str) -> None:
    """Create symlink, replacing any existing one."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def create(root: Path) -> Path:
    """
    Populate *root* with the fake sysfs tree and return the path to the
    hw_counters directory (the one the simulator will keep updating).
    """
    root = root.resolve()

    bridge_dev = root / "sys/devices/pci0000:e0" / BRIDGE_BDF
    bridge_dev.mkdir(parents=True, exist_ok=True)
    _write(bridge_dev / "vendor", VENDOR_ID_HEX)
    _write(bridge_dev / "device", BRIDGE_DEV_HEX)
    _write(bridge_dev / "subsystem_vendor", VENDOR_ID_HEX)
    _write(bridge_dev / "subsystem_device", "0x0000")
    _write(bridge_dev / "revision", "0x00")
    _write(bridge_dev / "class", "0x060400")
    _write(bridge_dev / "max_link_width", "16")
    _write(bridge_dev / "max_link_speed", "16")
    _write(bridge_dev / "numa_node", "0")

    port_dev = bridge_dev / PORT_BDF
    port_dev.mkdir(parents=True, exist_ok=True)
    _write(port_dev / "vendor", VENDOR_ID_HEX)
    _write(port_dev / "device", PORT_DEV_HEX)
    _write(port_dev / "subsystem_vendor", VENDOR_ID_HEX)
    _write(port_dev / "subsystem_device", "0x0000")
    _write(port_dev / "revision", "0x00")
    _write(port_dev / "numa_node", "0")

    ib_dev_dir = port_dev / "infiniband" / IB_DEV
    ib_dev_dir.mkdir(parents=True, exist_ok=True)
    _write(ib_dev_dir / "node_guid", "0xaabbccddeeff0011")
    _write(ib_dev_dir / "node_type", "1: CA")
    _write(ib_dev_dir / "sys_image_guid", "0xaabbccddeeff0011")
    _write(ib_dev_dir / "fw_ver", "1.15.0")

    ib_port_dir = ib_dev_dir / "ports" / IB_PORT
    ib_port_dir.mkdir(parents=True, exist_ok=True)
    _write(ib_port_dir / "state", "4: ACTIVE")
    _write(ib_port_dir / "max_mtu", "4096")
    _write(ib_port_dir / "active_mtu", "4096")

    hw_counters_dir = ib_port_dir / "hw_counters"
    hw_counters_dir.mkdir(parents=True, exist_ok=True)
    for counter in HW_COUNTERS:
        _write(hw_counters_dir / counter, "0")

    rdma_aux_dev = port_dev / "ionic_rdma.rdma.0"
    rdma_aux_dev.mkdir(parents=True, exist_ok=True)

    net_iface_dir = root / "sys/class/net" / IFACE
    net_iface_dir.mkdir(parents=True, exist_ok=True)
    _write(net_iface_dir / "address", "aa:bb:cc:dd:ee:ff")
    _write(net_iface_dir / "dev_port", "0")
    _write(net_iface_dir / "type", "32")
    _write(net_iface_dir / "ifindex", "5")
    _write(net_iface_dir / "carrier", "1")
    _write(net_iface_dir / "mtu", "4096")
    _write(net_iface_dir / "operstate", "up")
    _write(net_iface_dir / "speed", "100000")
    stats_dir = net_iface_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    _write(stats_dir / "rx_bytes", "0")
    _write(stats_dir / "tx_bytes", "0")
    _symlink(net_iface_dir / "device", "../../../devices/pci0000:e0/" + BRIDGE_BDF + "/" + PORT_BDF)

    ib_class_dir = root / "sys/class/infiniband"
    ib_class_dir.mkdir(parents=True, exist_ok=True)
    _symlink(
        ib_class_dir / IB_DEV,
        "../../devices/pci0000:e0/" + BRIDGE_BDF + "/" + PORT_BDF + "/infiniband/" + IB_DEV,
    )
    _symlink(ib_dev_dir / "subsystem", "../../../../../../class/infiniband")

    pci_dev_dir = root / "sys/bus/pci/devices"
    pci_dev_dir.mkdir(parents=True, exist_ok=True)
    _symlink(pci_dev_dir / BRIDGE_BDF, "../../../devices/pci0000:e0/" + BRIDGE_BDF)
    _symlink(pci_dev_dir / PORT_BDF, "../../../devices/pci0000:e0/" + BRIDGE_BDF + "/" + PORT_BDF)

    ionic_driver_dir = root / "sys/bus/pci/drivers/ionic"
    ionic_driver_dir.mkdir(parents=True, exist_ok=True)
    _symlink(
        ionic_driver_dir / PORT_BDF, "../../../../devices/pci0000:e0/" + BRIDGE_BDF + "/" + PORT_BDF
    )

    rdma_driver_dir = root / "sys/bus/auxiliary/drivers/ionic_rdma.rdma"
    rdma_driver_dir.mkdir(parents=True, exist_ok=True)
    _symlink(
        rdma_driver_dir / "ionic_rdma.rdma.0",
        "../../../../devices/pci0000:e0/" + BRIDGE_BDF + "/" + PORT_BDF + "/ionic_rdma.rdma.0",
    )

    _write(root / "sys/devices/system/node/node0/cpulist", "0-63")

    return hw_counters_dir


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: {sys.argv[0]} <SIM_ROOT>", file=sys.stderr)
        print("  Creates a fake AI NIC sysfs tree under SIM_ROOT and", file=sys.stderr)
        print("  prints the hw_counters directory path to stdout.", file=sys.stderr)
        sys.exit(0 if sys.argv[1:] == ["--help"] else 1)

    hw_counters_dir = create(Path(sys.argv[1]))
    print(hw_counters_dir)


if __name__ == "__main__":
    main()
