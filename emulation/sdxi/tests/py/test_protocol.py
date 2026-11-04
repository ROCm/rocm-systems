"""Protocol-level tests for the SDXI vfio-user emulator.

Exercises negotiation, device/region info, BAR0/BAR2 access, DMA mapping,
IRQ info, and device reset without submitting any work descriptors.

Usage (via CTest):
    ctest -R sdxi-protocol -V

Or directly:
    SDXI_SERVER_BIN=./build/rocm-sdxi python3 tests/py/test_protocol.py

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import os
import struct
import sys

from sdxi.check import Suite
from sdxi.sdxi_driver import (
    DEVID_MAGIC,
    NUM_QUEUES,
    REG_CAP,
    REG_DEVID,
    REG_NUM_QUEUES,
    REG_STATUS,
    REG_VERSION,
    STATUS_READY,
    VERSION_VAL,
    VFU_PCI_DEV_BAR0_REGION_IDX,
    VFU_PCI_DEV_BAR2_REGION_IDX,
)
from sdxi.server import Server, stop_and_check
from sdxi.vfu import (
    GuestMemory,
    VFU_DEV_MSIX_IRQ,
    VFU_PCI_DEV_CFG_REGION_IDX,
    VFU_PCI_DEV_NUM_REGIONS,
    VfioUserClient,
    VFIO_USER_F_DMA_REGION_READ,
    VFIO_USER_F_DMA_REGION_WRITE,
)

SDXI_BAR0_SIZE = 131072  # 128 KiB (expanded for CXT_L2 register at 0x10000)
SDXI_BAR2_SIZE = 4 * 1024 * 1024

PCI_VENDOR_AMD = 0x1022
PCI_DEVICE_SDXI = 0x149F


def test_negotiation(suite, vfu):
    suite.check(isinstance(vfu.server_caps, dict), "server returned caps dict")
    suite.check(len(vfu.regions) > 0, "at least one region reported")


def test_device_info(suite, vfu):
    suite.check(vfu.num_irqs > 0, "device has at least one IRQ type")
    suite.check(
        VFU_PCI_DEV_BAR0_REGION_IDX in vfu.regions, "BAR0 region present"
    )
    suite.check(
        VFU_PCI_DEV_BAR2_REGION_IDX in vfu.regions, "BAR2 region present"
    )


def test_config_space(suite, vfu):
    hdr = vfu.read_config(0, 8)
    vendor, device = struct.unpack_from("<HH", hdr)
    suite.check_eq(vendor, PCI_VENDOR_AMD, "PCI vendor ID == AMD (0x1022)")
    suite.check_eq(device, PCI_DEVICE_SDXI, "PCI device ID == 0x149f")


def test_bar0_access(suite, vfu):
    devid = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_DEVID)
    suite.check_eq(devid, DEVID_MAGIC, "DEVID == 0x53445849 ('SDXI')")

    version = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_VERSION)
    suite.check_eq(version, VERSION_VAL, "VERSION == 0x0100")

    status = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_STATUS)
    suite.check(status & STATUS_READY, "STATUS_READY bit is set")

    num_q = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_NUM_QUEUES)
    suite.check_eq(num_q, NUM_QUEUES, "NUM_QUEUES == 16")

    cap = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_CAP)
    suite.check(cap != 0, "CAP != 0 (at least one capability)")


def test_bar0_region_info(suite, vfu):
    region = vfu.regions.get(VFU_PCI_DEV_BAR0_REGION_IDX)
    suite.check(region is not None, "BAR0 region info present")
    if region is not None:
        suite.check_eq(region.size, SDXI_BAR0_SIZE, "BAR0 size == 64 KiB")


def test_bar2_region_info(suite, vfu):
    region = vfu.regions.get(VFU_PCI_DEV_BAR2_REGION_IDX)
    suite.check(region is not None, "BAR2 region info present")
    if region is not None:
        suite.check_eq(region.size, SDXI_BAR2_SIZE, "BAR2 size == 4 MiB")


def test_bar2_read_zero(suite, vfu):
    val = vfu.read_bar32(VFU_PCI_DEV_BAR2_REGION_IDX, 0)
    suite.check_eq(val, 0, "BAR2 reads return 0")


def test_dma_map_unmap(suite, vfu):
    # Use a different GPA base than the outer mem to avoid IOVA overlap
    with GuestMemory(size=4096, base=0x2000_0000, name="sdxi-dma-test") as inner:
        vfu.dma_map(inner)
        vfu.dma_unmap(inner.base, inner.size)
        suite.check(True, "dma_map + dma_unmap completed without error")


def test_irq_info(suite, vfu):
    flags, count = vfu.irq_info(VFU_DEV_MSIX_IRQ)
    suite.check(count > 0, "MSI-X IRQ count > 0 (got %d)", count)


def test_reset(suite, vfu):
    vfu.reset()
    # After reset the device must still be ready
    status = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_STATUS)
    suite.check(status & STATUS_READY, "STATUS_READY after device reset")


def test_reset_and_reconnect(suite, server):
    """Verify the server accepts a new connection after the previous one closes."""
    with server.connect() as vfu:
        devid = vfu.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_DEVID)
        suite.check_eq(devid, DEVID_MAGIC, "reconnect: DEVID readable")
    # Second reconnect to confirm the socket stays live
    with server.connect() as vfu2:
        status = vfu2.read_bar32(VFU_PCI_DEV_BAR0_REGION_IDX, REG_STATUS)
        suite.check(status & STATUS_READY, "reconnect (2nd): STATUS_READY")


def main():
    binary = os.environ.get(
        "SDXI_SERVER_BIN",
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "rocm-sdxi"),
    )
    suite = Suite("SDXI protocol tests")
    server = Server(binary)
    server.start()
    try:
        with GuestMemory(name="sdxi-guest-mem") as mem:
            with server.connect() as vfu:
                vfu.dma_map(mem)
                suite.run("negotiation", test_negotiation, vfu)
                suite.run("device_info", test_device_info, vfu)
                suite.run("config_space", test_config_space, vfu)
                suite.run("bar0_access", test_bar0_access, vfu)
                suite.run("bar0_region_info", test_bar0_region_info, vfu)
                suite.run("bar2_region_info", test_bar2_region_info, vfu)
                suite.run("bar2_read_zero", test_bar2_read_zero, vfu)
                suite.run("dma_map_unmap", test_dma_map_unmap, vfu)
                suite.run("irq_info", test_irq_info, vfu)
                suite.run("reset", test_reset, vfu)
            suite.run("reset_and_reconnect", test_reset_and_reconnect, server)
    finally:
        stop_and_check(suite, server)
    suite.finish()


if __name__ == "__main__":
    main()
