#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU ASIC: chip_rev_id and external_rev_id cross-checked against the DRM ioctl."""

import ctypes
import fcntl
import os
import struct
import unittest

import common.common as common
from common.common import amdsmi

_IOC_NRBITS = 8
_IOC_SIZESHIFT = 16
_IOC_DIRSHIFT = 30
_IOC_WRITE = 1

_DRM_IOCTL_BASE = ord("d")
_DRM_COMMAND_BASE = 0x40
_DRM_AMDGPU_INFO = 0x05
_AMDGPU_INFO_DEV_INFO = 0x16

# Leading word order of struct drm_amdgpu_info_device. Must track
# include/libdrm/amdgpu_drm.h.
_CHIP_REV_INDEX = 1
_EXTERNAL_REV_INDEX = 2
_PCI_REV_INDEX = 3


class _DrmAmdgpuInfo(ctypes.Structure):
    """struct drm_amdgpu_info from amdgpu_drm.h."""

    _fields_ = [
        ("return_pointer", ctypes.c_uint64),
        ("return_size", ctypes.c_uint32),
        ("query", ctypes.c_uint32),
        ("_union", ctypes.c_uint8 * 16),
    ]


_DRM_IOCTL_AMDGPU_INFO = (
    (_IOC_WRITE << _IOC_DIRSHIFT)
    | (_DRM_IOCTL_BASE << _IOC_NRBITS)
    | (_DRM_COMMAND_BASE + _DRM_AMDGPU_INFO)
    | (ctypes.sizeof(_DrmAmdgpuInfo) << _IOC_SIZESHIFT)
)


def _query_dev_info(render_node: str) -> tuple:
    """Return the leading words of drm_amdgpu_info_device for a render node."""
    fd = os.open(render_node, os.O_RDWR | os.O_CLOEXEC)
    try:
        buf = ctypes.create_string_buffer(1024)
        request = _DrmAmdgpuInfo()
        request.return_pointer = ctypes.addressof(buf)
        request.return_size = ctypes.sizeof(buf)
        request.query = _AMDGPU_INFO_DEV_INFO
        fcntl.ioctl(fd, _DRM_IOCTL_AMDGPU_INFO, request)
    finally:
        os.close(fd)
    return struct.unpack_from("<IIII", buf.raw, 0)


class TestGpuAsicRevisionIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_revision_ids_match_drm_dev_info(self):
        self.common.print_func_name("")
        self.assertTrue(self.common.processors, "no GPU processors enumerated")

        checked = 0
        unreachable = []
        for processor in self.common.processors:
            # drm_render maps a processor handle to its own node; a BDF lookup
            # cannot, because partitions of one GPU share a single BDF.
            try:
                render_index = amdsmi.amdsmi_get_gpu_enumeration_info(processor)["drm_render"]
                render_node = f"/dev/dri/renderD{render_index}"
                words = _query_dev_info(render_node)
            except OSError as exc:
                # Only an unreadable node is environmental; a library failure
                # must not be downgraded to a skip.
                unreachable.append(f"{processor}: {exc}")
                continue

            asic_info = amdsmi.amdsmi_get_gpu_asic_info(processor)
            expected = {
                "chip_rev_id": words[_CHIP_REV_INDEX],
                "external_rev_id": words[_EXTERNAL_REV_INDEX],
                "rev_id": words[_PCI_REV_INDEX],
            }
            for key, value in expected.items():
                reported = asic_info[key]
                self.assertNotEqual(
                    reported,
                    "N/A",
                    f"{render_node}: {key} reported N/A while the same ioctl returned {hex(value)}",
                )
                self.assertEqual(
                    int(reported, 16), value, f"{render_node}: {key} {reported} != DRM {hex(value)}"
                )
            checked += 1

        if checked == 0:
            self.skipTest(f"no render node was readable: {unreachable}")


if __name__ == "__main__":
    unittest.main()
