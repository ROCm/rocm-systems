#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards the lazy UALoE/IFoE session behavior.

The per-GPU IFoE/UALoE generic-netlink session used to be opened in the
AMDSmiGPUDevice constructor, so amdsmi_init() and every non-fabric query
touched the IFoE driver and hung indefinitely when that driver was wedged.
The session is now opened lazily on the first fabric query. These tests assert
that init and non-fabric queries succeed without any fabric access, and that
the first fabric query resolves cleanly (data or a normal status) rather than
blocking.

Note: the original failure needs a wedged IFoE driver (UALink hardware), which
CI cannot reproduce. On healthy hardware these run fast; the elapsed-time bound
documents the intent that these paths must not block.
"""

import time
import unittest

import common.common as common
from common.common import amdsmi

# Generous upper bound: init plus a per-GPU query completes in well under a
# second on healthy hardware. The bound documents that these paths are not
# allowed to block, without being tight enough to flake on a busy CI host.
MAX_SECONDS = 30.0


class TestFabricLazyInit(unittest.TestCase):
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
        # Skips (via SkipTest) when no compatible AMD drivers are present.
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_non_fabric_query_needs_no_fabric_session(self):
        """Init plus a non-fabric query must succeed for every GPU."""
        self.common.print_func_name("")
        if not self.common.processors:
            self.skipTest("No GPU processors detected")

        start = time.monotonic()
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # asic info is a plain sysfs/DRM read, unrelated to fabric.
            asic = amdsmi.amdsmi_get_gpu_asic_info(gpu)
            self.assertIsInstance(asic, dict)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, MAX_SECONDS, f"Non-fabric queries took {elapsed:.2f}s")

    def test_fabric_query_resolves_without_hanging(self):
        """The first fabric query must return data or a normal status, not block."""
        self.common.print_func_name("")
        if not self.common.processors:
            self.skipTest("No GPU processors detected")

        start = time.monotonic()
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            try:
                info = amdsmi.amdsmi_get_gpu_fabric_info(gpu)
                self.assertIsInstance(info, dict)
            except amdsmi.AmdSmiLibraryException:
                # Unsupported / not-a-UALink device: a clean status, not a hang.
                pass
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, MAX_SECONDS, f"Fabric queries took {elapsed:.2f}s")
