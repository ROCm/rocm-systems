#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only APU identification checks against live hardware."""

import ctypes
import unittest

from common.common import amdsmi


class TestGpuIsApuReadOnly(unittest.TestCase):
    def setUp(self):
        amdsmi.amdsmi_init()
        self.addCleanup(amdsmi.amdsmi_shut_down)
        self.handles = amdsmi.amdsmi_get_processor_handles()
        if not self.handles:
            self.skipTest("No AMD GPU available")

    def test_identification_matches_native_asic_flag(self):
        supported = 0
        for handle in self.handles:
            with self.subTest(handle=handle.value):
                try:
                    is_apu = amdsmi.amdsmi_is_gpu_apu(handle)
                except amdsmi.AmdSmiLibraryException as error:
                    if error.get_error_code() == amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                        continue
                    raise
                supported += 1
                asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
                self.assertIsInstance(is_apu, bool)
                # Checks the value reaches Python from the same flags word, not the bit test.
                self.assertEqual(is_apu, bool(asic["flags"] & 1))
                self.assertEqual(
                    amdsmi.amdsmi_get_processor_type(handle)["processor_type"], "AMD_GPU"
                )
        if not supported:
            self.skipTest("Backend does not provide APU identification")

    def test_invalid_arguments_preserve_output(self):
        wrapper = amdsmi.amdsmi_wrapper
        if not hasattr(wrapper, "amdsmi_is_gpu_apu"):
            self.skipTest("Library predates APU identification")
        self.assertEqual(
            wrapper.amdsmi_is_gpu_apu(self.handles[0], None), wrapper.AMDSMI_STATUS_INVAL
        )
        for original in (False, True):
            result = ctypes.c_bool(original)
            status = wrapper.amdsmi_is_gpu_apu(
                wrapper.amdsmi_processor_handle(), ctypes.byref(result)
            )
            self.assertEqual(status, wrapper.AMDSMI_STATUS_INVAL)
            self.assertIs(result.value, original)

    def test_non_gpu_handle_is_not_supported(self):
        wrapper = amdsmi.amdsmi_wrapper
        if not hasattr(wrapper, "amdsmi_is_gpu_apu"):
            self.skipTest("Library predates APU identification")
        non_gpu = []
        for socket in amdsmi.amdsmi_get_socket_handles():
            for processor_type in (
                amdsmi.AmdSmiProcessorType.AMD_CPU,
                amdsmi.AmdSmiProcessorType.AMD_CPU_CORE,
                amdsmi.AmdSmiProcessorType.AMD_AINIC,
            ):
                found = amdsmi.amdsmi_get_processor_handles_by_type(socket, processor_type)
                non_gpu.extend(found["processor_handles"])
        if not non_gpu:
            self.skipTest("No non-GPU processors enumerated")
        for handle in non_gpu[:4]:
            with self.subTest(handle=handle.value):
                result = ctypes.c_bool(True)
                self.assertEqual(
                    wrapper.amdsmi_is_gpu_apu(handle, ctypes.byref(result)),
                    wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
                )
                self.assertIs(result.value, True)


if __name__ == "__main__":
    unittest.main()
