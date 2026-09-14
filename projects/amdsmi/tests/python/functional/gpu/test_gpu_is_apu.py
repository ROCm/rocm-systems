# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only checks for APU identification and typed GPU discovery."""

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
                    is_apu = amdsmi.amdsmi_get_gpu_is_apu(handle)
                except amdsmi.AmdSmiLibraryException as error:
                    if error.get_error_code() == amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                        continue
                    raise
                supported += 1
                asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
                self.assertIsInstance(is_apu, bool)
                self.assertNotEqual(asic["flags"], 0xFFFFFFFFFFFFFFFF)
                self.assertEqual(is_apu, bool(asic["flags"] & 1))
                self.assertEqual(
                    amdsmi.amdsmi_get_processor_type(handle)["processor_type"], "AMD_GPU"
                )
        if not supported:
            self.skipTest("Backend does not provide APU identification")

    def test_invalid_arguments_preserve_output(self):
        wrapper = amdsmi.amdsmi_wrapper
        if not hasattr(wrapper, "amdsmi_get_gpu_is_apu"):
            self.skipTest("Library predates APU identification")
        self.assertEqual(
            wrapper.amdsmi_get_gpu_is_apu(self.handles[0], None), wrapper.AMDSMI_STATUS_INVAL
        )
        for original in (False, True):
            result = ctypes.c_bool(original)
            status = wrapper.amdsmi_get_gpu_is_apu(
                wrapper.amdsmi_processor_handle(), ctypes.byref(result)
            )
            self.assertNotEqual(status, wrapper.AMDSMI_STATUS_SUCCESS)
            self.assertIs(result.value, original)

    def test_typed_enumeration_count_and_list_agree(self):
        wrapper = amdsmi.amdsmi_wrapper
        for socket in amdsmi.amdsmi_get_socket_handles():
            for processor_type in amdsmi.AmdSmiProcessorType:
                with self.subTest(socket=socket.value, processor_type=processor_type):
                    count = ctypes.c_uint32()
                    status = wrapper.amdsmi_get_processor_handles_by_type(
                        socket, processor_type, None, ctypes.byref(count)
                    )
                    self.assertEqual(status, wrapper.AMDSMI_STATUS_SUCCESS)
                    handles = (wrapper.amdsmi_processor_handle * (count.value + 1))()
                    handles[count.value] = 1  # Guard against writes past the returned count.
                    capacity = ctypes.c_uint32(count.value)
                    status = wrapper.amdsmi_get_processor_handles_by_type(
                        socket, processor_type, handles, ctypes.byref(capacity)
                    )
                    self.assertEqual(status, wrapper.AMDSMI_STATUS_SUCCESS)
                    self.assertEqual(capacity.value, count.value)
                    self.assertEqual(handles[count.value], 1)
                    for raw_handle in handles[: count.value]:
                        actual = amdsmi.amdsmi_get_processor_type(
                            wrapper.amdsmi_processor_handle(raw_handle)
                        )
                        self.assertEqual(actual["processor_type"], processor_type.name)

            count = ctypes.c_uint32()
            self.assertEqual(
                wrapper.amdsmi_get_processor_handles_by_type(socket, 10, None, ctypes.byref(count)),
                wrapper.AMDSMI_STATUS_INVAL,
            )


if __name__ == "__main__":
    unittest.main()
