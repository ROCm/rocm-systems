# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""APU identification binding contract, without requiring GPU hardware."""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi


class TestGpuIsApu(unittest.TestCase):
    def setUp(self):
        self.handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle(1)

    def test_export_and_signature(self):
        self.assertTrue(callable(amdsmi.amdsmi_get_gpu_is_apu))
        function = amdsmi.amdsmi_wrapper.amdsmi_get_gpu_is_apu
        self.assertEqual(
            function.argtypes,
            [amdsmi.amdsmi_wrapper.amdsmi_processor_handle, ctypes.POINTER(ctypes.c_bool)],
        )
        self.assertIs(function.restype, amdsmi.amdsmi_wrapper.amdsmi_status_t)

    def test_returns_boolean_for_apu_and_discrete(self):
        for expected in (True, False):
            with self.subTest(is_apu=expected):

                def query(handle, output):
                    self.assertEqual(handle.value, self.handle.value)
                    ctypes.cast(output, ctypes.POINTER(ctypes.c_bool))[0] = expected
                    return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

                with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_is_apu", query):
                    self.assertIs(amdsmi.amdsmi_get_gpu_is_apu(self.handle), expected)

    def test_errors_are_not_reported_as_discrete(self):
        for status in (
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRM_ERROR,
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL,
        ):
            with self.subTest(status=status):
                with mock.patch.object(
                    amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_is_apu", return_value=status
                ):
                    with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                        amdsmi.amdsmi_get_gpu_is_apu(self.handle)
                    self.assertEqual(error.exception.get_error_code(), status)

    def test_old_library_without_symbol_is_not_supported(self):
        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_is_apu"):
            del amdsmi.amdsmi_wrapper.amdsmi_get_gpu_is_apu
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                amdsmi.amdsmi_get_gpu_is_apu(self.handle)
            self.assertEqual(
                error.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
            )

    def test_rejects_wrong_handle_types_without_calling_library(self):
        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_is_apu") as query:
            for handle in (None, 1, "gpu", object()):
                with self.subTest(handle=handle):
                    with self.assertRaises(amdsmi.AmdSmiParameterException):
                        amdsmi.amdsmi_get_gpu_is_apu(handle)
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
