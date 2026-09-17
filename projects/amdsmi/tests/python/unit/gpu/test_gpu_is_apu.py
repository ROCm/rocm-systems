#!/usr/bin/env python3
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

    def test_returns_boolean_for_apu_and_discrete(self):
        for expected in (True, False):
            with self.subTest(is_apu=expected):

                def query(handle, output):
                    self.assertEqual(handle.value, self.handle.value)
                    ctypes.cast(output, ctypes.POINTER(ctypes.c_bool))[0] = expected
                    return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

                with mock.patch.object(
                    amdsmi.amdsmi_wrapper, "amdsmi_is_gpu_apu", query, create=True
                ):
                    self.assertIs(amdsmi.amdsmi_is_gpu_apu(self.handle), expected)

    def test_errors_are_not_reported_as_discrete(self):
        for status in (
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRM_ERROR,
            amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL,
        ):
            with self.subTest(status=status):
                with mock.patch.object(
                    amdsmi.amdsmi_wrapper, "amdsmi_is_gpu_apu", return_value=status, create=True
                ):
                    with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                        amdsmi.amdsmi_is_gpu_apu(self.handle)
                    self.assertEqual(error.exception.get_error_code(), status)

    def test_old_library_without_symbol_is_not_supported(self):
        with mock.patch.dict(amdsmi.amdsmi_wrapper.__dict__):
            amdsmi.amdsmi_wrapper.__dict__.pop("amdsmi_is_gpu_apu", None)
            with self.assertRaises(amdsmi.AmdSmiLibraryException) as error:
                amdsmi.amdsmi_is_gpu_apu(self.handle)
            self.assertEqual(
                error.exception.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
            )

    def test_rejects_wrong_handle_types_without_calling_library(self):
        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_is_gpu_apu", create=True) as query:
            for handle in (None, 1, "gpu", object()):
                with self.subTest(handle=handle):
                    with self.assertRaises(amdsmi.AmdSmiParameterException):
                        amdsmi.amdsmi_is_gpu_apu(handle)
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
