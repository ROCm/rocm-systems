#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU system operations: init, shutdown, reset, status codes, socket handles."""

import ctypes
import unittest

import common.common as common
from common.common import amdsmi


class TestGpuSystem(unittest.TestCase):
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

    def test_get_socket_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_socket_handles():"
        try:
            ret = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in ret])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_init(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_init=amdsmi.amdsmi_init)
        return

    def test_reset_gpu(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu as it fails (MI350X, Hang)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu=amdsmi.amdsmi_reset_gpu)
        return

    def test_reset_gpu_fan(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_fan=amdsmi.amdsmi_reset_gpu_fan, index=0)
        return

    def test_shut_down(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_shut_down=amdsmi.amdsmi_shut_down)
        return

    def test_status_code_to_string(self):
        self.common.print_func_name("")

        # Every amdsmi status code, including the two sentinels, must resolve to a
        # description starting with its own enum name:
        #   * AMDSMI_STATUS_UNKNOWN_ERROR -> a `case` is missing from
        #     amdsmi_status_code_to_string() (e.g. TIMEOUT / MORE_DATA).
        #   * AMDSMI_STATUS_MAP_ERROR -> a lower-level rsmi/esmi/nic status has
        #     no amdsmi mapping.
        # Either of these two errors is a library bug, which test should catch & fail.
        # Only SUCCESS is accepted: NOT_SUPPORTED or any other status is a failure here.
        accept = [amdsmi.AmdSmiStatus.SUCCESS]
        wrong_names = []
        with self.common.status_sweep():
            for status in amdsmi.AmdSmiStatus:
                error_name = f"AMDSMI_STATUS_{status.name}"
                msg = f"\t### amdsmi_status_code_to_string({error_name}={status.value}):"
                data = None
                with self.common.expect_status(msg, accept):
                    data = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(status.value))
                if data is None:
                    continue  # status failure already recorded by the sweep

                # string_cast may return bytes; normalize for comparison.
                description = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                self.common.print(f"\t\t{error_name}: {description}")
                self.common.print("")

                # Every code's description must begin with its own enum name.
                # Eg. Some code fallbacks provide an RSMI_STATUS_* string,
                # but that is not a valid AMDSMI_STATUS_* string.
                if not description.startswith(error_name):
                    wrong_names.append(f"{error_name}: '{description}'")

        self.assertEqual(
            wrong_names,
            [],
            "Descriptions that do not start with their own AMDSMI_STATUS_* name:\n  "
            + "\n  ".join(wrong_names),
        )
        return

    def test_status_code_to_string_undefined_code(self):
        self.common.print_func_name("")
        # 100 sits in an enum gap (above the highest normal code, below the two
        # high-value error codes). It has no case and no lower-level mapping, so
        # it must fall through to UNKNOWN_ERROR. SUCCESS is deliberately not
        # accepted: succeeding would mean an undefined code resolved to a string.
        accept = [amdsmi.AmdSmiStatus.UNKNOWN_ERROR]
        msg = "\t### amdsmi_status_code_to_string(status=100):"
        with self.common.expect_status(msg, accept):
            amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(100))
        self.common.print("")
        return

    def test_status_code_to_string_rejects_non_status(self):
        self.common.print_func_name("")
        # A non-status fails the Python-side isinstance check before any library
        # call happens, so it is not a status expect_status can judge.
        self.common.print('\t### amdsmi_status_code_to_string(status="not_a_status"):')
        with self.assertRaises(amdsmi.AmdSmiParameterException) as cm:
            amdsmi.amdsmi_status_code_to_string("not_a_status")
        # Confirms it is *this* isinstance check that fired, not some other
        # AmdSmiParameterException raised earlier/elsewhere for a different reason.
        self.assertEqual(cm.exception.expectedType, amdsmi.amdsmi_wrapper.amdsmi_status_t)
        self.common.print(
            f"\tTEST SUCCESS, raised {type(cm.exception).__name__}, "
            f"actual type {cm.exception.actualType.__name__}, "
            f"expected type {cm.exception.expectedType.__name__}"
        )
        return
