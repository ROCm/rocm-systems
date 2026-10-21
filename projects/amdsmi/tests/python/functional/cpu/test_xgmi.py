#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU XGMI: link width get/set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuXgmi(unittest.TestCase):
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

    def test_set_cpu_xgmi_width(self):
        self.common.print_func_name("")

        # amdsmi_set_cpu_xgmi_width derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle would reuse its index as a socket index and
        # reprogram a real socket's XGMI link width on a combined GPU + EPYC host,
        # so that sweep is intentionally omitted.
        # Writing back a value just read is valid input, so only these are correct:
        # NO_HSMP_MSG_SUP (ENOMSG: message not implemented) and NO_PERM (EACCES/EPERM
        # opening /dev/hsmp for write).
        correct_processor = [
            amdsmi.AmdSmiStatus.SUCCESS,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            amdsmi.AmdSmiStatus.NO_PERM,
        ]
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        # Independent sections: a bug in one (anything expect_status re-raises, ie.
        # not a status-carrying amdsmi exception) must not stop the others running.
        failures = []

        # The null and None checks need no CPU socket: the native library rejects a
        # null handle (INVAL) and the Python wrapper rejects a non-handle type
        # before any native call, so both run on hosts with no CPU/HSMP driver.
        try:
            msg = "\t### amdsmi_set_cpu_xgmi_width(cpu=null handle):"
            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            with self.common.expect_status(msg, null_check):
                amdsmi.amdsmi_set_cpu_xgmi_width(null_handle, 0, 0)
        except Exception as e:
            failures.append(e)

        try:
            msg = "\t### amdsmi_set_cpu_xgmi_width(cpu=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_cpu_xgmi_width(None, 0, 0)
            self.common.print(f"{msg}\n\tTEST SUCCESS, raised AmdSmiParameterException as expected")
        except Exception as e:
            failures.append(e)

        try:
            cpu_handles = amdsmi.amdsmi_get_cpu_handles()["processor_handles"]
        except amdsmi.AmdSmiLibraryException:
            cpu_handles = []
        self.common.print(f"\t### amdsmi_get_cpu_handles(): cpu_count={len(cpu_handles)}")
        if not cpu_handles:
            self.common.print("\tNo CPU sockets on machine, unable to test with valid cpu handles.")

        try:
            with self.common.status_sweep():
                for i, cpu in enumerate(cpu_handles):
                    # Write back the current widths so the test never changes the
                    # socket's XGMI link configuration.
                    try:
                        current = amdsmi.amdsmi_get_cpu_xgmi_width(cpu)
                    except amdsmi.AmdSmiLibraryException as e:
                        self.common.print(
                            f"\t### amdsmi_get_cpu_xgmi_width(cpu={i}): {e}; skipping set"
                        )
                        continue
                    msg = (
                        f"\t### amdsmi_set_cpu_xgmi_width(cpu={i}, "
                        f"min_width={current['min_width']}, max_width={current['max_width']}):"
                    )
                    with self.common.expect_status(msg, correct_processor):
                        amdsmi.amdsmi_set_cpu_xgmi_width(
                            cpu, current["min_width"], current["max_width"]
                        )
        except Exception as e:
            failures.append(e)

        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return

    def test_get_cpu_xgmi_width(self):
        self.common.print_func_name("")

        # amdsmi_get_cpu_xgmi_width delegates to ESMI's esmi_xgmi_width_get, which
        # is defined only for *multi-socket* systems (XGMI is the socket-to-socket
        # link). On a single-socket host there is no XGMI link, so ESMI rejects the
        # query and it surfaces as AMDSMI_STATUS_INVAL -- a legitimate result on
        # the CPU path. The API derives the socket index from the handle's
        # processor index and does not validate that the handle is a CPU, so a GPU
        # handle reuses its index as a socket index and can legitimately succeed on
        # a combined GPU + EPYC host; there is no well-defined status to assert for
        # a GPU handle, and that sweep is intentionally omitted.
        correct_processor = [
            amdsmi.AmdSmiStatus.SUCCESS,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            amdsmi.AmdSmiStatus.INVAL,
            amdsmi.AmdSmiStatus.NO_PERM,
        ]
        # A null (default-constructed) handle hits the C-layer null check -> INVAL.
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        try:
            cpu_handles = amdsmi.amdsmi_get_cpu_handles()["processor_handles"]
        except amdsmi.AmdSmiLibraryException:
            cpu_handles = []
        self.common.print(f"\t### amdsmi_get_cpu_handles(): cpu_count={len(cpu_handles)}")

        # status_sweep() collects every expect_status failure in the block and
        # raises once at the end, so a single sweep covers both handle types.
        with self.common.status_sweep():
            for i, cpu in enumerate(cpu_handles):
                msg = f"\t### amdsmi_get_cpu_xgmi_width(cpu={i}):"
                val = "N/A"
                with self.common.expect_status(msg, correct_processor):
                    val = amdsmi.amdsmi_get_cpu_xgmi_width(cpu)
                self.common.print(f"\t    data: {val}")
                # A SUCCESS must also carry sane data -- status alone would let
                # swapped min/max out-params pass.
                if isinstance(val, dict):
                    assert val["min_width"] <= val["max_width"], (
                        f"xgmi width min ({val['min_width']}) must be <= max ({val['max_width']})"
                    )

            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            msg = "\t### amdsmi_get_cpu_xgmi_width(null handle):"
            val = "N/A"
            with self.common.expect_status(msg, null_check):
                val = amdsmi.amdsmi_get_cpu_xgmi_width(null_handle)
            self.common.print(f"\t    data: {val}")
        return
