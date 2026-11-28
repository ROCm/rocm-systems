#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU XGMI: link width get/set."""

import os
import unittest

import common.common as common
from common.common import amdsmi


_HSMP_WRITES_ENV = "AMDSMI_TEST_ALLOW_HSMP_WRITES"


def _skip_unless_hsmp_writes_allowed(test):
    """Skip a test that issues real HSMP writes unless explicitly opted in."""
    if os.environ.get(_HSMP_WRITES_ENV) != "1":
        test.skipTest(f"issues real HSMP writes; set {_HSMP_WRITES_ENV}=1 to run")


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
        _skip_unless_hsmp_writes_allowed(self)

        # amdsmi_set_cpu_xgmi_width derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle would reuse its index as a socket index and
        # reprogram a real socket's XGMI link width on a combined GPU + EPYC host,
        # so that sweep is intentionally omitted.
        # A write-back of a just-read value is valid input, so INVAL is not accepted.
        correct_processor = [amdsmi.AmdSmiStatus.SUCCESS]
        # NO_PERM is only legitimate when /dev/hsmp really is not writable.
        if not os.access("/dev/hsmp", os.W_OK):
            correct_processor.append(amdsmi.AmdSmiStatus.NO_PERM)
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

        # Original value of every socket, restored in the finally below so a failure
        # on one socket cannot leave any socket reprogrammed.
        originals = []
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
                    originals.append((i, cpu, current))
                    msg = (
                        f"\t### amdsmi_set_cpu_xgmi_width(cpu={i}, "
                        f"min_width={current['min_width']}, max_width={current['max_width']}):"
                    )
                    wrote = False
                    with self.common.expect_status(msg, correct_processor):
                        amdsmi.amdsmi_set_cpu_xgmi_width(
                            cpu, current["min_width"], current["max_width"]
                        )
                        wrote = True
                    # A rejected write must not read as success: re-read and
                    # require the value to be unchanged.
                    if wrote:
                        self.assertEqual(
                            amdsmi.amdsmi_get_cpu_xgmi_width(cpu),
                            current,
                            f"cpu={i}: xgmi width changed after write-back",
                        )
        except Exception as e:
            failures.append(e)
        finally:
            for i, cpu, original in originals:
                try:
                    if amdsmi.amdsmi_get_cpu_xgmi_width(cpu) != original:
                        amdsmi.amdsmi_set_cpu_xgmi_width(
                            cpu, original["min_width"], original["max_width"]
                        )
                except amdsmi.AmdSmiLibraryException as e:
                    self.common.print(f"\t### restore xgmi width cpu={i} failed: {e}")

        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return

    def test_get_cpu_xgmi_width(self):
        self.common.print_func_name("")

        # amdsmi_get_cpu_xgmi_width derives the socket index from the handle's
        # processor index and does not validate that the handle is a CPU, so a GPU
        # handle reuses its index as a socket index and can legitimately succeed on
        # a combined GPU + EPYC host; there is no well-defined status to assert for
        # a GPU handle, and that sweep is intentionally omitted.
        correct_processor = [amdsmi.AmdSmiStatus.SUCCESS]
        if not os.access("/dev/hsmp", os.R_OK):
            correct_processor.append(amdsmi.AmdSmiStatus.NO_PERM)
        # A null (default-constructed) handle hits the C-layer null check -> INVAL.
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        # Independent sections: a bug in one (anything expect_status re-raises, ie.
        # not a status-carrying amdsmi exception) must not stop the others running.
        failures = []

        # The null and None checks need no CPU socket: the native library rejects a
        # null handle (INVAL) and the Python wrapper rejects a non-handle type
        # before any native call, so both run on hosts with no CPU/HSMP driver.
        try:
            msg = "\t### amdsmi_get_cpu_xgmi_width(cpu=null handle):"
            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            with self.common.expect_status(msg, null_check):
                amdsmi.amdsmi_get_cpu_xgmi_width(null_handle)
        except Exception as e:
            failures.append(e)

        try:
            msg = "\t### amdsmi_get_cpu_xgmi_width(cpu=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_cpu_xgmi_width(None)
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
        elif len(cpu_handles) == 1:
            correct_processor.append(amdsmi.AmdSmiStatus.INVAL)

        # status_sweep() collects every expect_status failure in the block and
        # raises once at the end.
        values = []
        try:
            with self.common.status_sweep():
                for i, cpu in enumerate(cpu_handles):
                    msg = f"\t### amdsmi_get_cpu_xgmi_width(cpu={i}):"
                    try:
                        proto_ver = amdsmi.amdsmi_get_cpu_hsmp_proto_ver(cpu)
                    except amdsmi.AmdSmiLibraryException:
                        proto_ver = None
                    if proto_ver is not None and proto_ver < 7:
                        with self.assertRaises(amdsmi.AmdSmiLibraryException) as context:
                            amdsmi.amdsmi_get_cpu_xgmi_width(cpu)
                        self.assertEqual(
                            context.exception.err_code, amdsmi.AmdSmiStatus.NOT_SUPPORTED
                        )
                        continue
                    val = "N/A"
                    with self.common.expect_status(msg, correct_processor):
                        val = amdsmi.amdsmi_get_cpu_xgmi_width(cpu)
                    self.common.print(f"\t    data: {val}")
                    values.append((i, val))
        except Exception as e:
            failures.append(e)

        # A SUCCESS must also carry sane data -- status alone would let
        # swapped or out-of-range min/max out-params pass. Per ESMI
        # (esmi_xgmi_width_get), the width is an encoded level in [0, 2]
        # with min <= max.
        for i, val in values:
            try:
                if isinstance(val, dict):
                    min_width, max_width = val["min_width"], val["max_width"]
                    self.assertLessEqual(
                        0, min_width, f"cpu={i}: xgmi min_width out of range [0, 2]: {val}"
                    )
                    self.assertLessEqual(
                        min_width, max_width, f"cpu={i}: xgmi min_width must be <= max_width: {val}"
                    )
                    self.assertLessEqual(
                        max_width, 2, f"cpu={i}: xgmi max_width out of range [0, 2]: {val}"
                    )
            except Exception as e:
                failures.append(e)
        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return
