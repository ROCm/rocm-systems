#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU P-state: APB, DF P-state range."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuPstate(unittest.TestCase):
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

    def test_cpu_apb_disable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_disable=amdsmi.amdsmi_cpu_apb_disable, pstate=0)
        return

    def test_cpu_apb_enable(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_cpu_apb_enable=amdsmi.amdsmi_cpu_apb_enable)
        return

    def test_get_cpu_apb_status(self):
        self.common.print_func_name("")

        # amdsmi_get_cpu_apb_status derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle therefore reuses its index as a socket
        # index, which on a combined GPU + EPYC host aliases a real socket and can
        # legitimately succeed -- so there is no well-defined status to assert for
        # a GPU handle, and that sweep is intentionally omitted.
        correct_processor = [
            amdsmi.AmdSmiStatus.SUCCESS,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,  # firmware/protocol lacks the message (ENOMSG)
            amdsmi.AmdSmiStatus.NO_PERM,  # no read access to /dev/hsmp (mode/LSM/container)
        ]
        # A null (default-constructed) handle hits the C-layer null check -> INVAL.
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        # Independent sections: a bug in one (anything expect_status re-raises, ie.
        # not a status-carrying amdsmi exception) must not stop the others running.
        failures = []

        # The null and None checks need no CPU socket: the native library rejects a
        # null handle (INVAL) and the Python wrapper rejects a non-handle type
        # before any native call, so both run on hosts with no CPU/HSMP driver.
        try:
            msg = "\t### amdsmi_get_cpu_apb_status(cpu=null handle):"
            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            with self.common.expect_status(msg, null_check):
                amdsmi.amdsmi_get_cpu_apb_status(null_handle)
        except Exception as e:
            failures.append(e)

        try:
            msg = "\t### amdsmi_get_cpu_apb_status(cpu=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_cpu_apb_status(None)
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

        # status_sweep() collects every expect_status failure in the block and
        # raises once at the end.
        values = []
        try:
            with self.common.status_sweep():
                for i, cpu in enumerate(cpu_handles):
                    msg = f"\t### amdsmi_get_cpu_apb_status(cpu={i}):"
                    val = "N/A"
                    with self.common.expect_status(msg, correct_processor):
                        val = amdsmi.amdsmi_get_cpu_apb_status(cpu)
                    self.common.print(f"\t    data: {val}")
                    values.append((i, val))
        except Exception as e:
            failures.append(e)

        # A SUCCESS must also carry sane data -- status alone would let
        # swapped or garbage out-params (eg. apb_disabled == 7) pass.
        for i, val in values:
            try:
                if isinstance(val, dict):
                    assert val["apb_disabled"] in (0, 1), (
                        f"cpu={i}: apb_disabled must be 0 or 1, got {val['apb_disabled']}"
                    )
                    if val["apb_disabled"] == 1:
                        assert 0 <= val["pstate"] <= 7, (
                            f"cpu={i}: pstate out of range [0, 7], got {val['pstate']}"
                        )
            except Exception as e:
                failures.append(e)
        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        # amdsmi_set_cpu_df_pstate_range derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle would reuse its index as a socket index and
        # reprogram a real socket's DF P-state range on a combined GPU + EPYC host,
        # so that sweep is intentionally omitted.
        # A write-back of a just-read value is valid input, so INVAL is not accepted.
        correct_processor = [
            amdsmi.AmdSmiStatus.SUCCESS,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,  # firmware/protocol lacks the message (ENOMSG)
            amdsmi.AmdSmiStatus.NO_PERM,  # no write access to /dev/hsmp (root/LSM/container)
        ]
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        # Independent sections: a bug in one (anything expect_status re-raises, ie.
        # not a status-carrying amdsmi exception) must not stop the others running.
        failures = []

        # The null and None checks need no CPU socket: the native library rejects a
        # null handle (INVAL) and the Python wrapper rejects a non-handle type
        # before any native call, so both run on hosts with no CPU/HSMP driver.
        try:
            msg = "\t### amdsmi_set_cpu_df_pstate_range(cpu=null handle):"
            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            with self.common.expect_status(msg, null_check):
                amdsmi.amdsmi_set_cpu_df_pstate_range(null_handle, 0, 0)
        except Exception as e:
            failures.append(e)

        try:
            msg = "\t### amdsmi_set_cpu_df_pstate_range(cpu=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_cpu_df_pstate_range(None, 0, 0)
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
                    # Write back the current range so the test never changes the
                    # socket's DF P-state configuration.
                    try:
                        current = amdsmi.amdsmi_get_cpu_df_pstate_range(cpu)
                    except amdsmi.AmdSmiLibraryException as e:
                        self.common.print(
                            f"\t### amdsmi_get_cpu_df_pstate_range(cpu={i}): {e}; skipping set"
                        )
                        continue
                    msg = (
                        f"\t### amdsmi_set_cpu_df_pstate_range(cpu={i}, "
                        f"min_pstate={current['min_pstate']}, max_pstate={current['max_pstate']}):"
                    )
                    with self.common.expect_status(msg, correct_processor):
                        amdsmi.amdsmi_set_cpu_df_pstate_range(
                            cpu, current["min_pstate"], current["max_pstate"]
                        )
        except Exception as e:
            failures.append(e)

        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return

    def test_get_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        # amdsmi_get_cpu_df_pstate_range derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle therefore reuses its index as a socket
        # index, which on a combined GPU + EPYC host aliases a real socket and can
        # legitimately succeed -- so there is no well-defined status to assert for
        # a GPU handle, and that sweep is intentionally omitted.
        correct_processor = [
            amdsmi.AmdSmiStatus.SUCCESS,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,  # firmware/protocol lacks the message (ENOMSG)
            amdsmi.AmdSmiStatus.NO_PERM,  # no read access to /dev/hsmp (mode/LSM/container)
        ]
        # A null (default-constructed) handle hits the C-layer null check -> INVAL.
        null_check = [amdsmi.AmdSmiStatus.INVAL]

        # Independent sections: a bug in one (anything expect_status re-raises, ie.
        # not a status-carrying amdsmi exception) must not stop the others running.
        failures = []

        # The null and None checks need no CPU socket: the native library rejects a
        # null handle (INVAL) and the Python wrapper rejects a non-handle type
        # before any native call, so both run on hosts with no CPU/HSMP driver.
        try:
            msg = "\t### amdsmi_get_cpu_df_pstate_range(cpu=null handle):"
            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            with self.common.expect_status(msg, null_check):
                amdsmi.amdsmi_get_cpu_df_pstate_range(null_handle)
        except Exception as e:
            failures.append(e)

        try:
            msg = "\t### amdsmi_get_cpu_df_pstate_range(cpu=None):"
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_cpu_df_pstate_range(None)
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

        # status_sweep() collects every expect_status failure in the block and
        # raises once at the end.
        values = []
        try:
            with self.common.status_sweep():
                for i, cpu in enumerate(cpu_handles):
                    msg = f"\t### amdsmi_get_cpu_df_pstate_range(cpu={i}):"
                    val = "N/A"
                    with self.common.expect_status(msg, correct_processor):
                        val = amdsmi.amdsmi_get_cpu_df_pstate_range(cpu)
                    self.common.print(f"\t    data: {val}")
                    values.append((i, val))
        except Exception as e:
            failures.append(e)

        # A SUCCESS must also carry sane data -- status alone would let
        # out-of-range P-states pass. Per ESMI (esmi_df_pstate_range), DF
        # P-states are in [0, 2] with P0 as the highest-performance state,
        # so the range is numerically inverted: max_pstate <= min_pstate
        # (eg. a valid result is min_pstate=2, max_pstate=0).
        for i, val in values:
            try:
                if isinstance(val, dict):
                    min_pstate, max_pstate = val["min_pstate"], val["max_pstate"]
                    self.assertLessEqual(
                        0, max_pstate, f"cpu={i}: df max_pstate out of range [0, 2]: {val}"
                    )
                    self.assertLessEqual(
                        max_pstate,
                        min_pstate,
                        f"cpu={i}: df max_pstate must be <= min_pstate: {val}",
                    )
                    self.assertLessEqual(
                        min_pstate, 2, f"cpu={i}: df min_pstate out of range [0, 2]: {val}"
                    )
            except Exception as e:
                failures.append(e)
        if failures:
            raise AssertionError("\n\n".join(str(f) for f in failures)) from failures[0]
        return
