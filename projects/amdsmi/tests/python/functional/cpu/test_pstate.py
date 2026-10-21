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
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
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
                msg = f"\t### amdsmi_get_cpu_apb_status(cpu={i}):"
                val = "N/A"
                with self.common.expect_status(msg, correct_processor):
                    val = amdsmi.amdsmi_get_cpu_apb_status(cpu)
                self.common.print(f"\t    data: {val}")
                # A SUCCESS must also carry sane data -- status alone would let
                # swapped or garbage out-params (eg. apb_disabled == 7) pass.
                if isinstance(val, dict):
                    assert val["apb_disabled"] in (0, 1), (
                        f"apb_disabled must be 0 or 1, got {val['apb_disabled']}"
                    )
                    if val["apb_disabled"] == 1:
                        assert 0 <= val["pstate"] <= 7, (
                            f"pstate out of range [0, 7], got {val['pstate']}"
                        )

            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            msg = "\t### amdsmi_get_cpu_apb_status(null handle):"
            val = "N/A"
            with self.common.expect_status(msg, null_check):
                val = amdsmi.amdsmi_get_cpu_apb_status(null_handle)
            self.common.print(f"\t    data: {val}")
        return

    def test_set_cpu_df_pstate_range(self):
        self.common.print_func_name("")

        # amdsmi_set_cpu_df_pstate_range derives the socket index from the handle's
        # processor index and forwards it to ESMI; it does not validate that the
        # handle is a CPU. A GPU handle would reuse its index as a socket index and
        # reprogram a real socket's DF P-state range on a combined GPU + EPYC host,
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
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,
            amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
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
                msg = f"\t### amdsmi_get_cpu_df_pstate_range(cpu={i}):"
                val = "N/A"
                with self.common.expect_status(msg, correct_processor):
                    val = amdsmi.amdsmi_get_cpu_df_pstate_range(cpu)
                self.common.print(f"\t    data: {val}")
                # A SUCCESS must also carry sane data -- status alone would let
                # out-of-range P-states pass. Note DF P-states are numbered
                # with P0 as the highest-performance state, so min/max are not
                # numerically ordered (eg. a valid result is min_pstate=2,
                # max_pstate=0); only the [0, 7] range is asserted.
                if isinstance(val, dict):
                    assert 0 <= val["min_pstate"] <= 7 and 0 <= val["max_pstate"] <= 7, (
                        f"df pstate values out of range [0, 7]: {val}"
                    )

            null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
            msg = "\t### amdsmi_get_cpu_df_pstate_range(null handle):"
            val = "N/A"
            with self.common.expect_status(msg, null_check):
                val = amdsmi.amdsmi_get_cpu_df_pstate_range(null_handle)
            self.common.print(f"\t    data: {val}")
        return
