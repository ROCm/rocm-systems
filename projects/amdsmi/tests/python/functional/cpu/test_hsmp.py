#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU HSMP: driver version, protocol version, DDR bandwidth, ESMI error messages, metrics table."""

import unittest

import common.common as common
from common.common import amdsmi

# amdsmi_get_esmi_err_msg() reads its argument as an esmi_status_t code and returns the
# amdsmi status that code maps to (esmi_status_map in amd_smi_common.h), so an input
# inside the esmi enum range comes back as that mapped status rather than SUCCESS.
# Keyed by esmi_status_t ordinal (e_smi.h).
_ESMI_CODE_TO_STATUS = {
    0: amdsmi.AmdSmiStatus.SUCCESS,  # ESMI_SUCCESS / ESMI_INITIALIZED
    1: amdsmi.AmdSmiStatus.NO_ENERGY_DRV,
    2: amdsmi.AmdSmiStatus.NO_MSR_DRV,
    3: amdsmi.AmdSmiStatus.NO_HSMP_DRV,
    4: amdsmi.AmdSmiStatus.NO_HSMP_SUP,
    5: amdsmi.AmdSmiStatus.NO_DRV,
    6: amdsmi.AmdSmiStatus.FILE_NOT_FOUND,
    7: amdsmi.AmdSmiStatus.BUSY,
    8: amdsmi.AmdSmiStatus.NO_PERM,
    9: amdsmi.AmdSmiStatus.NOT_SUPPORTED,
    10: amdsmi.AmdSmiStatus.FILE_ERROR,
    11: amdsmi.AmdSmiStatus.INTERRUPT,
    12: amdsmi.AmdSmiStatus.IO,
    13: amdsmi.AmdSmiStatus.UNEXPECTED_SIZE,
    14: amdsmi.AmdSmiStatus.UNKNOWN_ERROR,
    15: amdsmi.AmdSmiStatus.ARG_PTR_NULL,
    16: amdsmi.AmdSmiStatus.OUT_OF_RESOURCES,
    17: amdsmi.AmdSmiStatus.NOT_INIT,
    18: amdsmi.AmdSmiStatus.INVAL,
    19: amdsmi.AmdSmiStatus.HSMP_TIMEOUT,
    20: amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
}


class TestCpuHsmp(unittest.TestCase):
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
        self.common.amdsmi_smart_init()

    def test_get_cpu_ddr_bw(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_ddr_bw,
            "amdsmi_get_cpu_ddr_bw",
            wrong_accept=[
                # NOT_SUPPORTED: a GPU handle reaches the CPU path, which reports the feature
                # absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: a GPU handle reaches the HSMP message check, which reports the
                # message absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                # NOT_FOUND: a GPU handle is not a CPU socket, so the lookup fails. The CPU path has
                # no processor-type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.NOT_FOUND,
                # INVAL: a GPU handle is rejected as invalid input. The CPU path has no processor-
                # type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_cpu_hsmp_driver_version(self):
        self.common.print_func_name("")
        # Returns the global HSMP driver version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_hsmp_driver_version,
            "amdsmi_get_cpu_hsmp_driver_version",
            check_wrong_handle=False,
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_cpu_hsmp_proto_ver(self):
        self.common.print_func_name("")
        # Returns the global HSMP protocol version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_cpu_hsmp_proto_ver,
            "amdsmi_get_cpu_hsmp_proto_ver",
            check_wrong_handle=False,
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_esmi_err_msg(self):
        self.common.print_func_name("")
        failures = []
        described = []
        wrong_text = []
        accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]

        try:  # NEGATIVE: the wrapper rejects non-AmdSmiStatus input
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_get_esmi_err_msg(None)
        except Exception as e:  # noqa: BLE001 - collected and re-raised together
            failures.append(f"[None input] {e}")

        try:
            with self.common.status_sweep():
                for status in amdsmi.AmdSmiStatus:
                    msg = f"\t### amdsmi_get_esmi_err_msg(status={status.name}):"
                    val = None
                    # NOT_SUPPORTED: not an ESMI status. MAP_ERROR is not accepted:
                    # an ESMI status without a description is a failure.
                    with self.common.expect_status(msg, accept):
                        val = amdsmi.amdsmi_get_esmi_err_msg(status)
                    self.common.print(f"\t    data: {val}")
                    if val is None:
                        continue
                    described.append(status)
                    # Collected, not asserted: raising here would end the sweep early
                    if f"AMDSMI_STATUS_{status.name}" not in val:
                        wrong_text.append(f"{status.name}: {val!r}")
        except Exception as e:  # noqa: BLE001 - collected and re-raised together
            failures.append(f"[status] {e}")

        if wrong_text:
            failures.append(
                "[message] description does not name its status:\n  " + "\n  ".join(wrong_text)
            )
        if amdsmi.AmdSmiStatus.SUCCESS not in described:
            failures.append("[guard] SUCCESS returned no description, so no status was exercised")

        if failures:
            raise AssertionError("\n\n".join(failures))
        return

    def test_get_hsmp_metrics_table(self):
        self.common.print_func_name("")
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_hsmp_metrics_table,
            "amdsmi_get_hsmp_metrics_table",
            wrong_accept=[
                # NOT_SUPPORTED: a GPU handle reaches the CPU path, which reports the feature
                # absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: a GPU handle reaches the HSMP message check, which reports the
                # message absent. Incidental; there is no processor-type check yet.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
                # NOT_FOUND: a GPU handle is not a CPU socket, so the lookup fails. The CPU path has
                # no processor-type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.NOT_FOUND,
                # INVAL: a GPU handle is rejected as invalid input. The CPU path has no processor-
                # type check yet, so this rejection is incidental.
                amdsmi.AmdSmiStatus.INVAL,
            ],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return

    def test_get_hsmp_metrics_table_version(self):
        self.common.print_func_name("")
        # Returns the global metrics-table version; the processor handle is not
        # validated, so a wrong (GPU) handle legitimately succeeds. Skip the
        # wrong-handle negative, which does not apply.
        self.common.run_cpu_api_contract(
            amdsmi.amdsmi_get_hsmp_metrics_table_version,
            "amdsmi_get_hsmp_metrics_table_version",
            check_wrong_handle=False,
            wrong_accept=[],
            correct_accept=[
                amdsmi.AmdSmiStatus.SUCCESS,
                # NOT_SUPPORTED: the platform does not provide this API, for example when no CPU
                # sockets are present or the CPU family is unsupported.
                amdsmi.AmdSmiStatus.NOT_SUPPORTED,
                # NO_HSMP_MSG_SUP: the HSMP firmware on this platform does not implement this API's
                # message.
                amdsmi.AmdSmiStatus.NO_HSMP_MSG_SUP,
            ],
            null_accept=[
                # INVAL: a null processor handle is rejected before it reaches the library.
                amdsmi.AmdSmiStatus.INVAL
            ],
        )
        return
