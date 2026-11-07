#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU clock: SoC P-state policy, set clock frequency, set PCI bandwidth."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuClock(unittest.TestCase):
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

    def test_soc_pstate(self):
        self.common.print_func_name("")

        # Run every check even if an earlier one fails, then report all failures,
        # each labelled with the check that produced it.
        checks = [
            ("None handle", self._check_none_handle_is_rejected),
            ("null handle", self._check_null_handle_is_rejected),
            ("unregistered handle", self._check_unregistered_handle_is_rejected),
            ("non-int policy id", self._check_non_int_policy_id_is_rejected),
            ("real GPU handles", self._check_every_gpu_policy_round_trips),
        ]
        failures = []
        for name, check in checks:
            try:
                check()
            except Exception as exc:
                failures.append(f"[{name}] {common.failure_text(exc)}")

        if failures:
            raise AssertionError("\n\n".join(failures))

    @staticmethod
    def _soc_pstate_apis():
        """(name, call) for each SoC P-state API, called with only a handle."""
        return [
            ("amdsmi_get_soc_pstate", amdsmi.amdsmi_get_soc_pstate),
            ("amdsmi_set_soc_pstate", lambda handle: amdsmi.amdsmi_set_soc_pstate(handle, 0)),
        ]

    def _check_none_handle_is_rejected(self):
        # The Python wrapper rejects a non-handle before any native call.
        for name, call in self._soc_pstate_apis():
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                call(None)
            self.common.print(
                f"\t### {name}(gpu=None):\n"
                "\tTEST SUCCESS, raised AmdSmiParameterException as expected"
            )

    def _check_null_handle_is_rejected(self):
        null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        with self.common.status_sweep():
            for name, call in self._soc_pstate_apis():
                msg = f"\t### {name}(gpu=null handle):"
                with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.INVAL]):
                    call(null_handle)

    def _check_unregistered_handle_is_rejected(self):
        # A non-null pointer the library never issued.
        unregistered_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle(0xDEADBEEF)
        with self.common.status_sweep():
            for name, call in self._soc_pstate_apis():
                msg = f"\t### {name}(gpu=unregistered handle):"
                with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.NOT_FOUND]):
                    call(unregistered_handle)

    def _check_non_int_policy_id_is_rejected(self):
        # Checked by the wrapper before any native call, so it needs no real GPU.
        null_handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_soc_pstate(null_handle, "0")
        self.common.print(
            "\t### amdsmi_set_soc_pstate(policy_id='0'):\n"
            "\tTEST SUCCESS, raised AmdSmiParameterException as expected"
        )

    def _check_every_gpu_policy_round_trips(self):
        failures = []
        for i, gpu in enumerate(self.common.processors):
            try:
                self._exercise_gpu_policies(i, gpu)
            except AssertionError as exc:
                failures.append(f"gpu={i}: {exc}")
        if failures:
            raise AssertionError("\n".join(failures))

    def _exercise_gpu_policies(self, i, gpu):
        """Set every supported policy on one GPU, check each took effect, then restore."""
        # Not every ASIC exposes SoC P-state policies, so NOT_SUPPORTED just means
        # there is nothing to exercise on this GPU. Anything else is a failure.
        info = self._read_policy_info(
            i, gpu, [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
        )
        if info is None:
            return

        original_id = info["current_id"]
        try:
            for policy in info["policies"]:
                self._set_policy(i, gpu, policy["policy_id"])
                self._check_current_policy(i, gpu, expected_id=policy["policy_id"])
        finally:
            self._set_policy(i, gpu, original_id)

    def _read_policy_info(self, i, gpu, accept):
        """Returns the policy info, or None if the call raised one of the *accept* statuses."""
        msg = f"\t### amdsmi_get_soc_pstate(gpu={i}):"
        info = None
        with self.common.expect_status(msg, accept):
            info = amdsmi.amdsmi_get_soc_pstate(gpu)
        self.common.print(f"\t    data: {'N/A' if info is None else info}")
        return info

    def _set_policy(self, i, gpu, policy_id):
        msg = f"\t### amdsmi_set_soc_pstate(gpu={i}, policy_id={policy_id}):"
        with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.SUCCESS]):
            amdsmi.amdsmi_set_soc_pstate(gpu, policy_id)

    def _check_current_policy(self, i, gpu, expected_id):
        info = self._read_policy_info(i, gpu, [amdsmi.AmdSmiStatus.SUCCESS])
        if info["current_id"] != expected_id:
            raise AssertionError(
                f"set policy {expected_id}, but the current policy is {info['current_id']}"
            )

    def test_set_clk_freq(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_clk_freq as it fails (Perm failure)."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # amdsmi_set_clk_freq() accepts only these specific string names.
            # AmdSmiClkType enum names (e.g. SYS/MEM/DF/SOC) are not accepted;
            # map only the supported ones and skip the rest.
            clk_type_str_map = {"SYS": "sclk", "MEM": "mclk", "DF": "fclk", "SOC": "socclk"}
            for clk_type_name, clk_type, clk_cond in common.CLK_TYPES:
                clk_type_str = clk_type_str_map.get(clk_type_name)
                if clk_type_str is None:
                    # No string mapping for this clock type; amdsmi_set_clk_freq
                    # would raise AmdSmiParameterException before reaching the API.
                    self.common.print(
                        f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_name}): skipped (no string mapping)"
                    )
                    continue
                # Set invalid clock frequency
                try:
                    freq_bitmask = 0x1234
                    msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                    ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.FAIL)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.FAIL):
                        self.raise_exception = e

                # Get clock frequency info
                msg = f"\t### amdsmi_get_clk_freq(gpu={i}, clk_type={clk_type_name}):"
                try:
                    clk_freq_info = amdsmi.amdsmi_get_clk_freq(gpu, clk_type)
                    self.common.print(msg, clk_freq_info)
                    self.common.check_ret("", "", self.common.PASS)

                    current = clk_freq_info["current"]
                    num_supported = clk_freq_info["num_supported"]
                    frequencies = clk_freq_info["frequency"]
                    if num_supported == 0:
                        self.common.print(f"No supported frequencies for clk_type={clk_type_name}")
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, clk_cond):
                        self.raise_exception = e
                        continue

                for index in range(0, num_supported):
                    # Set clock frequency for each frequency supported
                    try:
                        freq_bitmask = frequencies[index]
                        msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                        ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, clk_cond):
                            self.raise_exception = e
                            continue

                # Set clock frequency back
                try:
                    freq_bitmask = frequencies[current]
                    msg = f"\t### amdsmi_set_clk_freq(gpu={i}, clk_type={clk_type_str}, freq_bitmask={freq_bitmask}):"
                    ret = amdsmi.amdsmi_set_clk_freq(gpu, clk_type_str, freq_bitmask)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, clk_cond):
                        self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_set_gpu_pci_bandwidth(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_pci_bandwidth as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Get current PCI bandwidth info
            msg = f"\t### amdsmi_get_gpu_pci_bandwidth(gpu={i}):"
            try:
                bandwidth_info = amdsmi.amdsmi_get_gpu_pci_bandwidth(gpu)
                self.common.print(msg, bandwidth_info)
                self.common.check_ret("", "", self.common.PASS)

                current_bandwidth_index = bandwidth_info["transfer_rate"]["current"]
                if current_bandwidth_index > 0:
                    bitmask = 1 << (current_bandwidth_index - 1)
                else:
                    bitmask = 1 << (current_bandwidth_index)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set PCI bandwidth
            msg = f"\t### amdsmi_set_gpu_pci_bandwidth(gpu={i}, bitmask={bitmask}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set back to original PCI bandwidth
            msg = f"\t### amdsmi_set_gpu_pci_bandwidth(gpu={i}, bitmask={bitmask}):"
            try:
                bitmask = 1 << (current_bandwidth_index)
                ret = amdsmi.amdsmi_set_gpu_pci_bandwidth(gpu, bitmask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_get_clk_freq(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = (
                "\tSkipping test_get_clk_freq as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            )
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clk_freq=amdsmi.amdsmi_get_clk_freq)
        return

    def test_get_clock_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_clock_info as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_get_clock_info=amdsmi.amdsmi_get_clock_info)
        return
