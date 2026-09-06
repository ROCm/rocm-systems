#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU power: power cap set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuPower(unittest.TestCase):
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

    def test_gpu_power_cap(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)

            # Get Power Cap Info
            try:
                msg = f"\t### amdsmi_get_supported_power_cap(gpu={i}):"
                power_cap_types = amdsmi.amdsmi_get_supported_power_cap(gpu)
                # TODO(amdsmi_team): we should be iterating through all supported power cap sensors,
                #                    but for now we will just test the first one.
                #                    See amdsmi_get_supported_power_cap for more details
                #                    on the structure of power_cap_types
                sensor_type = power_cap_types["sensor_types"][0]
                self.common.print(msg, power_cap_types)
                self.common.check_ret("", "", self.common.PASS)

                msg = f"\t### amdsmi_get_power_cap_info(gpu={i}):"
                power_cap_info = amdsmi.amdsmi_get_power_cap_info(gpu, sensor_type)
                self.common.print(msg, power_cap_info)
                self.common.check_ret("", "", self.common.PASS)
                cap = int((power_cap_info["max_power_cap"] + power_cap_info["min_power_cap"]) / 2)
                current_cap = power_cap_info["power_cap"]
                # A power_cap of 0 means no user cap is set (driver uses default).
                # Restoring 0 would be rejected as out-of-range, so use default_power_cap instead.
                if current_cap == 0:
                    current_cap = power_cap_info["default_power_cap"]
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                # Have to be able to get info before setting
                continue

            # Set to Average Power Cap
            msg = f"\t### amdsmi_set_power_cap(gpu={i}, sensor={sensor_type}, power_cap={cap}):"
            try:
                ret = amdsmi.amdsmi_set_power_cap(gpu, sensor_type, cap)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Restore Power Cap
            msg = f"\t### amdsmi_set_power_cap(gpu={i}, sensor={sensor_type}, power_cap={current_cap}):"
            try:
                ret = amdsmi.amdsmi_set_power_cap(gpu, sensor_type, current_cap)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_set_gpu_power_profile(self):
        self.common.print_func_name("")
        # Valid rejections (although driver may include others)
        refused = [
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,  # eg. guest devices, or sysfs node absent
            amdsmi.AmdSmiStatus.INPUT_OUT_OF_BOUNDS,  # eg. invalid input
            amdsmi.AmdSmiStatus.NO_PERM,  # eg. R/O sysfs in a container (EROFS)
        ]
        # All valid responses
        applied_or_refused = [amdsmi.AmdSmiStatus.SUCCESS, *refused]
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)
                for mask_name, mask, _ in common.POWER_PROFILE_PRESET_MASKS:
                    accept = refused if mask_name == "INVALID" else applied_or_refused
                    msg = (
                        f"\t### amdsmi_set_gpu_power_profile(gpu={i}, "
                        f"power_profile_preset_mask={mask_name}):"
                    )
                    with self.common.expect_status(msg, accept):
                        amdsmi.amdsmi_set_gpu_power_profile(gpu, 0, mask)
                    self.common.print("")
        return
