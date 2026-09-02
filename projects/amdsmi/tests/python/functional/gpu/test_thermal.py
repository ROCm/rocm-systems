#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU thermal: fan speed set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuThermal(unittest.TestCase):
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

    def test_gpu_fan_speed(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)

            found_error = False

            # Set invalid fan speed
            fan_speed = -1
            msg = f"\t### amdsmi_set_gpu_fan_speed(gpu={i}, index=0, fan_speed={fan_speed}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.ANY_FAIL):
                    self.raise_exception = e

            # Get current fan speed
            msg = f"\t### amdsmi_get_gpu_fan_speed(gpu={i}, index=0):"
            try:
                fan_speed_orig = amdsmi.amdsmi_get_gpu_fan_speed(gpu, 0)
                self.common.print(msg, fan_speed_orig)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                found_error = True

            if found_error:
                continue

            # Verify max fan speed returns a sensible value
            fan_speed_max = 0
            msg = f"\t### amdsmi_get_gpu_fan_speed_max(gpu={i}, index=0):"
            try:
                fan_speed_max = amdsmi.amdsmi_get_gpu_fan_speed_max(gpu, 0)
                self.common.print(msg, fan_speed_max)
                self.common.check_ret("", "", self.common.PASS)
                self.assertGreater(
                    fan_speed_max, 0, f"Max fan speed must be > 0, got {fan_speed_max}"
                )
                # max fan speed is the hwmon pwm1_max ceiling on every interface
                self.assertLessEqual(
                    fan_speed_max, 255, f"Max fan speed must be <= 255, got {fan_speed_max}"
                )
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                found_error = True

            if found_error:
                continue

            # amdsmi_set_gpu_fan_speed() validates against the gpu_od OD_RANGE where
            # present, which fan_speed_max does not describe
            gpu_bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
            min_value, max_value = common.get_fan_speed_set_range(gpu_bdf)
            fan_speed = min_value + ((max_value - min_value) // 2)

            # Set fan speed
            msg = f"\t### amdsmi_set_gpu_fan_speed(gpu={i}, index=0, fan_speed={fan_speed}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Reset fan to driver control (works for both legacy and gpu_od)
            msg = f"\t### amdsmi_reset_gpu_fan(gpu={i}, index=0):"
            try:
                ret = amdsmi.amdsmi_reset_gpu_fan(gpu, 0)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_get_gpu_fan_rpms(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_fan_rpms=amdsmi.amdsmi_get_gpu_fan_rpms, index=0
        )
        return

    def test_get_temp_metric(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_temp_metric as it fails (Invalid param)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_temp_metric=amdsmi.amdsmi_get_temp_metric,
            temperature_type=common.TEMPERATURE_TYPES,
            temperature_metric=common.TEMPERATURE_METRICS,
        )
        return
