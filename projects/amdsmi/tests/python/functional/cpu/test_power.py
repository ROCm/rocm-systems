#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CPU power: boostlimit set, socket power cap set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestCpuPower(unittest.TestCase):
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

    def test_cpu_core_boostlimit(self):
        self.common.print_func_name("")

        try:
            ret = amdsmi.amdsmi_get_cpu_handles()
            cpu_processors = ret["processor_handles"]
        except amdsmi.AmdSmiLibraryException:
            cpu_processors = []
        if not cpu_processors:
            msg = "\tNo CPU processors found; skipping CPU-specific test"
            self.common.print(msg)
            self.skipTest(msg)

        for i, cpu in enumerate(cpu_processors):
            found_error = False

            # Set invalid boostlimit
            boost_limit = 0
            msg = f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit={boost_limit}):"
            try:
                ret = amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, boost_limit)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.FAIL):
                    self.raise_exception = e

            # Get current boostlimit
            msg = f"\t### amdsmi_get_cpu_core_boostlimit(cpu={i}):"
            try:
                boost_limit_orig = amdsmi.amdsmi_get_cpu_core_boostlimit(cpu)
                self.common.print(msg, boost_limit_orig)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                found_error = True

            if found_error:
                continue

            # Set boostlimit back
            msg = f"\t### amdsmi_set_cpu_core_boostlimit(cpu={i}, boost_limit={boost_limit_orig}):"
            try:
                ret = amdsmi.amdsmi_set_cpu_core_boostlimit(cpu, boost_limit_orig)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_cpu_socket_power_cap(self):
        self.common.print_func_name("")

        try:
            ret = amdsmi.amdsmi_get_cpu_handles()
            cpu_processors = ret["processor_handles"]
        except amdsmi.AmdSmiLibraryException:
            cpu_processors = []
        if not cpu_processors:
            msg = "\tNo CPU processors found; skipping CPU-specific test"
            self.common.print(msg)
            self.skipTest(msg)

        for i, cpu in enumerate(cpu_processors):
            found_error = False

            # Set cpu socket power to invalid number
            power_cap = 0
            msg = f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={power_cap}):"
            try:
                ret = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, power_cap)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.FAIL):
                    self.raise_exception = e

            # Get cpu socket power
            msg = f"\t### amdsmi_get_cpu_socket_power_cap(cpu={i}):"
            try:
                power_cap_orig = amdsmi.amdsmi_get_cpu_socket_power_cap(cpu)
                self.common.print(msg, power_cap_orig)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                    found_error = True

            # Get cpu socket power to max
            msg = f"\t### amdsmi_get_cpu_socket_power_cap_max(cpu={i}):"
            try:
                power_cap_max = amdsmi.amdsmi_get_cpu_socket_power_cap_max(cpu)
                self.common.print(msg, power_cap_max)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                found_error = True

            if found_error:
                continue

            # Set cpu socket power to max
            msg = f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={power_cap_max}):"
            try:
                ret = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, power_cap_max)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set cpu socket power back
            msg = f"\t### amdsmi_set_cpu_socket_power_cap(cpu={i}, power_cap={power_cap_orig}):"
            try:
                ret = amdsmi.amdsmi_set_cpu_socket_power_cap(cpu, power_cap_orig)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_cpu_socket_boostlimit(self):
        self.common.print_func_name("")

        try:
            ret = amdsmi.amdsmi_get_cpu_handles()
            cpu_processors = ret["processor_handles"]
        except amdsmi.AmdSmiLibraryException:
            cpu_processors = []
        if not cpu_processors:
            msg = "\tNo CPU processors found; skipping CPU-specific test"
            self.common.print(msg)
            self.skipTest(msg)

        # TODO boost_limit = 0
        boost_limit = 0
        for i, cpu in enumerate(cpu_processors):
            msg = f"cpu({i}):"
            msg1 = f"{msg} boost_limit({boost_limit}):"
            try:
                amdsmi.amdsmi_set_cpu_socket_boostlimit(cpu, boost_limit)
                self.common.print(msg1, "")
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg1, e, self.common.PASS):
                    self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_cpu_pwr_svi_telemetry_all_rails(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_pwr_svi_telemetry_all_rails=amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails
        )
        return

    def test_get_cpu_socket_power(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_cpu_socket_power=amdsmi.amdsmi_get_cpu_socket_power)
        return

    def test_get_cpu_socket_power_cap(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_power_cap=amdsmi.amdsmi_get_cpu_socket_power_cap
        )
        return

    def test_get_cpu_socket_power_cap_max(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_cpu_socket_power_cap_max=amdsmi.amdsmi_get_cpu_socket_power_cap_max
        )
        return

    def test_set_cpu_pwr_efficiency_mode(self):
        self.common.print_func_name("")
        modes = [0, 1, 2]
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for mode in modes:
                msg = f"\t### amdsmi_set_cpu_pwr_efficiency_mode(gpu={i}, mode={mode}):"
                try:
                    amdsmi.amdsmi_set_cpu_pwr_efficiency_mode(gpu, mode)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return
