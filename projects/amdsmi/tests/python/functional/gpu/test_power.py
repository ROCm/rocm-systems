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
"""GPU power: power cap set."""

import os
import sys
import unittest

# Allow direct execution: walk up to find the 'common/' package root
import os as _os

_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, "common")):
    _d = _os.path.dirname(_d)
sys.path.insert(0, _d)
del _d, _os

import common.helpers as common

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. '
        "Please set the correct path in your environment."
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as exc:
    raise ImportError(f"Could not import {amdsmi_path}") from exc

verbose = common.VERBOSITY_NORMAL
if "-q" in sys.argv or "--quiet" in sys.argv:
    verbose = common.VERBOSITY_QUIET
elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
    verbose = common.VERBOSITY_VERBOSE


class TestGpuPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)

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

    def test_get_gpu_power_profile_presets(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_power_profile_presets=amdsmi.amdsmi_get_gpu_power_profile_presets,
            index=0,
        )
        return

    def test_get_power_cap_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_cap_info=amdsmi.amdsmi_get_power_cap_info)
        return

    def test_get_power_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_info=amdsmi.amdsmi_get_power_info)
        return

    def test_is_gpu_power_management_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_is_gpu_power_management_enabled=amdsmi.amdsmi_is_gpu_power_management_enabled
        )
        return

    def test_set_gpu_power_profile(self):
        self.common.print_func_name("")
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for (
                power_profile_preset_mask_name,
                power_profile_preset_mask,
                power_profile_preset_masks_cond,
            ) in self.common.power_profile_preset_masks:
                msg = f"\t### amdsmi_set_gpu_power_profile(gpu={i}, power_profile_preset_mask={power_profile_preset_mask_name}):"
                try:
                    amdsmi.amdsmi_set_gpu_power_profile(gpu, 0, power_profile_preset_mask)
                    self.common.print(msg, "")
                    self.common.check_ret("", "", self.common.PASS)
                except amdsmi.AmdSmiLibraryException as e:
                    if self.common.check_ret(msg, e, power_profile_preset_masks_cond):
                        self.raise_exception = e
                self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    # pisolates


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Warning: Some tests may require elevated privileges (sudo/root).\n", file=sys.stderr)
    verbose = common.VERBOSITY_NORMAL
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
        verbose = common.VERBOSITY_VERBOSE
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)
    if "-h" in sys.argv or "--help" in sys.argv:
        unittest.main()
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()
    runner = unittest.TextTestRunner(
        stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose)
    )
    common.expand_glob_k_arg(globals())
    unittest.main(testRunner=runner)
