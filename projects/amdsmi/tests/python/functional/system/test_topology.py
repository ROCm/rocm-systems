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
"""System topology: processor handles, socket info, processor type discovery, utilization count."""

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


class TestSystemTopology(unittest.TestCase):
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

    def test_get_processor_handle_from_bdf(self):
        self.common.print_func_name("")

        # With invalid gpu
        gpu = -1
        msg = f"\t### amdsmi_get_gpu_device_bdf(gpu={gpu}):"
        try:
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
            self.common.print(msg, bdf)
            self.fail(
                f"{msg} Expected an exception for invalid gpu index {gpu}, but call succeeded with bdf {bdf}"
            )
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.FAIL):
                self.raise_exception = e

        # With invalid bdf
        bdf = "0"
        msg = f"\t### amdsmi_get_processor_handle_from_bdf(bdf={bdf}):"
        try:
            ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            self.common.print(msg, ret.value)
            self.fail(
                f'{msg} Expected an exception for invalid BDF "{bdf}", but call succeeded with handle {ret.value}'
            )
        except (
            amdsmi.AmdSmiLibraryException,
            amdsmi.AmdSmiParameterException,
            amdsmi.amdsmi_exception.AmdSmiBdfFormatException,
        ) as e:
            if self.common.check_ret(msg, e, self.common.FAIL):
                self.raise_exception = e

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"\t### amdsmi_get_gpu_device_bdf(gpu={i}):"
            try:
                bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu)
                self.common.print(msg, bdf)
                self.common.print(f"gpu.value={gpu.value}")
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            msg = f"\t### amdsmi_get_processor_handle_from_bdf(bdf={bdf}):"
            try:
                ret = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                self.common.print(msg, ret.value)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            if gpu.value != ret.value:
                msg += f"gpu={i}: Expected: {gpu.value}, Received: {ret.value}"
                self.raise_exception = amdsmi.AmdSmiParameterException(ret.value, gpu.value, msg)

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_processor_handles_by_type(self):
        self.common.print_func_name("")

        # With bad input
        socket = -1
        processor_type = amdsmi.AmdSmiProcessorType.UNKNOWN
        msg = f"\t### amdsmi_get_processor_handles_by_type(socket={socket}, processor_type={'UNKNOWN'}):"
        try:
            ret = amdsmi.amdsmi_get_processor_handles_by_type(socket, processor_type)
            self.common.print(msg, ret)
            self.fail(
                f"{msg} Expected an exception for invalid inputs (socket={socket}, "
                f"processor_type=UNKNOWN), but call succeeded with ret {ret}"
            )
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.FAIL):
                self.raise_exception = e

        msg = f"\t### amdsmi_get_socket_handles():"
        try:
            sockets = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in sockets])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                raise e

        self.assertGreaterEqual(len(sockets), 1)
        self.assertLessEqual(len(sockets), self.common.max_num_physical_devices)

        for i, socket in enumerate(sockets):
            for processor_name, processor_type, processor_cond in self.common.processor_types:
                msg = f"\t### amdsmi_get_processor_handles_by_type(socket={socket.value}, processor_type={processor_name}):"
                try:
                    ret = amdsmi.amdsmi_get_processor_handles_by_type(socket, processor_type)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, processor_cond):
                        self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_get_socket_info(self):
        self.common.print_func_name("")
        # With invalid socket
        socket = -1
        msg = f"\t### amdsmi_get_socket_info(socket={socket}):"
        try:
            ret = amdsmi.amdsmi_get_socket_info(socket)
            self.common.print(msg, ret)
            self.fail(
                f"{msg} Expected an exception for invalid socket index {socket}, "
                f"but call succeeded with ret {ret}"
            )
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.FAIL):
                self.raise_exception = e

        msg = f"\t### amdsmi_get_socket_handles():"
        try:
            sockets = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in sockets])
            self.common.check_ret("", "", self.common.PASS)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                raise e

        self.assertGreaterEqual(len(sockets), 1)
        self.assertLessEqual(len(sockets), self.common.max_num_physical_devices)

        for i, socket in enumerate(sockets):
            msg = f"\t### amdsmi_get_socket_info(socket={i}):"
            try:
                ret = amdsmi.amdsmi_get_socket_info(socket)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                    continue

        if self.raise_exception:
            raise self.raise_exception
        return

    def test_utilization_count(self):
        self.common.print_func_name("")

        util_good_counter_types = [
            amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_GFX_ACTIVITY,
            amdsmi.AmdSmiUtilizationCounterType.COARSE_GRAIN_MEM_ACTIVITY,
            amdsmi.AmdSmiUtilizationCounterType.COARSE_DECODER_ACTIVITY,
        ]
        util_bad_counter_types = [amdsmi.AmdSmiTemperatureMetric.CURRENT]

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"\t### amdsmi.amdsmi_get_utilization_count(gpu={i}, utilization_counter_types={util_good_counter_types}):"
            try:
                util_count = amdsmi.amdsmi_get_utilization_count(gpu, util_good_counter_types)
                self.common.print(msg, util_count)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # With invalid entry
            msg = f"\t### amdsmi.amdsmi_get_utilization_count(gpu={i}, utilization_counter_types={util_bad_counter_types}):"
            try:
                util_count = amdsmi.amdsmi_get_utilization_count(gpu, util_bad_counter_types)
                self.common.print(msg, util_count)
                self.fail(
                    f"{msg} Expected an exception for invalid counter type list "
                    f"(mixed AmdSmiTemperatureMetric in util_bad_counter_types), "
                    f"but call succeeded with util_count {util_count}"
                )
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.ANY_FAIL):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration


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
