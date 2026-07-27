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

import unittest

import common.common as common
from common.common import amdsmi


class TestSystemTopology(unittest.TestCase):
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

        msg = "\t### amdsmi_get_socket_handles():"
        try:
            sockets = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in sockets])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                raise e

        self.assertGreaterEqual(len(sockets), 1)
        self.assertLessEqual(len(sockets), self.common.max_num_physical_devices)

        gpu_handles_by_type = []
        for i, socket in enumerate(sockets):
            for processor_name, processor_type, processor_cond in common.PROCESSOR_TYPES:
                msg = f"\t### amdsmi_get_processor_handles_by_type(socket={socket.value}, processor_type={processor_name}):"
                try:
                    ret = amdsmi.amdsmi_get_processor_handles_by_type(socket, processor_type)
                    handles = ret["processor_handles"]
                    count = ret["processor_count"]
                    # Handles are ctypes objects, so print a JSON-safe summary.
                    self.common.print(
                        msg,
                        {"processor_count": count, "processor_handles": [id(h) for h in handles]},
                    )
                    # processor_count must match the number of handles returned.
                    self.assertEqual(count, len(handles))
                    # Returned handles must be usable amdsmi_processor_handle objects,
                    # not raw integers.
                    for handle in handles:
                        self.assertIsInstance(handle, amdsmi.amdsmi_wrapper.amdsmi_processor_handle)
                    if processor_name == "AMD_GPU":
                        gpu_handles_by_type.extend(handles)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, processor_cond):
                        self.raise_exception = e

        # Regression guard: by-type lookup must enumerate the GPUs present on the
        # system. A clamped processor_count would silently return an empty list.
        self.assertEqual(len(gpu_handles_by_type), len(self.common.processors))

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

        msg = "\t### amdsmi_get_socket_handles():"
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
