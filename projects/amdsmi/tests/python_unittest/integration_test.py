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

"""
In integration testing, what is specifically tested:
1. Module Interfaces: The primary focus is on the connections and data exchange points
   (interfaces) between individual software modules or components.
2. Data Flow: How data is passed between modules, ensuring it is formatted correctly and
   transferred without loss or corruption.
3. System Logic: The integrated logic across multiple modules is checked to confirm that
   the combined functionality aligns with requirements and produces the expected outcomes.
4. External Dependencies: Interactions with external systems like databases, file servers,
   or other applications (via APIs) are tested to ensure seamless operation.
5. Cohesion: Whether the various integrated units function as a single, cohesive unit to
   achieve a broader system goal
"""

import os
import sys
import unittest
import common

# Module-level default: match unittest's default verbosity (dots)
verbose = common.VERBOSITY_NORMAL


amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.'
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as exc:
    raise ImportError(f"Could not import {amdsmi_path}") from exc


class TestAmdSmiInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)
        return

    def test_init_shutdown(self):
        self.common.print(f"## test_init_shutdown()")

        msg = f"\t### amdsmi_init():"
        try:
            ret = self.common.amdsmi_smart_init()[0]
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e

        msg = f"\t### amdsmi_shut_down():"
        try:
            ret = amdsmi.amdsmi_shut_down()
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e
        return


class TestAmdSmiPython(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)

        if cls.common.verbose > common.VERBOSITY_QUIET:
            # Execute the following to print the asic and board info once per test run
            for i, _ in enumerate(cls.common.processors):
                msg = f"gpu={i}"
                cls.common.print(msg)
                if i < len(cls.common.virt_mode):
                    msg = f"virtualization mode(gpu={i})"
                    cls.common.print(msg, cls.common.virt_mode[i])
                if i < len(cls.common.asic_info):
                    msg = f"asic info(gpu={i})"
                    cls.common.print(msg, cls.common.asic_info[i])
                if i < len(cls.common.board_info):
                    msg = f"board info(gpu={i})"
                    cls.common.print(msg, cls.common.board_info[i])
                cls.common.print("")
        return

    @classmethod
    def tearDownClass(cls):
        return

    def setUp(self):
        # Called before each test by unittest framework
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        # Refresh processor handles after re-init: handles obtained from a previous
        # init/shutdown cycle are invalid and return AMDSMI_STATUS_NOT_FOUND.
        self.common.processors = amdsmi.amdsmi_get_processor_handles()
        return

    def tearDown(self):
        # Called after each test by unittest framework
        amdsmi.amdsmi_shut_down()
        return

    # integration
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

    def test_nic_bdf_device_id(self):
        self.common.print_func_name("")
        common.Common._skip_if_missing(
            self,
            [
                "amdsmi_get_nic_processor_handles",
                "amdsmi_get_nic_info",
                "amdsmi_get_nic_device_uuid",
            ],
        )
        processors = amdsmi.amdsmi_get_nic_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
        for i in range(0, len(processors)):
            bdf = ""
            nic_info = amdsmi.amdsmi_get_nic_info(processors[i])
            if nic_info:
                bdf = nic_info["bdf"]
            print(f"\n\n###Test nic Processor {i}, bdf: {bdf}")
            print("\n###Test amdsmi_get_processor_handle_from_bdf\n")
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_nic_device_uuid\n")
            uuid = amdsmi.amdsmi_get_nic_device_uuid(processor)
            print(f"  uuid is: {uuid}")
        print()
        return

    def test_switch_bdf_device_id(self):
        self.common.print_func_name("")
        common.Common._skip_if_missing(
            self,
            [
                "amdsmi_get_switch_processor_handles",
                "amdsmi_get_switch_device_bdf",
                "amdsmi_get_device_id",
            ],
        )
        processors = amdsmi.amdsmi_get_switch_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_switch_device_bdf(processors[i])
            print(f"\n\n###Test switch Processor {i}, bdf: {bdf}")
            print("\n###Test amdsmi_get_processor_handle_from_bdf\n")
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_device_id\n")
            device_id = amdsmi.amdsmi_get_device_id(processor)
            print(f"  device_id is: {device_id}")
        print()

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
    def test_gpu_counter(self):
        self.common.print_func_name("")

        results = {}
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            results[i] = {}
            for event_group_name, event_group, event_group_cond in self.common.event_groups:
                results[i][event_group_name] = {}
                results[i][event_group_name]["supported"] = False
                results[i][event_group_name]["counters"] = 0

                # Is supported
                msg = f"\t### amdsmi_gpu_counter_group_supported(gpu={i}, event_group={event_group_name}):"
                try:
                    event_handle = amdsmi.amdsmi_gpu_counter_group_supported(gpu, event_group)
                    self.common.print(msg, event_handle)
                    results[i][event_group_name]["supported"] = True
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, event_group_cond):
                        self.raise_exception = e

                # Are counters available
                msg = f"\t### amdsmi_get_gpu_available_counters(gpu={i}, event_group={event_group_name}):"
                try:
                    counters = amdsmi.amdsmi_get_gpu_available_counters(gpu, event_group)
                    self.common.print(msg, counters)
                    results[i][event_group_name]["counters"] = counters
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, event_group_cond):
                        self.raise_exception = e

                # To continue, both have to pass
                if (
                    not results[i][event_group_name]["supported"]
                    or not results[i][event_group_name]["counters"]
                ):
                    msg_add = ""
                    if not results[i][event_group_name]["supported"]:
                        msg_add = "\n\tAMDSMI API Returned AMDSMI_STATUS_NOT_SUPPORTED"
                    # Record that these would have been tested if supported
                    self.common.print(f"\t### amdsmi_gpu_create_counter(){msg_add}")
                    self.common.print(f"\t### amdsmi_gpu_control_counter(){msg_add}")
                    self.common.print(f"\t### amdsmi_gpu_read_counter(){msg_add}")
                    self.common.print(f"\t### amdsmi_gpu_control_counter(){msg_add}")
                    self.common.print(f"\t### amdsmi_gpu_destroy_counter(){msg_add}")
                    continue

                for event_type_name, event_type, event_type_cond in self.common.event_types:
                    results[i][event_group_name][event_type_name] = {}
                    results[i][event_group_name][event_type_name]["handle"] = 0
                    results[i][event_group_name][event_type_name]["num_counts"] = 0

                    # Create
                    msg = f"\t### amdsmi_gpu_create_counter(gpu={i}, event_type={event_type_name}):"
                    try:
                        event_handle = amdsmi.amdsmi_gpu_create_counter(gpu, event_type)
                        self.common.print(msg, id(event_handle))
                        self.common.check_ret("", "", self.common.PASS)
                        results[i][event_group_name][event_type_name]["handle"] = id(event_handle)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, event_type_cond):
                            self.raise_exception = e
                        # Record that these would have been tested if supported
                        msg_add = "\n\tAMDSMI API Returned AMDSMI_STATUS_NOT_SUPPORTED"
                        self.common.print(f"\t### amdsmi_gpu_control_counter(){msg_add}")
                        self.common.print(f"\t### amdsmi_gpu_read_counter(){msg_add}")
                        self.common.print(f"\t### amdsmi_gpu_control_counter(){msg_add}")
                        self.common.print(f"\t### amdsmi_gpu_destroy_counter(){msg_add}")
                        continue

                    # Start a program that generates the events of interest

                    # Start control counter
                    counter_command_name, counter_command, counter_commands_cond = (
                        self.common.counter_commands[0]
                    )
                    msg = f"\t### amdsmi_gpu_control_counter(event_handle={id(event_handle)}, counter_command={counter_command_name}):"
                    try:
                        ret = amdsmi.amdsmi_gpu_control_counter(event_handle, counter_command)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, event_type_cond):
                            self.raise_exception = e

                    # Wait...

                    # Read control counter
                    msg = f"\t### amdsmi_gpu_read_counter(event_handle={id(event_handle)}):"
                    try:
                        ret = amdsmi.amdsmi_gpu_read_counter(event_handle)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                        results[i][event_group_name][event_type_name]["num_counts"] = ret
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, event_type_cond):
                            self.raise_exception = e

                    # Stop control counter
                    counter_command_name, counter_command, counter_commands_cond = (
                        self.common.counter_commands[1]
                    )
                    msg = f"\t### amdsmi_gpu_control_counter(event_handle={id(event_handle)}, counter_command={counter_command_name}):"
                    try:
                        ret = amdsmi.amdsmi_gpu_control_counter(event_handle, counter_command)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, event_type_cond):
                            self.raise_exception = e

                    # Destroy control counter
                    msg = f"\t### amdsmi_gpu_destroy_counter(event_handle={id(event_handle)}):"
                    try:
                        ret = amdsmi.amdsmi_gpu_destroy_counter(event_handle)
                        self.common.print(msg, ret)
                        self.common.check_ret("", "", self.common.PASS)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if self.common.check_ret(msg, e, event_type_cond):
                            self.raise_exception = e

        msg = "gpu counter results"
        self.common.print(msg, results)

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
    def test_gpu_event(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_gpu_event as it fails (File Error)."
            self.common.print(msg)
            self.skipTest(msg)

        mask = 1 << (amdsmi.AmdSmiEvtNotificationType.GPU_PRE_RESET - 1) | 1 << (
            amdsmi.AmdSmiEvtNotificationType.GPU_POST_RESET - 1
        )
        timeout_ms = 1000

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"\t### amdsmi_init_gpu_event_notification(gpu={i}):"

            # Init
            try:
                ret = amdsmi.amdsmi_init_gpu_event_notification(gpu)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                # Skip remaining tests on any exception when initializing
                continue

            # Set Mask
            msg = f"\t### amdsmi_set_gpu_event_notification_mask(gpu={i}, mask={mask}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_event_notification_mask(gpu, mask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Get
            msg = f"\t### amdsmi_get_gpu_event_notification(timeout_ms={timeout_ms}):"
            try:
                ret = amdsmi.amdsmi_get_gpu_event_notification(timeout_ms)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Stop
            msg = f"\t### amdsmi_stop_gpu_event_notification(gpu={i}):"
            try:
                ret = amdsmi.amdsmi_stop_gpu_event_notification(gpu)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
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
            for clk_type_name, clk_type, clk_cond in self.common.clk_types:
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
    def test_cpu_core_boostlimit(self):
        self.common.print_func_name("")

        try:
            cpu_processors = amdsmi.amdsmi_get_cpusocket_handles()
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
            cpu_processors = amdsmi.amdsmi_get_cpusocket_handles()
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

                # Convert power_cap_max from string that has units to an integer
                # Ex.  power_cap_max = "5000 mW"  to   power_cap_max = 5000
                power_cap_max = int(power_cap_max.split()[0])
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
    def test_set_gpu_compute_partition(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_compute_partition as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            default_compute_partition_type = self.common.compute_partition_types[0][1]
            msg = f"\t### amdsmi_get_gpu_compute_partition(gpu={i}):"
            try:
                default_compute_partition_name = amdsmi.amdsmi_get_gpu_compute_partition(gpu)
                self.common.print(msg, default_compute_partition_name)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            for (
                compute_partition_type_name,
                compute_partition_type,
                compute_partition_type_cond,
            ) in self.common.compute_partition_types:
                if default_compute_partition_name == compute_partition_type_name:
                    default_compute_partition_type = compute_partition_type
                msg = f"\t### amdsmi_set_gpu_compute_partition(gpu={i}, compute_partition_type={compute_partition_type_name}):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_compute_partition(gpu, compute_partition_type)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, compute_partition_type_cond):
                        self.raise_exception = e

            msg = f"\t### amdsmi_set_gpu_compute_partition(gpu={i}, default_compute_partition={default_compute_partition_name}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_compute_partition(gpu, default_compute_partition_type)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
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

            # Determine max fan speed
            msg = f"\t### amdsmi_get_gpu_fan_speed_max(gpu={i}, index=0):"
            try:
                fan_speed_max = amdsmi.amdsmi_get_gpu_fan_speed_max(gpu, 0)
                self.common.print(msg, fan_speed_max)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                found_error = True

            if found_error:
                continue

            if fan_speed_orig == fan_speed_max:
                fan_speed = int(fan_speed_max / 2)
            else:
                fan_speed = fan_speed_max

            # Set fan speed
            msg = f"\t### amdsmi_set_gpu_fan_speed(gpu={i}, index=0, fan_speed={fan_speed}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set to original fan speed
            msg = f"\t### amdsmi_set_gpu_fan_speed(gpu={i}, index=0, fan_speed_current={fan_speed_orig}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_fan_speed(gpu, 0, fan_speed_orig)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
    def test_set_gpu_overdrive_level(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Find current overdrive value
            msg = f"\t### amdsmi_get_gpu_overdrive_level(gpu={i}):"
            try:
                overdrive_value_current = amdsmi.amdsmi_get_gpu_overdrive_level(gpu)
                self.common.print(msg, overdrive_value_current)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            if overdrive_value_current != 1:
                overdrive_value = 1
            else:
                overdrive_value = 2

            # Set overdrive value
            msg = (
                f"\t### amdsmi_set_gpu_overdrive_level(gpu={i}, overdrive_value={overdrive_value}):"
            )
            try:
                ret = amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set back to original overdrive value
            msg = f"\t### amdsmi_set_gpu_overdrive_level(gpu={i}, overdrive_value={overdrive_value_current}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value_current)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
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
    def test_set_gpu_perf_level(self):
        self.common.print_func_name("")

        dev_perf_level_current = self.common.dev_perf_levels[0][1]
        dev_perf_level_current_cond = self.common.dev_perf_levels[0][2]
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"\t### amdsmi_get_gpu_perf_level(gpu={i}):"
            try:
                dev_perf_level_name_current = amdsmi.amdsmi_get_gpu_perf_level(gpu)
                items = dev_perf_level_name_current.split("_")
                level_name_current = items[-1]
                for name, level, cond in self.common.dev_perf_levels:
                    if name == level_name_current:
                        dev_perf_level_current = level
                        dev_perf_level_current_cond = cond
                        break
                self.common.print(msg, level_name_current)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                continue

            for (
                dev_perf_level_name,
                dev_perf_level,
                dev_perf_level_cond,
            ) in self.common.dev_perf_levels:
                msg = f"\t### amdsmi_set_gpu_perf_level(gpu={i}, dev_perf_level={dev_perf_level_name}):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, dev_perf_level_cond):
                        self.raise_exception = e

            msg = f"\t### amdsmi_set_gpu_perf_level(gpu={i}, dev_perf_level={dev_perf_level_name_current}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level_current)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, dev_perf_level_current_cond):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
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
    def test_soc_pstate(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Get current policy info
            msg = f"\t### amdsmi_get_soc_pstate(gpu={i}):"
            try:
                policy_info = amdsmi.amdsmi_get_soc_pstate(gpu)
                self.common.print(msg, "")
                self.common.check_ret("", "", self.common.PASS)

                num_supported = policy_info["num_supported"]
                if not isinstance(num_supported, int):
                    self.common.print(f"Cannot determine num_supported={num_supported}", "")
                    continue
                policy_id_current = policy_info["current_id"]
                if not isinstance(policy_id_current, int):
                    self.common.print(f"Cannot determine policy_id_current={policy_id_current}", "")
                    continue
                policy_id_orig = policy_info["policies"][policy_id_current]["policy_id"]
                if not isinstance(policy_id_orig, int):
                    self.common.print(f"Cannot determine orig policy_id={policy_id_orig}", "")
                    continue

                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info["policies"][index]["policy_id"]
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set SOC Pstate policy
            msg = f"\t### amdsmi_set_soc_pstate(gpu={i}):"
            try:
                ret = amdsmi.amdsmi_set_soc_pstate(gpu, policy_id)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set back to original policy
            msg = f"\t### amdsmi_set_soc_pstate(gpu={i}, policy_id={policy_id_orig}):"
            try:
                ret = amdsmi.amdsmi_set_soc_pstate(gpu, policy_id_orig)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
    def test_xgmi_plpd(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_xgmi_plpd as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"gpu({i}):"

            # Get current policy info
            msg = f"\t### amdsmi_get_xgmi_plpd(gpu={i}):"
            try:
                policy_info = amdsmi.amdsmi_get_xgmi_plpd(gpu)
                self.common.print(msg, "")
                self.common.check_ret("", "", self.common.PASS)

                num_supported = policy_info["num_supported"]
                if not isinstance(num_supported, int):
                    self.common.print(f"Cannot determine num_supported={num_supported}", "")
                    continue
                policy_id_current = policy_info["current_id"]
                if not isinstance(policy_id_current, int):
                    self.common.print(f"Cannot determine policy_id_current={policy_id_current}", "")
                    continue
                policy_id_orig = policy_info["policies"][policy_id_current]["policy_id"]
                if not isinstance(policy_id_orig, int):
                    self.common.print(f"Cannot determine orig policy_id={policy_id_orig}", "")
                    continue
                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info["policies"][index]["policy_id"]
                if not isinstance(policy_id, int):
                    self.common.print(f"Cannot determine policy_id={policy_id}", "")
                    continue
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set policy
            msg = f"\t### amdsmi_set_xgmi_plpd(gpu={i}, policy_id={policy_id}):"
            try:
                ret = amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set back to original policy
            msg = f"\t### amdsmi_set_xgmi_plpd(gpu={i}, policy_id={policy_id_orig}):"
            try:
                ret = amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id_orig)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # import multiprocessing
    # Unstable on workstation cards
    # def test_walkthrough_multiprocess(self):
    #     print("\n\n========> test_walkthrough_multiprocess start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     p0 = multiprocessing.Process(target=walk_through, args=[self])
    #     p1 = multiprocessing.Process(target=walk_through, args=[self])
    #     p2 = multiprocessing.Process(target=walk_through, args=[self])
    #     p3 = multiprocessing.Process(target=walk_through, args=[self])
    #     p0.start()
    #     p1.start()
    #     p2.start()
    #     p3.start()
    #     p0.join()
    #     p1.join()
    #     p2.join()
    #     p3.join()
    #     print("\n========> test_walkthrough_multiprocess end <========\n")

    # import threading
    # Unstable on workstation cards
    # def test_walkthrough_multithread(self):
    #     print("\n\n========> test_walkthrough_multithread start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     t0 = threading.Thread(target=walk_through, args=[self])
    #     t1 = threading.Thread(target=walk_through, args=[self])
    #     t2 = threading.Thread(target=walk_through, args=[self])
    #     t3 = threading.Thread(target=walk_through, args=[self])
    #     t0.start()
    #     t1.start()
    #     t2.start()
    #     t3.start()
    #     t0.join()
    #     t1.join()
    #     t2.join()
    #     t3.join()
    #     print("\n========> test_walkthrough_multithread end <========\n")

    # # Unstable - do not run
    # def test_z_gpureset_asicinfo_multithread(self):
    #     def get_asic_info(processor):
    #         try:
    #             print("\n###Test amdsmi_get_gpu_asic_info \n")
    #             asic_info = amdsmi.amdsmi_get_gpu_asic_info(processor)
    #         except amdsmi.AmdSmiLibraryException as e:
    #             self.common.check_exception(e)
    #             continue
    #         print("  asic_info['market_name'] is: {}".format(
    #             asic_info['market_name']))
    #         print("  asic_info['vendor_id'] is: {}".format(
    #             asic_info['vendor_id']))
    #         print("  asic_info['vendor_name'] is: {}".format(
    #             asic_info['vendor_name']))
    #         print("  asic_info['device_id'] is: {}".format(
    #             asic_info['device_id']))
    #         print("  asic_info['rev_id'] is: {}".format(
    #             asic_info['rev_id']))
    #         print("  asic_info['asic_serial'] is: {}".format(
    #             asic_info['asic_serial']))
    #         print("  asic_info['oam_id'] is: {}\n".format(
    #             asic_info['oam_id']))
    #     def gpu_reset(processor):
    #         print("\n###Test amdsmi_reset_gpu \n")
    #         amdsmi.amdsmi_reset_gpu(processor)
    #         print("  GPU reset completed.\n")
    #     print("\n\n========> test_z_gpureset_asicinfo_multithread start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     for i in range(0, len(processors)):
    #         bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
    #         print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
    #         t0 = threading.Thread(target=get_asic_info, args=[processors[i]])
    #         t1 = threading.Thread(target=gpu_reset, args=[processors[i]])
    #         # t2 = threading.Thread(target=walk_through, args=[self])
    #         # t3 = threading.Thread(target=walk_through, args=[self])
    #         t0.start()
    #         t1.start()
    #         # t2.start()
    #         # t3.start()
    #         t0.join()
    #         t1.join()
    #         # t2.join()
    #         # t3.join()
    #     print("\n========> test_z_gpureset_asicinfo_multithread end <========\n")

    def test_uma_carveout_info(self):
        """Test UMA carveout (VRAM) information retrieval"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            self.common.print(f"\n\n###Test Processor {i}, bdf: {bdf}")

            msg = f"\t### amdsmi_get_gpu_uma_carveout_info(gpu={i}):"
            try:
                uma_info = amdsmi.amdsmi_get_gpu_uma_carveout_info(processors[i])
                self.common.print(msg, uma_info)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
                )
                continue

            # Validate returned data structure
            self.assertIn("current_index", uma_info)
            self.assertIn("num_options", uma_info)
            self.assertIn("options", uma_info)

            # Validate that current_index is within valid range
            self.assertGreaterEqual(uma_info["current_index"], 0)
            self.assertLess(uma_info["current_index"], uma_info["num_options"])

            # Validate that we have at least one option
            self.assertGreater(uma_info["num_options"], 0)
            self.assertLessEqual(uma_info["num_options"], 16)

            # Validate options list
            self.assertEqual(len(uma_info["options"]), uma_info["num_options"])

            for j, opt in enumerate(uma_info["options"]):
                self.assertIn("index", opt)
                self.assertIn("description", opt)
                self.assertEqual(opt["index"], j)
                self.assertGreater(len(opt["description"]), 0)
                marker = "*" if opt["index"] == uma_info["current_index"] else " "
                self.common.print(f"  {marker} Option {opt['index']}: {opt['description']}")
        return

    def test_uma_carveout_set_dry_run(self):
        """Test UMA carveout write operations in DRY_RUN mode"""
        self.common.print_func_name("")
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), self.common.max_num_physical_devices)

        # Enable DRY_RUN mode; ensure cleanup even if test fails
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)

        for i in range(0, len(processors)):
            self.common.print_device_header(i)
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            self.common.print(f"\n\n###Test Processor {i}, bdf: {bdf}")

            msg = f"\t### amdsmi_get_gpu_uma_carveout_info(gpu={i}):"
            try:
                uma_info = amdsmi.amdsmi_get_gpu_uma_carveout_info(processors[i])
                self.common.print(msg, uma_info)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(
                    e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
                )
                continue

            # Test setting to current value
            msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={uma_info['current_index']}) (DRY_RUN):"
            try:
                ret = amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], uma_info["current_index"])
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.fail(f"Failed to set UMA carveout to current value in DRY_RUN mode: {e}")

            # Test setting to different valid index if available
            if uma_info["num_options"] > 1:
                test_index = (uma_info["current_index"] + 1) % uma_info["num_options"]
                msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={test_index}) (DRY_RUN):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], test_index)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    self.fail(f"Failed to set UMA carveout to valid index in DRY_RUN mode: {e}")

            # Test setting to invalid index (should fail with AMDSMI_STATUS_INVAL)
            invalid_index = uma_info["num_options"] + 10
            msg = f"\t### amdsmi_set_gpu_uma_carveout(gpu={i}, index={invalid_index}) (DRY_RUN):"
            try:
                amdsmi.amdsmi_set_gpu_uma_carveout(processors[i], invalid_index)
                self.fail(f"Should have raised exception for invalid index {invalid_index}")
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                self.assertEqual(e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg)
        return

    def test_ttm_info(self):
        """Test TTM (GTT/shared memory) information retrieval"""
        self.common.print_func_name("")

        msg = "\t### amdsmi_get_ttm_info():"
        try:
            ttm_info = amdsmi.amdsmi_get_ttm_info()
            self.common.print(msg, ttm_info)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            self.assertEqual(
                e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
            )
            return

        # Validate returned data structure
        self.assertIn("current_pages", ttm_info)

        # Validate that pages value is reasonable (> 0)
        self.assertGreater(ttm_info["current_pages"], 0)

        page_size = os.sysconf("SC_PAGESIZE")
        gb = (ttm_info["current_pages"] * page_size) / (1024**3)
        self.common.print(f"  TTM size: {gb:.2f} GB")
        return

    def test_ttm_set_dry_run(self):
        """Test TTM write operations in DRY_RUN mode"""
        self.common.print_func_name("")

        # Get current TTM info first
        msg = "\t### amdsmi_get_ttm_info():"
        try:
            ttm_info = amdsmi.amdsmi_get_ttm_info()
            self.common.print(msg, ttm_info)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            self.assertEqual(
                e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED, msg
            )
            return

        # Enable DRY_RUN mode; ensure cleanup even if test fails
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)

        # Test setting TTM pages limit to current value
        msg = f"\t### amdsmi_set_ttm_pages_limit(pages={ttm_info['current_pages']}) (DRY_RUN):"
        try:
            ret = amdsmi.amdsmi_set_ttm_pages_limit(ttm_info["current_pages"])
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            self.fail(f"Failed to set TTM to current value in DRY_RUN mode: {e}")

        # Test setting TTM to a different value
        test_pages = ttm_info["current_pages"] // 2
        if test_pages > 0:
            msg = f"\t### amdsmi_set_ttm_pages_limit(pages={test_pages}) (DRY_RUN):"
            try:
                ret = amdsmi.amdsmi_set_ttm_pages_limit(test_pages)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.fail(f"Failed to set TTM to different value in DRY_RUN mode: {e}")

        # Test setting TTM to 0 (should fail with AMDSMI_STATUS_INVAL)
        msg = "\t### amdsmi_set_ttm_pages_limit(pages=0) (DRY_RUN):"
        try:
            amdsmi.amdsmi_set_ttm_pages_limit(0)
            self.fail("Should have raised exception for pages=0")
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            self.common.print(msg, e)
            self.assertEqual(e.get_error_code(), amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, msg)

        # Test resetting TTM pages limit
        msg = "\t### amdsmi_reset_ttm_pages_limit() (DRY_RUN):"
        try:
            ret = amdsmi.amdsmi_reset_ttm_pages_limit()
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            self.fail(f"Failed to reset TTM in DRY_RUN mode: {e}")
        return


if __name__ == "__main__":
    verbose = common.VERBOSITY_NORMAL
    # Parse verbosity from command line.
    # -v/-vv/--verbose all select VERBOSITY_VERBOSE; -q/--quiet selects QUIET.
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
        verbose = common.VERBOSITY_VERBOSE

    # If no -k or --keyword argument is given, print all available tests.
    # Do this before the -h check so the test list appears above unittest's help output.
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)

    # Skip legend/title/"Running" preamble when the user just wants help text.
    if "-h" in sys.argv or "--help" in sys.argv:
        unittest.main()

    # Only show the dot-character legend when not in verbose mode; in verbose
    # mode each test prints its own result line so the dot legend is irrelevant.
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()

    if verbose > common.VERBOSITY_QUIET:
        print(f"AMD SMI Integration Tests\n")
        print("Running tests...\n")

    # Detect if ran without sudo or root privileges
    if os.geteuid() != 0:
        print(
            "Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n",
            file=sys.stderr,
        )
        print("Please relaunch with elevated privileges.\n", file=sys.stderr)
        sys.exit(1)

    # WARNING: Future developers! Please read. :)
    # Avoid per-test ASIC skipping because:
    # 1) Masks API bugs — we should verify the API handles unsupported cases correctly, not skip past them.
    # 2) Unknown behavior — we don't know what the API actually does in unsupported configurations if we never run it.
    # 3) Tests may be wrong — skipped tests are never validated and can silently rot.
    # 4) Hides driver/firmware gaps — a missing implementation looks the same as "not supported"/etc...
    # 5) False coverage — a suite that skips isn't really passing, it's just not running.
    # 6) Skips become permanent — they rarely get revisited and turn into long-term technical debt.
    #
    # Preferred approach: Run the test. If the API returns an "unsupported" result, assert that response explicitly
    # rather than skipping.

    # ---------------------------------------------------------------------------
    # Skip approaches to AVOID in tests
    #
    # Approach                        | Example                                         | Problem
    # --------------------------------|-------------------------------------------------|------------------------------------------
    # Unconditional TODO skip         | if self.common.TODO_SKIP_FAIL: skipTest(...)    | Never runs; API behavior stays unknown
    # GFX filter / target version     | if gfx in GFX_FILTER: skipTest(...)             | Explicit but still hides API behavior
    # Feature flag skip               | if not gpu_supports_feature: skipTest(...)      | Doesn't verify API returns correct error
    # Exception swallow               | except Exception: pass                          | Hides failures silently; worse than skip
    # Broad except + skip             | except Exception: skipTest(...)                 | Skips on *any* error, including test bugs
    # Commented-out assertions        | # self.assertEqual(...)                         | Test always passes; nothing is verified
    #
    # Preferred approach:
    #   Run the test on all ASICs. If the feature is unsupported, assert the API
    #   returns the expected error code rather than skipping.
    #
    #   try:
    #       result = amdsmi.amdsmi_get_some_feature(processor)
    #       self.assertIsNotNone(result)
    #   except amdsmi.AmdSmiLibraryException as e:
    #       self.assertEqual(e.get_error_code(), amdsmi.AmdSmiStatus.AMDSMI_STATUS_NOT_SUPPORTED)
    # ---------------------------------------------------------------------------

    runner = unittest.TextTestRunner(verbosity=common.make_runner_verbosity(verbose))

    common.expand_glob_k_arg(globals())
    unittest.main(testRunner=runner)
