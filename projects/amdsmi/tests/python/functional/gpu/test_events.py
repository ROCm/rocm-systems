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
"""GPU events: GPU counter and event notification."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuEvents(unittest.TestCase):
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

    def test_gpu_counter(self):
        self.common.print_func_name("")

        results = {}
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            results[i] = {}
            for event_group_name, event_group, event_group_cond in common.EVENT_GROUPS:
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

                for event_type_name, event_type, event_type_cond in common.EVENT_TYPES:
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
                        common.COUNTER_COMMANDS[0]
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
                        common.COUNTER_COMMANDS[1]
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
