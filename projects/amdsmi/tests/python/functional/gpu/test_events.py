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

import time
import unittest
from collections import defaultdict

import common.common as common
from common.common import amdsmi

# TODO(amdsmi_team): drive xGMI traffic across this window so the counter reads a
# non-zero value. Until then only time_running is checked, which proves the perf event
# was scheduled but not that it counted the right thing.
_SAMPLE_WINDOW_S = 0.1


def _event_group(event_type):
    """Mirror the library's fixed type->group binding (EvtGrpFromEvtID).

    A type added upstream falls outside both ranges and surfaces as GRP_INVALID rather
    than being folded silently into XGMI.
    """
    types = amdsmi.AmdSmiEventType
    groups = amdsmi.AmdSmiEventGroup
    if types.XGMI_0_NOP_TX <= event_type <= types.XGMI_1_BEATS_TX:
        return groups.XGMI
    if types.XGMI_DATA_OUT_0 <= event_type <= types.XGMI_DATA_OUT_5:
        return groups.XGMI_DATA_OUT
    return groups.GRP_INVALID


def _api_msg(api, **args):
    """Render the '### api(arg=value):' header printed above each call's verdict."""
    rendered = ", ".join(f"{name}={value}" for name, value in args.items())
    return f"\t### {api}({rendered}):"


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

    def _probe_counter_group(self, gpu, gpu_idx, group, group_name):
        """Ask whether an event group works here; returns (supported, available)."""
        if group == amdsmi.AmdSmiEventGroup.GRP_INVALID:
            # GRP_INVALID must never be reported as usable, so SUCCESS is left out.
            accept = [amdsmi.AmdSmiStatus.NOT_SUPPORTED, amdsmi.AmdSmiStatus.INVAL]
        else:
            # An ASIC without DF/xGMI perf counters may legitimately answer "no".
            accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]

        supported = False
        msg = _api_msg("amdsmi_gpu_counter_group_supported", gpu=gpu_idx, event_group=group_name)
        with self.common.expect_status(msg, accept):
            amdsmi.amdsmi_gpu_counter_group_supported(gpu, group)
            supported = True

        available = 0
        msg = _api_msg("amdsmi_get_gpu_available_counters", gpu=gpu_idx, event_group=group_name)
        with self.common.expect_status(msg, accept):
            available = amdsmi.amdsmi_get_gpu_available_counters(gpu, group)
        self.common.print(f"\t\tavailable counters: {available}")

        return supported, available

    def _control_counter(self, gpu_idx, type_name, handle, command, accept):
        """Start or stop a counter, returns True when the command was accepted."""
        msg = _api_msg(
            "amdsmi_gpu_control_counter",
            gpu=gpu_idx,
            event_type=type_name,
            counter_command=command.name,
        )
        controlled = False
        with self.common.expect_status(msg, accept):
            amdsmi.amdsmi_gpu_control_counter(handle, command)
            controlled = True
        return controlled

    def _run_counter(self, gpu, gpu_idx, event_type, type_name, usable):
        """Take one counter through create/start/read/stop/destroy.

        Runs on unsupported hardware too: *usable* decides whether each call is expected
        to succeed or to refuse. Returns the counter that was read, or None.
        """
        handle = None
        msg = _api_msg("amdsmi_gpu_create_counter", gpu=gpu_idx, event_type=type_name)
        create_accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.OUT_OF_RESOURCES]
        with self.common.expect_status(msg, create_accept):
            handle = amdsmi.amdsmi_gpu_create_counter(gpu, event_type)
        if handle is None:
            return None

        if usable:
            start_accept = amdsmi.AmdSmiStatus.SUCCESS
        else:
            start_accept = amdsmi.AmdSmiStatus.NOT_SUPPORTED

        started = self._control_counter(
            gpu_idx, type_name, handle, amdsmi.AmdSmiCounterCommand.CMD_START, start_accept
        )

        if started:
            time.sleep(_SAMPLE_WINDOW_S)
            read_accept = amdsmi.AmdSmiStatus.SUCCESS
            teardown_accept = amdsmi.AmdSmiStatus.SUCCESS
        else:
            # No perf fd to read or close, which each path reports as its own status.
            read_accept = amdsmi.AmdSmiStatus.UNEXPECTED_SIZE
            teardown_accept = amdsmi.AmdSmiStatus.FILE_ERROR

        counter = None
        msg = _api_msg("amdsmi_gpu_read_counter", gpu=gpu_idx, event_type=type_name)
        with self.common.expect_status(msg, read_accept):
            counter = amdsmi.amdsmi_gpu_read_counter(handle)
        if counter is not None:
            self.common.print(f"\t\t{counter}")

        self._control_counter(
            gpu_idx, type_name, handle, amdsmi.AmdSmiCounterCommand.CMD_STOP, teardown_accept
        )

        msg = _api_msg("amdsmi_gpu_destroy_counter", gpu=gpu_idx, event_type=type_name)
        with self.common.expect_status(msg, teardown_accept):
            amdsmi.amdsmi_gpu_destroy_counter(handle)

        return counter

    def test_gpu_counter(self):
        """Exercise every xGMI/DF counter end to end, on supported and unsupported hardware."""
        self.common.print_func_name("")

        # create_counter takes an event type, not a group, so collect the types per group.
        types_by_group = defaultdict(list)
        for type_name, event_type, _type_cond in common.EVENT_TYPES:
            types_by_group[_event_group(event_type)].append((type_name, event_type))

        results = {}
        idle_counters = []
        with self.common.status_sweep():
            for gpu_idx, gpu in enumerate(self.common.processors):
                self.common.print_device_header(gpu_idx)
                results[gpu_idx] = {}

                for group_name, group, _group_cond in common.EVENT_GROUPS:
                    supported, available = self._probe_counter_group(
                        gpu, gpu_idx, group, group_name
                    )
                    usable = bool(supported and available)
                    events = {}

                    for type_name, event_type in types_by_group[group]:
                        counter = self._run_counter(gpu, gpu_idx, event_type, type_name, usable)
                        events[type_name] = counter
                        if counter and not counter["time_running"]:
                            idle_counters.append(f"gpu={gpu_idx} {type_name}")

                    results[gpu_idx][group_name] = {
                        "supported": supported,
                        "available_counters": available,
                        "events": events,
                    }

        self.common.print("gpu counter results", results)

        if idle_counters:
            raise AssertionError(
                f"{len(idle_counters)} counter(s) reported time_running=0 after a "
                f"{_SAMPLE_WINDOW_S}s sample window, so the perf event never ran: "
                f"{', '.join(idle_counters)}"
            )
        return

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
