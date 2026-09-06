#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU events: GPU counter and event notification."""

import os
import unittest
from collections import defaultdict

import common.common as common
from common.common import amdsmi


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

    def _run_counter(self, gpu, gpu_idx, event_type, type_name, supported):
        """Take one counter through create/start/read/stop/destroy.

        Runs on unsupported hardware too: *supported* decides whether each call is
        expected to succeed or to refuse. Returns the counter that was read, or None.
        """
        handle = None
        msg = _api_msg("amdsmi_gpu_create_counter", gpu=gpu_idx, event_type=type_name)
        with self.common.expect_status(msg, amdsmi.AmdSmiStatus.SUCCESS):
            handle = amdsmi.amdsmi_gpu_create_counter(gpu, event_type)
        if handle is None:
            return None

        if supported:
            start_accept = amdsmi.AmdSmiStatus.SUCCESS
        else:
            # No perf event source to open, so the sysfs read behind it returns
            # ENOENT, which the library maps to NOT_SUPPORTED.
            start_accept = amdsmi.AmdSmiStatus.NOT_SUPPORTED

        started = self._control_counter(
            gpu_idx, type_name, handle, amdsmi.AmdSmiCounterCommand.CMD_START, start_accept
        )

        # Nothing was opened yet, so a teardown now reports the missing fd.
        teardown_accept = amdsmi.AmdSmiStatus.FILE_ERROR
        counter = None
        try:
            if started:
                read_accept = amdsmi.AmdSmiStatus.SUCCESS
                teardown_accept = amdsmi.AmdSmiStatus.SUCCESS
            else:
                # fd_ stayed -1, so every call below hits EBADF. read reports that as
                # UNEXPECTED_SIZE because it collapses every error into that one status,
                # while stop and destroy map it to FILE_ERROR.
                read_accept = amdsmi.AmdSmiStatus.UNEXPECTED_SIZE

            msg = _api_msg("amdsmi_gpu_read_counter", gpu=gpu_idx, event_type=type_name)
            with self.common.expect_status(msg, read_accept):
                counter = amdsmi.amdsmi_gpu_read_counter(handle)
            if counter is not None:
                self.common.print(f"\t\t{counter}")

            self._control_counter(
                gpu_idx, type_name, handle, amdsmi.AmdSmiCounterCommand.CMD_STOP, teardown_accept
            )
        finally:
            msg = _api_msg("amdsmi_gpu_destroy_counter", gpu=gpu_idx, event_type=type_name)
            with self.common.expect_status(msg, teardown_accept):
                amdsmi.amdsmi_gpu_destroy_counter(handle)

        return counter

    def test_gpu_counter(self):
        """Exercise every xGMI/DF counter end to end, on supported and unsupported hardware."""
        # TODO(amdsmi_team): this only judges the status of each call, so nothing here
        # proves a counter counted anything. Improve it by running on xGMI-capable
        # hardware with a workload driving xGMI traffic, then assert the value read
        # back over a sample window.
        self.common.print_func_name("")

        # create_counter takes an event type, not a group, so collect the types per group.
        types_by_group = defaultdict(list)
        for type_name, event_type, _ in common.EVENT_TYPES:
            types_by_group[_event_group(event_type)].append((type_name, event_type))

        # Nothing iterates GRP_INVALID, so a type landing there would go untested. That
        # means _event_group has drifted from the library's EvtGrpFromEvtID.
        unclassified = types_by_group[amdsmi.AmdSmiEventGroup.GRP_INVALID]
        self.assertFalse(unclassified, f"event types not mapped to a group: {unclassified}")

        results = {}
        with self.common.status_sweep():
            for gpu_idx, gpu in enumerate(self.common.processors):
                self.common.print_device_header(gpu_idx)
                results[gpu_idx] = {}

                for group_name, group, _ in common.EVENT_GROUPS:
                    supported, available = self._probe_counter_group(
                        gpu, gpu_idx, group, group_name
                    )
                    events = {}
                    supp_avail_print = (
                        f" | Supported: {bool(supported)}, Available: {bool(available)}"
                    )
                    self.common.print(
                        f"\t\tRunning counters for {group_name} events {supp_avail_print}"
                    )
                    for type_name, event_type in types_by_group[group]:
                        counter = self._run_counter(gpu, gpu_idx, event_type, type_name, supported)
                        events[type_name] = counter

                    results[gpu_idx][group_name] = {
                        "supported": supported,
                        "available_counters": available,
                        "events": events,
                    }

        self.common.print("gpu counter results", results)
        return

    def test_gpu_event(self):
        self.common.print_func_name("")

        # Opt-in: amdsmi_reset_gpu() is destructive and needs root, so only run it
        # when AMDSMI_TEST_TRIGGER_RESET is set to actually generate reset events.
        # Ex. sudo AMDSMI_TEST_TRIGGER_RESET=1 /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "test_events" -v
        # (Note: filter not required, but helpful for fast checks)
        # Without AMDSMI_TEST_TRIGGER_RESET=1 the test will still run and check for events,
        # but will not generate any reset events.
        trigger_reset = bool(os.environ.get("AMDSMI_TEST_TRIGGER_RESET"))
        if trigger_reset and os.geteuid() != 0:
            self.skipTest("AMDSMI_TEST_TRIGGER_RESET requires root to reset the GPU.")

        # Enable all event types (bit position starts at 1).
        mask = 0
        for event_type in amdsmi.AmdSmiEvtNotificationType:
            if event_type != amdsmi.AmdSmiEvtNotificationType.NONE:
                mask |= 1 << (int(event_type) - 1)
        # get drains buffered events first, then polls the rest of the timeout for new
        # ones; keep it short so an idle GPU doesn't stall the run.
        timeout_ms = 1000

        # No events fire on an idle GPU, so NO_DATA is a passing outcome for get.
        get_expected = [self.common.PASS, "AMDSMI_STATUS_NO_DATA"]

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
            msg = f"\t### amdsmi_set_gpu_event_notification_mask(gpu={i}, mask=0x{mask:X}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_event_notification_mask(gpu, mask)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Trigger (opt-in): reset the GPU to generate reset events.
            if trigger_reset:
                msg = f"\t### amdsmi_reset_gpu(gpu={i}):"
                try:
                    ret = amdsmi.amdsmi_reset_gpu(gpu)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
            else:
                self.common.print(
                    f"\tSKIPPED (amdsmi_reset_gpu(gpu={i})): opt-in reset disabled "
                    "\n\t(set AMDSMI_TEST_TRIGGER_RESET=1 as root to generate reset events); "
                    "expecting AMDSMI_STATUS_NO_DATA (or SUCCESS with no events) below."
                )

            # Get
            msg = f"\t### amdsmi_get_gpu_event_notification(timeout_ms={timeout_ms}):"
            seen_events = set()
            gpu_addr = gpu.value  # records carry processor_handle as a raw address (int)
            try:
                ret = amdsmi.amdsmi_get_gpu_event_notification(timeout_ms)
                self.common.print(msg, f"num_elem={ret['num_elem']}")
                # Decode each record: 'event' is a raw enum value, 'message' holds details.
                for event_data in ret["data"]:
                    # get is process-global, only attribute events for the current GPU.
                    if event_data["processor_handle"] != gpu_addr:
                        self.common.print(f"\t\tSKIPPED event for GPU {i}: {event_data}")
                        continue
                    try:
                        event_name = amdsmi.AmdSmiEvtNotificationType(event_data["event"]).name
                    except ValueError:
                        event_name = f"UNKNOWN({event_data['event']})"
                    seen_events.add(event_name)
                    # message fields are double-space separated; render as "a | b | c".
                    fields = [f.strip() for f in event_data["message"].split("  ") if f.strip()]
                    details = " | ".join(fields) if fields else "(no details)"
                    self.common.print(f"\t\t{event_name:<16} {details}")
                self.common.check_ret("", "", get_expected)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, get_expected):
                    self.raise_exception = e

            # A forced reset must produce both reset events.
            if trigger_reset:
                for expected_event in ("GPU_PRE_RESET", "GPU_POST_RESET"):
                    if expected_event not in seen_events:
                        self.raise_exception = AssertionError(
                            f"Expected {expected_event} after amdsmi_reset_gpu(gpu={i}) "
                            f"but it was not received (saw: {sorted(seen_events)})."
                        )

            # Stop
            msg = f"\t### amdsmi_stop_gpu_event_notification(gpu={i}):"
            try:
                ret = amdsmi.amdsmi_stop_gpu_event_notification(gpu)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration
