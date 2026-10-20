#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU events: GPU counter and event notification."""

import os
import time
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


# Event message fields are separated by two spaces, e.g. "ns: 5  pid: 7".
FIELD_SEPARATOR = "  "
NANOS_PER_SECOND = 1_000_000_000
HEXADECIMAL = 16
RESET_SEQUENCE_FIELD = "reset sequence number"
NO_LONGER_RUNNING = "no longer running"


def _message_parts(message):
    """The "label: value" parts of an event message, without empty ones."""
    return [part.strip() for part in message.split(FIELD_SEPARATOR) if part.strip()]


def _message_fields(message):
    """An event message as {label: value}, e.g. {"ns": "5", "pid": "7"}."""
    fields = {}
    for part in _message_parts(message):
        label, separator, value = part.partition(":")
        if separator:
            fields[label.strip()] = value.strip()
    return fields


def _decimal_and_hex(value):
    """22215 -> "22215 (0x56c7)": the kernel sends ids in hex, sysfs shows them in decimal."""
    return f"{value} (0x{value:x})"


def _boot_time_ns():
    """The clock the kernel stamps events with; it includes time spent suspended."""
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _process_name(pid):
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as comm:
            return comm.read().strip() or NO_LONGER_RUNNING
    except OSError:
        return NO_LONGER_RUNNING


def _reset_groups(processors):
    """Group device indexes that one amdsmi_reset_gpu() resets together (an XGMI hive).

    A device with no hive id, or whose hive id cannot be read, is its own group.
    """
    groups = []
    by_hive = {}
    for i, gpu in enumerate(processors):
        try:
            hive_id = amdsmi.amdsmi_get_xgmi_info(gpu)["xgmi_hive_id"]
        except amdsmi.AmdSmiLibraryException:
            hive_id = 0
        if hive_id == 0:
            groups.append([i])
        elif hive_id in by_hive:
            by_hive[hive_id].append(i)
        else:
            by_hive[hive_id] = [i]
            groups.append(by_hive[hive_id])
    return groups


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
            # Group support is a stat() on the PMU directory, so it says nothing about
            # which events exist. See amdgpu_pmu.c in the Linux kernel:
            #    - gfx906 wires only xgmi_link{0,1}_data_outbound
            #    - gfx908 wires xgmi_link{0-5}_data_outbound
            start_accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
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

        # Checked outside the sweep so the per-call failures above are still reported.
        # A counter only reads back when it opened, so accepting NOT_SUPPORTED on START
        # would otherwise hide a group that claims support while none of it works.
        for gpu_idx, groups in results.items():
            for group_name, group_result in groups.items():
                if not group_result["supported"]:
                    continue
                any_read = any(c is not None for c in group_result["events"].values())
                self.assertTrue(
                    any_read,
                    f"gpu {gpu_idx}: {group_name} reports supported but no counter was read",
                )
        return

    def _check_event_fields(self, gpu_idx, fields, kfd_id, window_start_ns, window_end_ns):
        """Compare an event's fields with sources the library's parser does not use.

        node is the KFD gpu_id (amdsmi_get_gpu_kfd_info kfd_id; a device without one is already
        reported by the caller), ns is on the boot-time clock and must fall inside this test's
        window, and pid is shown with its process name (the kernel reports it in the host's PID
        namespace). Returns the problems found.
        """
        problems = []
        if "node" in fields and kfd_id is not None:
            node = int(fields["node"], HEXADECIMAL)
            matches = node == kfd_id
            self.common.print(
                f"\t\t  node {_decimal_and_hex(node)} {'==' if matches else '!='} "
                f"kfd_id of gpu {gpu_idx}"
            )
            if not matches:
                problems.append(
                    f"gpu {gpu_idx}: node {_decimal_and_hex(node)} is not kfd_id "
                    f"{_decimal_and_hex(kfd_id)}"
                )
        if "ns" in fields:
            ns = int(fields["ns"])
            seconds_after_start = (ns - window_start_ns) / NANOS_PER_SECOND
            self.common.print(f"\t\t  ns is {seconds_after_start:.3f} s after this test started")
            if not window_start_ns <= ns <= window_end_ns:
                problems.append(f"gpu {gpu_idx}: ns {ns} is outside this test's window")
        if "pid" in fields:
            pid = int(fields["pid"])
            this_process = " (this process)" if pid == os.getpid() else ""
            self.common.print(f"\t\t  pid {pid}: {_process_name(pid)}{this_process}")
            if pid <= 0:
                problems.append(f"gpu {gpu_idx}: pid {pid} is not positive")
        return problems

    def test_gpu_event(self):
        self.common.print_func_name("")

        # Opt-in: amdsmi_reset_gpu() is destructive and needs root, so only run it
        # when AMDSMI_TEST_TRIGGER_RESET is set to actually generate reset events.
        # Ex. sudo AMDSMI_TEST_TRIGGER_RESET=1 /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k TestGpuEvents.test_gpu_event -v
        # (Note: filter not required, but helpful for fast checks)
        # Without AMDSMI_TEST_TRIGGER_RESET=1 the test will still run and check for events,
        # but will not generate any reset events.
        trigger_reset = os.environ.get("AMDSMI_TEST_TRIGGER_RESET") == "1"
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

        success = amdsmi.AmdSmiStatus.SUCCESS
        # No events fire on an idle GPU, so NO_DATA is a passing outcome for get.
        get_accept = [success, amdsmi.AmdSmiStatus.NO_DATA]

        missing_events = []
        unexpected_events = []
        field_problems = []
        window_start_ns = _boot_time_ns()
        reset_events = ("GPU_PRE_RESET", "GPU_POST_RESET")
        max_reads = 100  # only a guard; the loop normally ends on NO_DATA
        with self.common.status_sweep():
            # One reset covers a whole XGMI hive, so each group is opened, reset once, read and
            # stopped together.
            for group in _reset_groups(self.common.processors):
                streams = []  # (index, handle) of the group's devices whose stream opened
                for i in group:
                    gpu = self.common.processors[i]
                    self.common.print_device_header(i)

                    initialized = False
                    msg = _api_msg("amdsmi_init_gpu_event_notification", gpu=i)
                    with self.common.expect_status(msg, success):
                        amdsmi.amdsmi_init_gpu_event_notification(gpu)
                        initialized = True
                    if not initialized:
                        continue
                    streams.append((i, gpu))

                    msg = _api_msg(
                        "amdsmi_set_gpu_event_notification_mask", gpu=i, mask=f"0x{mask:X}"
                    )
                    with self.common.expect_status(msg, success):
                        amdsmi.amdsmi_set_gpu_event_notification_mask(gpu, mask)
                if not streams:
                    continue

                # KFD gpu_id per device. Every device must report one: it is what the events'
                # node field is checked against. None stands for a failure recorded here.
                kfd_ids = {}
                for i, gpu in streams:
                    kfd_info = None
                    msg = _api_msg("amdsmi_get_gpu_kfd_info", gpu=i)
                    with self.common.expect_status(msg, success):
                        kfd_info = amdsmi.amdsmi_get_gpu_kfd_info(gpu)
                    kfd_id = None
                    if kfd_info is not None:
                        if isinstance(kfd_info["kfd_id"], int):
                            kfd_id = kfd_info["kfd_id"]
                            self.common.print(f"\t\tkfd_id={_decimal_and_hex(kfd_id)}")
                        else:
                            field_problems.append(
                                f"gpu {i}: kfd_id is not supported ({kfd_info['kfd_id']})"
                            )
                    kfd_ids[i] = kfd_id

                # Trigger (opt-in): reset the group to generate reset events.
                reset_index, reset_gpu = streams[0]
                if trigger_reset:
                    msg = _api_msg("amdsmi_reset_gpu", gpu=reset_index)
                    with self.common.expect_status(msg, success):
                        amdsmi.amdsmi_reset_gpu(reset_gpu)
                else:
                    self.common.print(
                        f"\tSKIPPED (amdsmi_reset_gpu(gpu={reset_index})): opt-in reset disabled "
                        "\n\t(set AMDSMI_TEST_TRIGGER_RESET=1 as root to generate reset events); "
                        "expecting AMDSMI_STATUS_NO_DATA (or SUCCESS with no events) below."
                    )

                # get is process-global and returns records for every open stream, so keep
                # reading until the stream goes quiet.
                records = []
                for _ in range(max_reads):
                    notifications = None
                    msg = _api_msg("amdsmi_get_gpu_event_notification", timeout_ms=timeout_ms)
                    with self.common.expect_status(msg, get_accept):
                        notifications = amdsmi.amdsmi_get_gpu_event_notification(timeout_ms)
                    read_done_ns = _boot_time_ns()
                    if notifications is None:
                        break
                    self.common.print(f"\t\tnum_elem={notifications['num_elem']}")
                    records.extend((record, read_done_ns) for record in notifications["data"])

                # Records carry processor_handle as a raw address (int).
                index_by_handle = {gpu.value: i for i, gpu in streams}
                seen_events = {i: set() for i, _ in streams}
                reset_sequences = {i: {name: set() for name in reset_events} for i, _ in streams}
                for event_data, read_done_ns in records:
                    i = index_by_handle.get(event_data["processor_handle"])
                    if i is None:
                        self.common.print(f"\t\tSKIPPED event for another GPU: {event_data}")
                        continue
                    try:
                        event_name = amdsmi.AmdSmiEvtNotificationType(event_data["event"]).name
                    except ValueError:
                        event_name = f"UNKNOWN({event_data['event']})"
                    seen_events[i].add(event_name)
                    # message fields are double-space separated; render as "a | b | c".
                    fields = _message_parts(event_data["message"])
                    details = " | ".join(fields) if fields else "(no details)"
                    self.common.print(f"\t\t[gpu={i}] {event_name:<16} {details}")

                    event_fields = _message_fields(event_data["message"])
                    field_problems.extend(
                        self._check_event_fields(
                            i, event_fields, kfd_ids[i], window_start_ns, read_done_ns
                        )
                    )
                    if event_name in reset_events and RESET_SEQUENCE_FIELD in event_fields:
                        reset_sequences[i][event_name].add(event_fields[RESET_SEQUENCE_FIELD])

                # PRE and POST of one reset carry the same sequence number.
                for i, by_event in reset_sequences.items():
                    pre, post = (by_event[name] for name in reset_events)
                    if pre != post:
                        field_problems.append(
                            f"gpu {i}: GPU_PRE_RESET sequence numbers {sorted(pre)} "
                            f"differ from GPU_POST_RESET {sorted(post)}"
                        )

                # A forced reset must produce both reset events on every device in the group;
                # without one, neither may appear.
                for i, seen in seen_events.items():
                    if trigger_reset:
                        for expected_event in reset_events:
                            if expected_event not in seen:
                                missing_events.append(
                                    f"gpu {i}: {expected_event} (saw: {sorted(seen)})"
                                )
                    else:
                        stray = [event for event in reset_events if event in seen]
                        if stray:
                            unexpected_events.append(f"gpu {i}: {stray}")

                for i, gpu in streams:
                    msg = _api_msg("amdsmi_stop_gpu_event_notification", gpu=i)
                    with self.common.expect_status(msg, success):
                        amdsmi.amdsmi_stop_gpu_event_notification(gpu)

        # Checked outside the sweep so the per-call failures above are still reported.
        self.assertFalse(
            missing_events, f"reset events not received after amdsmi_reset_gpu: {missing_events}"
        )
        self.assertFalse(
            unexpected_events,
            f"reset events received without amdsmi_reset_gpu: {unexpected_events}",
        )
        self.assertFalse(
            field_problems, f"event fields disagree with other sources: {field_problems}"
        )
        return
