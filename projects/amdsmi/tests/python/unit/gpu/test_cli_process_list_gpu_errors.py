#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A GPU whose process list fails must not hide the other GPUs' processes.

``amdsmi_get_gpu_process_list()`` reports a list it cannot read (for example
``AMDSMI_STATUS_RETRY`` while processes keep starting) instead of returning an
empty list. ``amd-smi process`` and ``amd-smi monitor -q`` loop over the GPUs, so
they show that GPU's error, keep the other GPUs, and record the status for the
exit code. NO_PERM stays a command-wide error. The real CLI logger prints the
output, so each format is checked as the user sees it.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import types
import unittest

from common.common import (
    add_class_cleanup,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else None
PROCESS_PATH = os.path.join(_CLI_DIR, "subcommands", "process.py") if _CLI_DIR else None
MONITOR_PATH = os.path.join(_CLI_DIR, "subcommands", "monitor.py") if _CLI_DIR else None

_STATUS_NOT_SUPPORTED = 2
_STATUS_RETRY = 9
_STATUS_NO_PERM = 10
_STATUS_NAMES = {
    _STATUS_NOT_SUPPORTED: "AMDSMI_STATUS_NOT_SUPPORTED",
    _STATUS_RETRY: "AMDSMI_STATUS_RETRY",
    _STATUS_NO_PERM: "AMDSMI_STATUS_NO_PERM",
}
_RETRY_TEXT = "[AMDSMI_STATUS_RETRY] Unable to get the process list"
_PID = 4242


class _FakeLibraryException(Exception):
    def __init__(self, err_code):
        super().__init__(err_code)
        self.err_code = err_code
        self.err_info = f"{_STATUS_NAMES[err_code]} - test description"

    def get_error_code(self):
        return self.err_code

    def get_error_info(self, detailed=True):
        return self.err_info if detailed else self.err_info.split(" - ")[0]


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


def _unsupported(*_args, **_kwargs):
    raise _FakeLibraryException(_STATUS_NOT_SUPPORTED)


def _fake_amdsmi_modules():
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED,
        AMDSMI_STATUS_RETRY=_STATUS_RETRY,
        AMDSMI_STATUS_NO_PERM=_STATUS_NO_PERM,
        AMDSMI_STATUS_INVAL=5,
        AMDSMI_STATUS_UNEXPECTED_SIZE=42,
        AMDSMI_STATUS_UNEXPECTED_DATA=43,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        MaxUIntegerTypes=types.SimpleNamespace(UINT64_T=0xFFFFFFFFFFFFFFFF),
        AmdSmiInitFlags=_FakeInitFlags,
        AmdSmiLibraryException=_FakeLibraryException,
        AmdSmiParameterException=type("AmdSmiParameterException", (Exception,), {}),
        amdsmi_init=lambda _flag: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: ["gpu0", "gpu1"],
        _NA_amdsmi_get_gpu_metrics_info=dict,
    )

    # Every other library getter monitor() calls reports NOT_SUPPORTED.
    def _getattr(name):
        if name.startswith("amdsmi_get_"):
            return _unsupported
        raise AttributeError(name)

    interface.__getattr__ = _getattr
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


def _restore_import_side_effects(had_limit, limit, handlers, level):
    """Undo the two changes importing amdsmi_init makes that later tests would
    see (sys.tracebacklimit and the root logger). Its signal handlers and exit
    hook stay."""
    if had_limit:
        sys.tracebacklimit = limit
    elif hasattr(sys, "tracebacklimit"):
        del sys.tracebacklimit
    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level)


def _process_info():
    return {
        "name": "worker",
        "pid": _PID,
        "mem": 3,
        "engine_usage": {"gfx": 0, "enc": 0},
        "memory_usage": {"gtt_mem": 1, "cpu_mem": 0, "vram_mem": 2},
        "cu_occupancy": "N/A",
        "sdma_usage": "N/A",
        "evicted_time": "N/A",
    }


class TestCliProcessListGpuErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(
            path and os.path.isfile(path) for path in (PROCESS_PATH, MONITOR_PATH, LOGGER_PATH)
        ):
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = _fake_amdsmi_modules()
        modules.update(
            {
                name: None
                for name in ("amdsmi_init", "amdsmi_helpers", "amdsmi_cli_exceptions", "BDF")
            }
        )
        stub_modules(cls, modules)
        root = logging.getLogger()
        add_class_cleanup(
            cls,
            _restore_import_side_effects,
            hasattr(sys, "tracebacklimit"),
            getattr(sys, "tracebacklimit", None),
            list(root.handlers),
            root.level,
        )
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_cli_exceptions
        import amdsmi_helpers

        cls.exceptions = amdsmi_cli_exceptions
        cls.helpers_class = amdsmi_helpers.AMDSMIHelpers
        cls.logger_class = load_cli_module("logger_under_test", LOGGER_PATH, _CLI_DIR).AMDSMILogger
        cls.process_module = load_cli_module("process_under_test", PROCESS_PATH, _CLI_DIR)
        cls.monitor_module = load_cli_module("monitor_under_test", MONITOR_PATH, _CLI_DIR)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def setUp(self):
        self.failures = {}
        self.interface.amdsmi_get_gpu_process_list = self._process_list
        helpers = object.__new__(self.helpers_class)
        helpers._is_virtual_os = False
        helpers.error_collector = self.exceptions.AmdSmiErrorCollector()
        helpers.get_gpu_id_from_device_handle = {"gpu0": 0, "gpu1": 1}.get
        helpers.convert_bytes_to_readable = str
        helpers.convert_time_to_readable = lambda value, _unit: value
        helpers.unit_format = lambda _logger, value, _unit: value
        helpers._get_metric_version_and_partition_info = lambda *_args: {
            "partition_id": 0,
            "num_partition": 1,
        }
        self.helpers = helpers

    def _process_list(self, handle):
        if handle in self.failures:
            raise _FakeLibraryException(self.failures[handle])
        return [_process_info()]

    def _command(self, module, class_name, output_format):
        command = object.__new__(getattr(module, class_name))
        command.helpers = self.helpers
        command.logger = self.logger_class(output_format, "stdout", self.helpers)
        command.device_handles = ["gpu0", "gpu1"]
        command.group_check_printed = True  # monitor(): skip the user-group check
        return command

    def _process_args(self, gpu=None, pid=None):
        return argparse.Namespace(
            gpu=gpu,
            general=False,
            engine=False,
            pid=pid,
            name=None,
            watch=None,
            watch_time=None,
            iterations=None,
            sort_by_pid=False,
        )

    def _monitor_args(self, gpu, sort_by_pid=False):
        # Only ECC and the process table, so monitor() needs no other metric data.
        flags = (
            "power_usage temperature base_board_temps gpu_board_temps gfx mem encoder "
            "decoder vram_usage pcie violation"
        )
        args = argparse.Namespace(**dict.fromkeys(flags.split(), False))
        args.ecc = args.process = True
        args.gpu = gpu
        args.watch = args.watch_time = args.iterations = None
        args.loglevel = "ERROR"
        args.sort_by_pid = sort_by_pid
        return args

    def _by_pid_list_fails(self, status):
        def by_pid(_handles):
            raise _FakeLibraryException(status)

        self.interface.amdsmi_get_gpu_process_list_by_pid = by_pid
        self.addCleanup(delattr, self.interface, "amdsmi_get_gpu_process_list_by_pid")

    def _printed(self, call):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            call()
        return stdout.getvalue()

    def _json_process_lists(self, printed):
        return {entry["gpu"]: entry["process_list"] for entry in json.loads(printed)}

    def test_process_json_keeps_other_gpus_and_shows_the_failed_one(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.process_module, "ProcessCommands", "json")
        lists = self._json_process_lists(
            self._printed(lambda: command.process(self._process_args()))
        )
        self.assertEqual(lists[0][0]["process_info"]["pid"], _PID)
        self.assertEqual(lists[1], [{"process_info": _RETRY_TEXT}])
        self.assertEqual(self.helpers.error_collector.resolve_exit_code(), _STATUS_RETRY)

    def test_process_human_readable_shows_the_failed_gpu(self):
        self.failures["gpu0"] = _STATUS_RETRY
        command = self._command(self.process_module, "ProcessCommands", "human_readable")
        printed = self._printed(lambda: command.process(self._process_args()))
        self.assertIn(_RETRY_TEXT, printed)
        self.assertIn(str(_PID), printed)

    def test_process_pid_filter_keeps_the_error(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.process_module, "ProcessCommands", "json")
        printed = self._printed(lambda: command.process(self._process_args(pid=_PID)))
        lists = self._json_process_lists(printed)
        self.assertEqual(lists[0][0]["process_info"]["pid"], _PID)
        self.assertEqual(lists[1], [{"process_info": _RETRY_TEXT}])

    def test_process_no_perm_still_aborts(self):
        self.failures["gpu1"] = _STATUS_NO_PERM
        command = self._command(self.process_module, "ProcessCommands", "json")
        with self.assertRaises(PermissionError):
            self._printed(lambda: command.process(self._process_args()))
        self.assertFalse(self.helpers.error_collector.has_errors)

    def test_process_single_gpu_error_still_raises(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.process_module, "ProcessCommands", "json")
        with self.assertRaises(_FakeLibraryException) as caught:
            self._printed(lambda: command.process(self._process_args(gpu="gpu1")))
        self.assertEqual(caught.exception.get_error_code(), _STATUS_RETRY)
        self.assertFalse(self.helpers.error_collector.has_errors)

    def test_monitor_human_readable_keeps_other_gpus_and_shows_the_failed_one(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.monitor_module, "MonitorCommands", "human_readable")
        printed = self._printed(lambda: command.monitor(self._monitor_args(["gpu0", "gpu1"])))
        process_table = printed.split("PROCESS INFO")[1]
        self.assertIn(str(_PID), process_table)
        self.assertIn(_RETRY_TEXT, process_table)
        self.assertEqual(self.helpers.error_collector.resolve_exit_code(), _STATUS_RETRY)

    def test_monitor_json_keeps_the_failed_gpus_metrics(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.monitor_module, "MonitorCommands", "json")
        printed = self._printed(lambda: command.monitor(self._monitor_args(["gpu0", "gpu1"])))
        entries = {entry["gpu"]: entry for entry in json.loads(printed)}
        self.assertEqual(entries[0]["process_list"][0]["process_info"]["pid"], _PID)
        self.assertEqual(entries[1]["process_list"], [{"process_info": _RETRY_TEXT}])
        self.assertIn("ecc", entries[1])

    def test_monitor_csv_shows_the_failed_gpu(self):
        self.failures["gpu1"] = _STATUS_RETRY
        command = self._command(self.monitor_module, "MonitorCommands", "csv")
        printed = self._printed(lambda: command.monitor(self._monitor_args(["gpu0", "gpu1"])))
        self.assertIn(_RETRY_TEXT, printed)
        self.assertIn(str(_PID), printed)

    def test_monitor_single_gpu_keeps_its_metrics(self):
        self.failures["gpu0"] = _STATUS_RETRY
        command = self._command(self.monitor_module, "MonitorCommands", "json")
        printed = self._printed(lambda: command.monitor(self._monitor_args("gpu0")))
        entry = json.loads(printed)
        entry = entry[0] if isinstance(entry, list) else entry
        self.assertEqual(entry["process_list"], [{"process_info": _RETRY_TEXT}])
        self.assertIn("ecc", entry)
        self.assertEqual(self.helpers.error_collector.resolve_exit_code(), _STATUS_RETRY)

    def test_monitor_sort_by_pid_json_shows_the_listing_error(self):
        self._by_pid_list_fails(_STATUS_RETRY)
        command = self._command(self.monitor_module, "MonitorCommands", "json")
        args = self._monitor_args(["gpu0", "gpu1"], sort_by_pid=True)
        rows = json.loads(self._printed(lambda: command.monitor(args)))
        self.assertIn({"gpu_index": "N/A", "message": _RETRY_TEXT}, rows)
        self.assertEqual(self.helpers.error_collector.resolve_exit_code(), _STATUS_RETRY)

    def test_monitor_sort_by_pid_human_readable_shows_the_listing_error(self):
        self._by_pid_list_fails(_STATUS_RETRY)
        command = self._command(self.monitor_module, "MonitorCommands", "human_readable")
        args = self._monitor_args(["gpu0", "gpu1"], sort_by_pid=True)
        printed = self._printed(lambda: command.monitor(args))
        self.assertIn(_RETRY_TEXT, printed.partition("PROCESS INFO (grouped by PID)")[2])
        self.assertEqual(self.helpers.error_collector.resolve_exit_code(), _STATUS_RETRY)

    def test_monitor_sort_by_pid_no_perm_aborts(self):
        self._by_pid_list_fails(_STATUS_NO_PERM)
        command = self._command(self.monitor_module, "MonitorCommands", "json")
        args = self._monitor_args(["gpu0", "gpu1"], sort_by_pid=True)
        with self.assertRaises(PermissionError):
            self._printed(lambda: command.monitor(args))
        self.assertFalse(self.helpers.error_collector.has_errors)

    def test_monitor_no_perm_still_aborts(self):
        self.failures["gpu0"] = _STATUS_NO_PERM
        command = self._command(self.monitor_module, "MonitorCommands", "json")
        with self.assertRaises(PermissionError):
            self._printed(lambda: command.monitor(self._monitor_args(["gpu0", "gpu1"])))
        self.assertFalse(self.helpers.error_collector.has_errors)
