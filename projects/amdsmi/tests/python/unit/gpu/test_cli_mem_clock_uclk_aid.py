#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based tests for the memory clock reported by ``amd-smi monitor`` and ``metric``.

Parts with per-AID UCLK telemetry (gpu_metrics v1.9) leave ``current_uclk`` unset
and publish the memory clock in ``current_uclk_aid``. Both CLI paths must fall
back to that array instead of reporting ``N/A``.
"""

import argparse
import enum
import os
import types
import unittest

from common.common import amdsmi_path, cli_search_order, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
METRIC_PATH = os.path.join(_CLI_DIR, "subcommands", "metric.py") if _CLI_DIR else None
MONITOR_PATH = os.path.join(_CLI_DIR, "subcommands", "monitor.py") if _CLI_DIR else None

_UCLK_AID_ONLY = {"current_uclk": "N/A", "current_uclk_aid": [1650, 1650]}


class _FakeAmdSmiException(Exception):
    pass


class _FakeLibraryException(_FakeAmdSmiException):
    def __init__(self, message="AMDSMI_STATUS_NOT_SUPPORTED"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


class _Metrics(dict):
    """gpu_metrics payload; any field not seeded reads as N/A."""

    def __missing__(self, _key):
        return "N/A"


class _EnumMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return f"{cls.__name__}.{name}"

    def __getitem__(cls, name):
        return f"{cls.__name__}.{name}"


class _FakeEnum(metaclass=_EnumMeta):
    pass


def _build_fake_modules():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_NUM_GFX_CLKS = 8
    interface.AMDSMI_MAX_NUM_CLKS = 4
    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS = 3
    interface.amdsmi_get_gpu_metrics_info = lambda _handle: _Metrics()
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: _Metrics()
    interface.amdsmi_get_gpu_partition_metrics_info = lambda _handle: None
    interface.AmdSmiLibraryException = _FakeLibraryException

    def _unsupported(name):
        if name.startswith("AmdSmi"):
            return _FakeEnum
        if name.startswith("amdsmi_"):

            def _raise(*_args, **_kwargs):
                raise _FakeLibraryException(f"{name}: AMDSMI_STATUS_NOT_SUPPORTED")

            return _raise
        raise AttributeError(name)

    interface.__getattr__ = _unsupported
    exception.AmdSmiException = _FakeAmdSmiException
    exception.AmdSmiLibraryException = _FakeLibraryException
    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    # monitor.py imports AMDSMIHelpers at load; the real module initializes the library.
    helpers_module = types.ModuleType("amdsmi_helpers")

    class AMDSMIHelpers:
        class SI_Unit(enum.Enum):
            MICRO = 1

    helpers_module.AMDSMIHelpers = AMDSMIHelpers

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi_helpers": helpers_module,
    }


class _FakeLogger:
    def __init__(self, fmt="human"):
        self.fmt = fmt
        self.captured_values = None
        self.table_header = ""
        self.destination = "stdout"
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self.fmt == "json"

    def is_csv_format(self):
        return self.fmt == "csv"

    def is_human_readable_format(self):
        return self.fmt == "human"

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass

    def store_watch_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def is_virtual_os(self):
        return False

    def is_hypervisor(self):
        return False

    def is_windows(self):
        return False

    def is_baremetal(self):
        return True

    def is_linux(self):
        return True

    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {"partition_id": 0, "num_partition": 1}

    def get_gpu_board_temperatures(self, *args, **kwargs):
        return {}

    def get_base_board_temperatures(self, *args, **kwargs):
        return {}

    def build_xcp_dict(self, *args, **kwargs):
        return {}

    def unit_format(self, logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(logger, v, unit) for v in value]
        if value == "N/A":
            return "N/A"
        if logger.is_json_format():
            return {"value": value, "unit": unit}
        return f"{value} {unit}".rstrip()


_METRIC_SECTIONS = (
    "partition", "usage", "power", "temperature", "voltage", "pcie", "ecc", "ecc_blocks",
    "base_board", "gpu_board", "mem_usage", "fan", "voltage_curve", "overdrive", "perf_level",
    "xgmi_err", "energy", "throttle", "violation", "schedule", "guard", "guest_data", "fb_usage",
    "xgmi",
)  # fmt: skip

_MONITOR_SECTIONS = (
    "power_usage", "temperature", "base_board_temps", "gpu_board_temps", "gfx", "encoder",
    "decoder", "ecc", "vram_usage", "pcie", "process", "violation",
)  # fmt: skip


class TestCliMemClockUclkAid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (METRIC_PATH, MONITOR_PATH):
            if not path or not os.path.isfile(path):
                raise unittest.SkipTest(
                    f"amd-smi CLI subcommands not found (looked in {_CLI_DIR or amdsmi_path})"
                )
        modules = _build_fake_modules()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.metric_module = load_cli_module(
            "metric_uclk_under_test", METRIC_PATH, sys_path_dir=_CLI_DIR
        )
        cls.monitor_module = load_cli_module(
            "monitor_uclk_under_test", MONITOR_PATH, sys_path_dir=_CLI_DIR
        )

    def _commands(self, cls, metrics, fmt):
        payload = _Metrics(metrics)
        self.interface.amdsmi_get_gpu_metrics_info = lambda _handle: payload
        commands = object.__new__(cls)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        commands.device_handles = []
        return commands

    def _monitor_mem_clock(self, metrics, fmt="human"):
        commands = self._commands(self.monitor_module.MonitorCommands, metrics, fmt)
        args = argparse.Namespace(
            gpu=object(), watch=False, watch_time=None, iterations=None, loglevel="INFO", mem=True
        )
        for section in _MONITOR_SECTIONS:
            setattr(args, section, False)
        commands.monitor(args)
        self.assertIsNotNone(commands.logger.captured_values)
        return commands.logger.captured_values["mem_clock"]

    def _metric_mem_clock(self, metrics, fmt="human"):
        commands = self._commands(self.metric_module.MetricCommands, metrics, fmt)
        args = argparse.Namespace(
            gpu=object(), watch=False, watch_time=None, iterations=None, loglevel="INFO", clock=True
        )
        for section in _METRIC_SECTIONS:
            setattr(args, section, False)
        commands.metric_gpu(args)
        self.assertIsNotNone(commands.logger.captured_values)
        return commands.logger.captured_values["clock"]["mem_0"]["clk"]

    def test_monitor_mem_clock_falls_back_to_uclk_aid(self):
        self.assertEqual(self._monitor_mem_clock(_UCLK_AID_ONLY), "1650 MHz")
        self.assertEqual(
            self._monitor_mem_clock(_UCLK_AID_ONLY, fmt="json"), {"value": 1650, "unit": "MHz"}
        )
        self.assertEqual(self._monitor_mem_clock(_UCLK_AID_ONLY, fmt="csv"), 1650)

    def test_metric_mem_clock_falls_back_to_uclk_aid(self):
        self.assertEqual(self._metric_mem_clock(_UCLK_AID_ONLY), "1650 MHz")
        self.assertEqual(
            self._metric_mem_clock(_UCLK_AID_ONLY, fmt="json"), {"value": 1650, "unit": "MHz"}
        )

    def test_uclk_aid_skips_unreported_entries(self):
        metrics = {"current_uclk": "N/A", "current_uclk_aid": ["N/A", 1800]}
        self.assertEqual(self._monitor_mem_clock(metrics), "1800 MHz")
        self.assertEqual(self._metric_mem_clock(metrics), "1800 MHz")

    def test_current_uclk_takes_precedence(self):
        metrics = {"current_uclk": 1300, "current_uclk_aid": [1650, 1650]}
        self.assertEqual(self._monitor_mem_clock(metrics), "1300 MHz")
        self.assertEqual(self._metric_mem_clock(metrics), "1300 MHz")

    def test_mem_clock_stays_na_without_any_source(self):
        for aid in ("N/A", ["N/A", "N/A"]):
            with self.subTest(current_uclk_aid=aid):
                metrics = {"current_uclk": "N/A", "current_uclk_aid": aid}
                self.assertEqual(self._monitor_mem_clock(metrics), "N/A")
                self.assertEqual(self._metric_mem_clock(metrics), "N/A")
