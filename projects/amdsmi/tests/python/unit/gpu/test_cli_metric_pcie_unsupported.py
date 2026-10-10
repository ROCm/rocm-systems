#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PCIe link output when GPU metrics and independent counters are unavailable."""

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))
_METRIC_PATH = Path(_CLI_DIR) / "subcommands" / "metric.py" if _CLI_DIR else None
_METRIC_COUNTERS = (
    "bandwidth",
    "replay_count",
    "l0_to_recovery_count",
    "replay_roll_over_count",
    "nak_received_count",
    "nak_sent_count",
    "lc_perf_other_end_recovery_count",
)
_THROUGHPUT_FIELDS = ("current_bandwidth_sent", "current_bandwidth_received", "max_packet_size")


class _FakeLibraryException(Exception):
    def get_error_info(self) -> str:
        return str(self)


def _fake_modules() -> dict:
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        AMDSMI_MAX_RAIL_INDEX=7,
        amdsmi_get_pcie_info=Mock(),
        _NA_amdsmi_get_gpu_metrics_info=Mock(return_value={"is_apu": False}),
    )
    for name in (
        "amdsmi_get_gpu_metrics_header_info",
        "amdsmi_get_gpu_metrics_info",
        "amdsmi_get_gpu_pci_throughput",
        "amdsmi_get_gpu_pci_replay_counter",
    ):
        setattr(interface, name, Mock(side_effect=_FakeLibraryException("NOT_SUPPORTED")))
    exception = fake_module(
        "amdsmi.amdsmi_exception",
        AmdSmiException=_FakeLibraryException,
        AmdSmiLibraryException=_FakeLibraryException,
    )
    return {
        "amdsmi": fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception),
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
    }


def _build_args() -> argparse.Namespace:
    sections = (
        "usage power clock temperature voltage ecc ecc_blocks base_board gpu_board "
        "mem_usage fan voltage_curve overdrive perf_level xgmi_err energy throttle partition"
    ).split()
    return argparse.Namespace(
        gpu=object(), pcie=True, watch=False, loglevel="INFO", **dict.fromkeys(sections, False)
    )


class TestCliMetricPcieUnsupported(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _METRIC_PATH or not _METRIC_PATH.is_file():
            raise unittest.SkipTest("amd-smi CLI metric module not available")
        modules = _fake_modules()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        try:
            cls.metric_module = load_cli_module("metric_pcie_under_test", _METRIC_PATH)
        except BaseException:
            cls.tearDownClass()
            raise

    def _build_commands(self, fmt: str) -> object:
        commands = self.metric_module.MetricCommands()
        commands.group_check_printed = True
        commands.helpers = SimpleNamespace(
            is_hypervisor=lambda: False,
            is_windows=lambda: False,
            is_baremetal=lambda: True,
            is_linux=lambda: True,
            os_info=lambda: "test-os",
            get_gpu_id_from_device_handle=lambda _handle: 0,
            _get_metric_version_and_partition_info=lambda *_args: {"num_partition": 1},
        )
        commands.logger = SimpleNamespace(
            is_human_readable_format=lambda: fmt == "human",
            is_json_format=lambda: fmt == "json",
            is_csv_format=lambda: fmt == "csv",
            store_output=Mock(),
            print_output=Mock(),
            store_gpu_json_output=[],
        )
        return commands

    def _run_metric(self, fmt: str, speed: int, loglevel: str = "INFO") -> dict:
        pcie_metric = {"pcie_" + name: "N/A" for name in _METRIC_COUNTERS}
        pcie_metric.update(pcie_width=16, pcie_speed=speed)
        self.interface.amdsmi_get_pcie_info.return_value = {"pcie_metric": pcie_metric}
        for value in vars(self.interface).values():
            if isinstance(value, Mock):
                value.reset_mock()
        commands = self._build_commands(fmt)
        args = _build_args()
        args.loglevel = loglevel
        commands.metric_gpu(args)
        for name in (
            "amdsmi_get_pcie_info",
            "amdsmi_get_gpu_pci_throughput",
            "amdsmi_get_gpu_pci_replay_counter",
        ):
            getattr(self.interface, name).assert_called_once_with(args.gpu)
        self.interface.amdsmi_get_gpu_metrics_info.assert_called_with(args.gpu)
        self.assertEqual(
            self.interface.amdsmi_get_gpu_metrics_info.call_count, 2 if loglevel == "DEBUG" else 1
        )
        if loglevel == "DEBUG":
            self.interface.amdsmi_get_gpu_metrics_header_info.assert_called_once_with(args.gpu)
        self.interface._NA_amdsmi_get_gpu_metrics_info.assert_called_once_with()
        self.assertEqual(len(commands.logger.store_gpu_json_output), 1)
        values = commands.logger.store_gpu_json_output[0]
        commands.logger.store_output.assert_called_once_with(args.gpu, "values", values)
        if fmt == "json":
            self.assertEqual(set(values), {"gpu", "pcie"})
            self.assertEqual(values["gpu"], 0)
            commands.logger.print_output.assert_not_called()
        else:
            self.assertEqual(set(values), {"pcie"})
            commands.logger.print_output.assert_called_once_with(watching_output=False)
        return values["pcie"]

    def _check_output(self, fmt: str, loglevel: str = "INFO") -> None:
        for speed, expected_speed in ((32000, 32), (2500, 2.5)):
            with self.subTest(format=fmt, speed=speed):
                pcie = self._run_metric(fmt=fmt, speed=speed, loglevel=loglevel)
                expected = dict.fromkeys(_METRIC_COUNTERS + _THROUGHPUT_FIELDS, "N/A")
                expected["width"] = 16
                if fmt == "human":
                    expected["speed"] = f"{expected_speed} GT/s"
                elif fmt == "json":
                    expected["speed"] = {"value": expected_speed, "unit": "GT/s"}
                else:
                    expected["speed"] = expected_speed
                self.assertEqual(pcie, expected)
                self.assertIs(type(pcie["width"]), int)
                if fmt == "json":
                    self.assertIs(type(pcie["speed"]["value"]), type(expected_speed))
                elif fmt == "csv":
                    self.assertIs(type(pcie["speed"]), type(expected_speed))

    def test_human_link_with_unavailable_counters(self) -> None:
        self._check_output("human")

    def test_json_link_with_unavailable_counters(self) -> None:
        self._check_output("json")

    def test_csv_link_with_unavailable_counters(self) -> None:
        self._check_output("csv")

    def test_debug_link_with_unavailable_counters(self) -> None:
        for fmt in ("human", "json", "csv"):
            self._check_output(fmt, loglevel="DEBUG")
