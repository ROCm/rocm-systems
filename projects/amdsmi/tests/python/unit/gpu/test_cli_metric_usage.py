#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Usage selection through the real CLI with mocked device queries."""

import argparse
import ast
import contextlib
import copy
import csv
import io
import itertools
import json
import logging
import re
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules_at_import,
)


class _AmdSmiError(Exception):
    pass


class _LibraryError(_AmdSmiError):
    def get_error_info(self) -> str:
        return str(self)


class _UsageFixture(unittest.TestCase):
    def setUp(self) -> None:
        cli_dir = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))
        if cli_dir is None:
            self.skipTest("CLI sources are unavailable")
        self.cli_dir = Path(cli_dir)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
        sys.path.insert(0, str(self.cli_dir))
        self.metrics = {"vcn_activity": [0, "N/A", 100], "jpeg_activity": "N/A"}
        self.activity = {"gfx_activity": 0, "umc_activity": 100, "mm_activity": "N/A"}
        self.real_interface = sys.modules["amdsmi"].amdsmi_interface
        self.interface = fake_module(
            "amdsmi.amdsmi_interface",
            amdsmi_wrapper=sys.modules["amdsmi"].amdsmi_interface.amdsmi_wrapper,
            AMDSMI_MAX_RAIL_INDEX=7,
            amdsmi_get_gpu_metrics_info=mock.Mock(side_effect=lambda _gpu: self.metrics),
            amdsmi_get_gpu_metrics_header_info=mock.Mock(return_value={}),
            _NA_amdsmi_get_gpu_metrics_info=mock.Mock(return_value={}),
            amdsmi_get_gpu_activity=mock.Mock(side_effect=lambda _gpu: self.activity),
            amdsmi_get_gpu_partition_metrics_info=mock.Mock(return_value=None),
            amdsmi_get_vcn_busy_percent=mock.Mock(return_value=23),
            amdsmi_get_power_info=mock.Mock(side_effect=_LibraryError("unsupported power")),
            amdsmi_is_gpu_power_management_enabled=mock.Mock(
                side_effect=_LibraryError("unsupported power management")
            ),
        )
        for name in (
            "amdsmi_get_pcie_info",
            "amdsmi_get_gpu_memory_usage",
            "amdsmi_get_gpu_fan_speed",
            "amdsmi_get_temp_metric",
        ):
            setattr(self.interface, name, mock.Mock(side_effect=AssertionError(name)))
        exception = fake_module(
            "amdsmi.amdsmi_exception",
            AmdSmiException=_AmdSmiError,
            AmdSmiLibraryException=_LibraryError,
        )
        package = fake_module("amdsmi", amdsmi_interface=self.interface, amdsmi_exception=exception)
        # Restore only replaced modules; unloading Unicode data breaks later source parsing.
        self.stack.callback(
            stub_modules_at_import(
                {
                    "amdsmi": package,
                    "amdsmi.amdsmi_interface": self.interface,
                    "amdsmi.amdsmi_exception": exception,
                    "amdsmi_init": fake_module(
                        "amdsmi_init", amdsmi_interface=self.interface, amdsmi_exception=exception
                    ),
                    "amdsmi_helpers": None,
                    "_version": fake_module("_version", __version__="0.0.0+test"),
                }
            )
        )
        self.helper_module = load_cli_module(
            "usage_helpers_test", str(self.cli_dir / "amdsmi_helpers.py")
        )
        sys.modules["amdsmi_helpers"] = self.helper_module
        self.parser_module = load_cli_module(
            "usage_parser_test", str(self.cli_dir / "amdsmi_parser.py")
        )
        self.logger_module = load_cli_module(
            "usage_logger_test", str(self.cli_dir / "amdsmi_logger.py")
        )
        self.metric_module = load_cli_module(
            "usage_metric_test", str(self.cli_dir / "subcommands/metric.py")
        )
        self.invalid_parameter = sys.modules[
            "amdsmi_cli_exceptions"
        ].AmdSmiInvalidParameterException
        self.missing_parameter = sys.modules[
            "amdsmi_cli_exceptions"
        ].AmdSmiMissingParameterValueException
        self.helpers = object.__new__(self.helper_module.AMDSMIHelpers)
        flags = {
            "is_amdgpu_initialized": True,
            "is_amd_hsmp_initialized": False,
            "is_baremetal": True,
            "is_linux": True,
            "is_windows": False,
            "is_hypervisor": False,
            "is_brcm_nic_initialized": False,
            "is_brcm_switch_initialized": False,
            "is_ainic_initialized": False,
        }
        for name, value in flags.items():
            setattr(self.helpers, name, mock.Mock(return_value=value))
        self.helpers.get_output_format = mock.Mock(return_value="human_readable")
        self.helpers.get_gpu_id_from_device_handle = mock.Mock(
            side_effect=lambda handle: handle - 10
        )
        self.helpers.check_required_groups = mock.Mock()
        self.helpers.os_info = mock.Mock(return_value="test platform")
        self.helpers._get_metric_version_and_partition_info = mock.Mock(
            return_value={"num_partition": "N/A"}
        )
        self.stack.enter_context(
            mock.patch.object(self.parser_module, "AMDSMIHelpers", return_value=self.helpers)
        )
        self.stack.enter_context(
            mock.patch.object(self.helper_module.time, "sleep", return_value=None)
        )

    @contextlib.contextmanager
    def subtest_directory(self, **params: object):
        """Subtest with a temporary directory; kept separate so Python 3.8 can parse it."""
        with self.subTest(**params), tempfile.TemporaryDirectory() as directory:
            yield directory

    def parse(self, *options: str) -> argparse.Namespace:
        parser = object.__new__(self.parser_module.AMDSMIParser)
        argparse.ArgumentParser.__init__(parser, prog="amd-smi", description="Usage tests")
        parser.helpers = self.helpers
        for device in ("gpu", "cpu", "core", "nic", "switch"):
            setattr(parser, device + "_choices", {})
            setattr(parser, device + "_choices_str", "")
        subparsers = parser.add_subparsers(parser_class=self.parser_module.AMDSMISubParser)
        parser._add_metric_parser(subparsers, func=lambda _args: None)
        self.last_parser = parser
        with mock.patch.object(sys, "argv", ["amd-smi", "metric", *options]):
            return parser.parse_args(["metric", *options])

    def commands(self, format: str = "json", destination: object = "stdout") -> object:
        commands = object.__new__(self.metric_module.MetricCommands)
        commands.helpers = self.helpers
        commands.logger = self.logger_module.AMDSMILogger(
            format=format, destination=destination, helpers=self.helpers
        )
        commands.device_handles = [10]
        commands.cpu_handles = [20]
        commands.core_handles = [30]
        commands.group_check_printed = True
        return commands

    def run_gpu(self, args: argparse.Namespace, format: str = "json") -> dict:
        commands = self.commands(format=format)
        with contextlib.redirect_stdout(io.StringIO()):
            commands.metric_gpu(args)
        return commands.logger.store_gpu_json_output[-1]

    def run_metric(self, commands: object, args: argparse.Namespace) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            commands.metric(args)
        return output.getvalue()


class TestUsageParser(_UsageFixture):
    def test_selector_implies_usage(self) -> None:
        try:
            args = self.parse("--usage-fields", "gfx_activity")
        except self.invalid_parameter as error:
            self.fail(f"Selector rejected: {error}")
        self.assertTrue(args.usage)
        self.assertEqual(args.usage_fields, ("gfx_activity",))

    def test_selector_keeps_order_and_trims_names(self) -> None:
        args = self.parse("--usage", "--usage-fields", " umc_activity , gfx_activity ")
        self.assertEqual(args.usage_fields, ("umc_activity", "gfx_activity"))
        self.assertTrue(args.usage)

    def test_selector_names_are_case_insensitive(self) -> None:
        # The CLI entry point lowercases argv, so the parser must agree.
        args = self.parse("--usage-fields", "GFX_ACTIVITY,Umc_Activity")
        self.assertEqual(args.usage_fields, ("gfx_activity", "umc_activity"))

    def test_invalid_fields_raise_argument_errors(self) -> None:
        for value, message in (
            ("", "empty"),
            (" ", "empty"),
            ("gfx_activity,", "empty"),
            (",gfx_activity", "empty"),
            ("gfx_activity,,umc_activity", "empty"),
            ("unknown", "unknown"),
            ("gfx_activity,gfx_activity", "duplicate"),
            ("gfx_activity, gfx_activity ", "duplicate"),
            ("gfx_activity,GFX_ACTIVITY", "duplicate"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(self.invalid_parameter) as caught:
                    self.parse("--usage-fields", value)
                self.assertIn(message, str(caught.exception).lower())

    def test_unknown_field_error_points_to_help(self) -> None:
        with self.assertRaises(self.invalid_parameter) as caught:
            self.parse("--usage-fields", "unknown")
        self.assertIn("--help", str(caught.exception))

    def test_repeated_selector_is_rejected(self) -> None:
        with self.assertRaises(self.invalid_parameter) as caught:
            self.parse("--usage-fields", "gfx_activity", "--usage-fields", "umc_activity")
        self.assertIn("once", str(caught.exception).lower())

    def test_missing_selector_value_raises(self) -> None:
        with self.assertRaises(self.missing_parameter):
            self.parse("--usage-fields")

    def test_all_accepted_fields_parse_and_cover_filtered_fields(self) -> None:
        accepted = self.parser_module._USAGE_FIELDS
        args = self.parse("--usage-fields", ",".join(accepted))
        self.assertEqual(args.usage_fields, accepted)
        field_support = load_cli_module(
            "usage_field_support_test", str(self.cli_dir / "amdsmi_metrics_field_support.py")
        )
        filtered = {
            key for section, key in field_support._CLI_PATH_REQUIRED_SOURCES if section == "usage"
        }
        self.assertLessEqual(filtered, set(accepted))

    def test_documented_fields_match_accepted_fields(self) -> None:
        docs = self.cli_dir.parent / "docs" / "how-to" / "amdsmi-cli-tool.md"
        if not docs.is_file():
            self.skipTest("CLI docs exist only in the source tree")
        section = docs.read_text(encoding="utf-8").split("#### Selecting usage fields", 1)[1]
        bullets = section.split("Supported fields:", 1)[1].split("\n\n", 2)[1]
        documented = tuple(re.findall(r"`([a-z0-9_]+)`", bullets))
        self.assertEqual(documented, self.parser_module._USAGE_FIELDS)

    def test_legacy_usage_remains_boolean(self) -> None:
        args = self.parse("--usage")
        self.assertIs(args.usage, True)
        self.assertIsNone(args.usage_fields)

    def test_selector_accepts_existing_modifiers(self) -> None:
        for output_format in ("--json", "--csv"):
            with self.subtest_directory(output_format=output_format) as directory:
                path = Path(directory) / "usage-output"
                args = self.parse(
                    "--usage-fields",
                    "gfx_activity",
                    "--power",
                    "--partition",
                    output_format,
                    "--watch",
                    "1",
                    "--watch_time",
                    "3",
                    "--iterations",
                    "2",
                    "--file",
                    str(path),
                )
                self.assertTrue(args.power)
                self.assertTrue(args.partition)
                self.assertEqual(args.watch, 1)
                self.assertEqual(args.iterations, 2)
                self.assertEqual(args.file, path)
                self.assertTrue(path.is_file())

    def test_help_explains_selector_and_formats(self) -> None:
        self.parse("--usage")
        action = next(
            action
            for action in self.last_parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        help_text = action.choices["metric"].format_help()
        for phrase in (
            "--usage-fields",
            "--json",
            "--csv",
            "N/A",
            "0 is idle",
            "instantaneous busy readings",
            "ignore case",
            *self.parser_module._USAGE_FIELDS,
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, help_text)
        lines = help_text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("  --usage-fields"))
        end = next(i for i, line in enumerate(lines) if line.startswith("  -p, --power"))
        for line in lines[start:end]:
            self.assertLessEqual(len(line), 90, line)
        # A longer invocation would push every metric option's help to a new column.
        usage_line = next(line for line in lines if line.startswith("  -u, --usage "))
        self.assertEqual(usage_line.index("Displays engine usage"), 31)


class TestUsageSelection(_UsageFixture):
    def test_only_requested_field_is_returned(self) -> None:
        result = self.run_gpu(self.parse("--usage-fields", "gfx_activity"))
        self.assertEqual(result, {"gpu": 0, "usage": {"gfx_activity": {"value": 0, "unit": "%"}}})

    def test_order_and_unavailable_selected_fields(self) -> None:
        result = self.run_gpu(
            self.parse("--usage-fields", "mm_activity,umc_activity,apu_average_ipu_reads")
        )
        self.assertEqual(
            list(result["usage"]), ["mm_activity", "umc_activity", "apu_average_ipu_reads"]
        )
        self.assertEqual(result["usage"]["mm_activity"], "N/A")
        self.assertEqual(result["usage"]["umc_activity"], {"value": 100, "unit": "%"})
        self.assertEqual(result["usage"]["apu_average_ipu_reads"], "N/A")

    def test_selector_does_not_query_unrelated_sensors(self) -> None:
        self.run_gpu(self.parse("--usage-fields", "gfx_activity"))
        for name in (
            "amdsmi_get_power_info",
            "amdsmi_get_pcie_info",
            "amdsmi_get_gpu_memory_usage",
            "amdsmi_get_gpu_fan_speed",
            "amdsmi_get_temp_metric",
        ):
            getattr(self.interface, name).assert_not_called()

    def test_explicit_power_is_preserved(self) -> None:
        result = self.run_gpu(self.parse("--usage-fields", "gfx_activity", "--power"))
        self.assertEqual(set(result), {"gpu", "usage", "power"})
        self.assertEqual(list(result["usage"]), ["gfx_activity"])
        self.interface.amdsmi_get_power_info.assert_called_once()

    def test_selector_routes_only_to_gpus(self) -> None:
        self.helpers.is_amd_hsmp_initialized.return_value = True
        self.helpers.is_brcm_nic_initialized.return_value = True
        args = self.parse("--usage-fields", "gfx_activity")
        commands = self.commands()
        commands.metric_cpu = mock.Mock(side_effect=AssertionError("CPU query"))
        commands.metric_core = mock.Mock(side_effect=AssertionError("core query"))
        commands.metric_nic = mock.Mock(side_effect=AssertionError("NIC query"))
        result = json.loads(self.run_metric(commands, args))
        self.assertEqual(list(result), ["gpu_data"])
        self.assertEqual(list(result["gpu_data"][0]["usage"]), ["gfx_activity"])
        commands.metric_cpu.assert_not_called()
        commands.metric_core.assert_not_called()
        commands.metric_nic.assert_not_called()

    def test_selected_fields_survive_unsupported_filter(self) -> None:
        # v1.3 carries no JPEG activity, so the human-output filter targets it.
        self.interface.amdsmi_get_gpu_metrics_header_info.return_value = {
            "format_revision": 1,
            "content_revision": 3,
        }
        for selection in ("jpeg_activity", "gfx_activity,jpeg_activity"):
            with self.subTest(selection=selection):
                result = self.run_gpu(
                    self.parse("--usage-fields", selection), format="human_readable"
                )
                self.assertEqual(list(result["usage"]), selection.split(","))
                self.assertEqual(result["usage"]["jpeg_activity"], "N/A")

    def test_unselected_usage_still_filters_unsupported(self) -> None:
        self.interface.amdsmi_get_gpu_metrics_header_info.return_value = {
            "format_revision": 1,
            "content_revision": 3,
        }
        result = self.run_gpu(self.parse("--usage"), format="human_readable")
        self.assertNotIn("jpeg_activity", result["usage"])
        self.assertEqual(result["usage"]["gfx_activity"], "0 %")


class TestUsageSources(_UsageFixture):
    def test_average_failure_keeps_socket_activity(self) -> None:
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        result = self.run_gpu(self.parse("--usage"))
        self.assertIsInstance(result["usage"], dict)
        self.assertEqual(result["usage"]["gfx_activity"], "N/A")
        self.assertEqual(
            result["usage"]["vcn_activity"],
            [{"value": 0, "unit": "%"}, "N/A", {"value": 100, "unit": "%"}],
        )

    def test_partition_survives_socket_and_average_failures(self) -> None:
        self.interface.amdsmi_get_gpu_metrics_info.side_effect = _LibraryError("socket unavailable")
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        self.interface.amdsmi_get_gpu_partition_metrics_info.return_value = {
            "xcp_stats.gfx_busy_inst": [[0, 100, 7]],
            "xcp_stats.jpeg_busy": [[0, "N/A", 20]],
            "xcp_stats.vcn_busy": [[0, 13]],
        }
        result = self.run_gpu(self.parse("--usage", "--partition"))
        self.assertIsInstance(result["usage"], dict)
        self.assertEqual(
            result["usage"]["gfx_busy_inst"]["xcp_0"],
            [{"value": 0, "unit": "%"}, {"value": 100, "unit": "%"}, {"value": 7, "unit": "%"}],
        )
        self.assertEqual(result["usage"]["jpeg_busy"]["xcp_0"][1], "N/A")
        self.interface.amdsmi_get_gpu_metrics_info.assert_called_once()
        self.interface.amdsmi_get_gpu_activity.assert_called_once()
        self.interface.amdsmi_get_gpu_partition_metrics_info.assert_called_once()
        self.interface.amdsmi_get_vcn_busy_percent.assert_not_called()

    def test_sysfs_survives_failed_metrics(self) -> None:
        self.interface.amdsmi_get_gpu_metrics_info.side_effect = _LibraryError("socket unavailable")
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        self.interface.amdsmi_get_vcn_busy_percent.return_value = 0
        result = self.run_gpu(self.parse("--usage-fields", "vcn_busy,gfx_activity"))
        self.assertEqual(
            result["usage"], {"vcn_busy": {"value": 0, "unit": "%"}, "gfx_activity": "N/A"}
        )
        self.interface.amdsmi_get_vcn_busy_percent.assert_called_once()
        self.interface.amdsmi_get_gpu_partition_metrics_info.assert_not_called()

    def test_partition_failure_uses_existing_socket_path(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.side_effect = _LibraryError(
            "partition unavailable"
        )
        self.helpers._get_metric_version_and_partition_info.return_value = {"num_partition": 1}
        self.metrics.update(
            {
                "xcp_stats.gfx_busy_inst": [[7, 0]],
                "xcp_stats.jpeg_busy": [[9]],
                "xcp_stats.vcn_busy": [[13]],
            }
        )
        result = self.run_gpu(self.parse("--usage-fields", "vcn_busy", "--partition"))
        self.assertEqual(result["usage"]["vcn_busy"], {"xcp_0": [{"value": 13, "unit": "%"}]})
        self.interface.amdsmi_get_gpu_partition_metrics_info.assert_called_once()
        self.interface.amdsmi_get_vcn_busy_percent.assert_not_called()

    def test_socket_xcp_values_are_limited_to_partition_count(self) -> None:
        self.helpers._get_metric_version_and_partition_info.return_value = {"num_partition": 1}
        self.metrics.update(
            {
                "xcp_stats.gfx_busy_inst": [[7, 0], [9, 0]],
                "xcp_stats.jpeg_busy": [[1], [2]],
                "xcp_stats.vcn_busy": [[3], [4]],
            }
        )
        result = self.run_gpu(self.parse("--usage"))
        for field in ("gfx_busy_inst", "jpeg_busy", "vcn_busy"):
            with self.subTest(field=field):
                self.assertEqual(list(result["usage"][field]), ["xcp_0"])

    def test_produced_fields_match_accepted_fields(self) -> None:
        apu_metrics = {
            "apu_metrics." + field[len("apu_") :]: 0
            for field in self.parser_module._USAGE_FIELDS
            if field.startswith("apu_")
        }
        self.metrics.update({"is_apu": True, **apu_metrics})
        result = self.run_gpu(self.parse("--usage"))
        self.assertEqual(sorted(result["usage"]), sorted(self.parser_module._USAGE_FIELDS))

    def test_partition_values_take_precedence(self) -> None:
        self.helpers._get_metric_version_and_partition_info.return_value = {"num_partition": 1}
        self.metrics["xcp_stats.vcn_busy"] = [[99]]
        self.interface.amdsmi_get_gpu_partition_metrics_info.return_value = {
            "xcp_stats.vcn_busy": [[0]]
        }
        result = self.run_gpu(self.parse("--usage-fields", "vcn_busy", "--partition"))
        self.assertEqual(result["usage"]["vcn_busy"], {"xcp_0": [{"value": 0, "unit": "%"}]})
        self.interface.amdsmi_get_vcn_busy_percent.assert_not_called()

    def test_apu_data_survives_average_failure(self) -> None:
        self.metrics.update({"is_apu": True, "apu_metrics.average_ipu_reads": 0})
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        result = self.run_gpu(
            self.parse("--usage-fields", "apu_average_ipu_reads,apu_average_mm_activity")
        )
        self.assertEqual(
            result["usage"],
            {
                "apu_average_ipu_reads": {"value": 0, "unit": "MB/s"},
                "apu_average_mm_activity": "N/A",
            },
        )

    def test_all_unavailable_retains_selected_keys(self) -> None:
        self.interface.amdsmi_get_gpu_metrics_info.side_effect = _LibraryError("socket unavailable")
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        self.interface.amdsmi_get_vcn_busy_percent.side_effect = _LibraryError("sysfs unavailable")
        result = self.run_gpu(
            self.parse("--usage-fields", "gfx_activity,vcn_busy,apu_average_dram_reads")
        )
        self.assertEqual(
            result["usage"],
            {"gfx_activity": "N/A", "vcn_busy": "N/A", "apu_average_dram_reads": "N/A"},
        )

    def test_all_unavailable_without_selection_keeps_every_field(self) -> None:
        self.interface.amdsmi_get_gpu_metrics_info.side_effect = _LibraryError("socket unavailable")
        self.interface.amdsmi_get_gpu_activity.side_effect = _LibraryError("average unavailable")
        self.interface.amdsmi_get_vcn_busy_percent.side_effect = _LibraryError("sysfs unavailable")
        fields = (
            "gfx_activity",
            "umc_activity",
            "mm_activity",
            "vcn_activity",
            "jpeg_activity",
            "gfx_busy_inst",
            "jpeg_busy",
            "vcn_busy",
        )
        for output_format in ("json", "csv", "human_readable"):
            with self.subTest(format=output_format):
                result = self.run_gpu(self.parse("--usage"), format=output_format)
                self.assertEqual(list(result["usage"].items()), [(f, "N/A") for f in fields])

    def test_unexpected_exception_is_not_hidden(self) -> None:
        for name in ("amdsmi_get_gpu_activity", "amdsmi_get_vcn_busy_percent"):
            with self.subTest(source=name):
                with mock.patch.object(
                    self.interface, name, side_effect=ValueError("bad test payload")
                ):
                    with self.assertRaisesRegex(ValueError, "bad test payload"):
                        self.run_gpu(self.parse("--usage"))

    def test_scalar_unavailable_xcp_does_not_hide_siblings(self) -> None:
        self.helpers._get_metric_version_and_partition_info.return_value = {"num_partition": 1}
        self.metrics.update(
            {
                "xcp_stats.gfx_busy_inst": "N/A",
                "xcp_stats.jpeg_busy": [[0]],
                "xcp_stats.vcn_busy": [[12]],
            }
        )
        result = self.run_gpu(self.parse("--usage"))
        self.assertEqual(result["usage"]["gfx_busy_inst"], "N/A")
        self.assertEqual(result["usage"]["vcn_busy"], {"xcp_0": [{"value": 12, "unit": "%"}]})

    def test_sysfs_failure_does_not_hide_averages(self) -> None:
        self.interface.amdsmi_get_vcn_busy_percent.side_effect = _LibraryError("sysfs unavailable")
        result = self.run_gpu(self.parse("--usage"))
        self.assertEqual(result["usage"]["gfx_activity"], {"value": 0, "unit": "%"})
        self.assertEqual(result["usage"]["vcn_busy"], "N/A")


class TestUsageFormatting(_UsageFixture):
    def test_repeated_formats_do_not_change_input_payloads(self) -> None:
        self.helpers._get_metric_version_and_partition_info.return_value = {"num_partition": 1}
        self.metrics.update(
            {
                "xcp_stats.gfx_busy_inst": [[0, 17]],
                "xcp_stats.jpeg_busy": [["N/A", 100]],
                "xcp_stats.vcn_busy": [[11]],
            }
        )
        before_metrics = copy.deepcopy(self.metrics)
        before_activity = copy.deepcopy(self.activity)
        for output_format in ("human_readable", "json", "csv", "json"):
            with self.subTest(output_format=output_format):
                self.run_gpu(self.parse("--usage"), format=output_format)
                self.assertEqual(self.metrics, before_metrics)
                self.assertEqual(self.activity, before_activity)

    def test_successful_legacy_usage_shape(self) -> None:
        result = self.run_gpu(self.parse("--usage"))["usage"]
        self.assertEqual(
            list(result),
            [
                "gfx_activity",
                "umc_activity",
                "mm_activity",
                "vcn_activity",
                "jpeg_activity",
                "gfx_busy_inst",
                "jpeg_busy",
                "vcn_busy",
            ],
        )
        self.assertEqual(
            result,
            {
                "gfx_activity": {"value": 0, "unit": "%"},
                "umc_activity": {"value": 100, "unit": "%"},
                "mm_activity": "N/A",
                "vcn_activity": [{"value": 0, "unit": "%"}, "N/A", {"value": 100, "unit": "%"}],
                "jpeg_activity": "N/A",
                "gfx_busy_inst": "N/A",
                "jpeg_busy": "N/A",
                "vcn_busy": {"value": 23, "unit": "%"},
            },
        )

    def test_percent_arrays_keep_each_format(self) -> None:
        expected = {
            "human_readable": "[0 %, N/A, 100 %]",
            "json": [{"value": 0, "unit": "%"}, "N/A", {"value": 100, "unit": "%"}],
            "csv": [0, "N/A", 100],
        }
        for output_format, value in expected.items():
            with self.subTest(output_format=output_format):
                result = self.run_gpu(
                    self.parse("--usage-fields", "vcn_activity"), format=output_format
                )
                self.assertEqual(result["usage"]["vcn_activity"], value)

    def test_apu_units_and_legacy_field_order(self) -> None:
        self.metrics.update(
            {
                "is_apu": True,
                "apu_metrics.average_gfx_activity": 5,
                "apu_metrics.average_core_c0_activity": [0, 100],
                "apu_metrics.average_ipu_reads": 0,
                "apu_metrics.average_ipu_writes": 9,
            }
        )
        result = self.run_gpu(self.parse("--usage"))["usage"]
        self.assertEqual(
            list(result)[-4:],
            [
                "apu_average_gfx_activity",
                "apu_average_core_c0_activity",
                "apu_average_ipu_reads",
                "apu_average_ipu_writes",
            ],
        )
        self.assertEqual(result["apu_average_ipu_reads"], {"value": 0, "unit": "MB/s"})
        self.assertEqual(result["apu_average_gfx_activity"], {"value": 5, "unit": "%"})
        self.assertEqual(
            result["apu_average_core_c0_activity"],
            [{"value": 0, "unit": "%"}, {"value": 100, "unit": "%"}],
        )
        self.assertNotIn("apu_average_mm_activity", result)

    def test_selected_empty_xcp_value_is_unavailable(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.return_value = {
            "xcp_stats.gfx_busy_inst": []
        }
        for output_format in ("human_readable", "json", "csv"):
            with self.subTest(output_format=output_format):
                result = self.run_gpu(
                    self.parse("--usage-fields", "gfx_busy_inst", "--partition"),
                    format=output_format,
                )
                self.assertEqual(result["usage"], {"gfx_busy_inst": "N/A"})

    def test_selected_empty_activity_arrays_are_unavailable(self) -> None:
        self.metrics.update(
            {
                "is_apu": True,
                "vcn_activity": [],
                "apu_metrics.average_core_c0_activity": [],
                "apu_metrics.average_ipu_reads": [],
            }
        )
        fields = ("vcn_activity", "apu_average_core_c0_activity", "apu_average_ipu_reads")
        for output_format in ("human_readable", "json", "csv"):
            with self.subTest(output_format=output_format):
                result = self.run_gpu(
                    self.parse("--usage-fields", ",".join(fields)), format=output_format
                )
                self.assertEqual(result["usage"], dict.fromkeys(fields, "N/A"))


class TestUsageCsv(_UsageFixture):
    def test_single_nested_field_keeps_its_name(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.return_value = {
            "xcp_stats.gfx_busy_inst": [[0, 17]]
        }
        commands = self.commands(format="csv")
        text = self.run_metric(
            commands, self.parse("--usage-fields", "gfx_busy_inst", "--partition", "--csv")
        )
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        self.assertEqual(reader.fieldnames, ["gpu", "gfx_busy_inst_xcp_0"])
        self.assertEqual(rows, [{"gpu": "0", "gfx_busy_inst_xcp_0": "[0, 17]"}])

    def test_mixed_devices_keep_column_values_in_requested_order(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.side_effect = lambda gpu: (
            {"xcp_stats.gfx_busy_inst": [[7, 0]], "xcp_stats.jpeg_busy": [[9]]}
            if gpu == 11
            else {"xcp_stats.gfx_busy_inst": "N/A", "xcp_stats.jpeg_busy": "N/A"}
        )
        for devices in ([10, 11], [11, 10]):
            with self.subTest(devices=devices):
                commands = self.commands(format="csv")
                commands.device_handles = devices
                args = self.parse(
                    "--usage-fields", "gfx_busy_inst,gfx_activity,jpeg_busy", "--partition", "--csv"
                )
                reader = csv.DictReader(io.StringIO(self.run_metric(commands, args)))
                rows = list(reader)
                fields = reader.fieldnames
                self.assertNotIn("usage", fields)
                for field in ("gfx_busy_inst", "gfx_busy_inst_xcp_0"):
                    self.assertLess(fields.index(field), fields.index("gfx_activity"))
                for field in ("jpeg_busy", "jpeg_busy_xcp_0"):
                    self.assertLess(fields.index("gfx_activity"), fields.index(field))
                by_gpu = {row["gpu"]: row for row in rows}
                self.assertEqual(by_gpu["0"]["gfx_busy_inst_xcp_0"], "N/A")
                self.assertEqual(by_gpu["0"]["jpeg_busy_xcp_0"], "N/A")
                self.assertEqual(by_gpu["1"]["gfx_busy_inst_xcp_0"], "[7, 0]")
                self.assertEqual(by_gpu["1"]["gfx_activity"], "0")
                self.assertEqual(by_gpu["1"]["jpeg_busy_xcp_0"], "[9]")
                self.assertEqual(by_gpu["1"]["gfx_busy_inst"], "N/A")

    def test_selected_csv_does_not_change_rows_or_generic_flattening(self) -> None:
        logger = self.commands(format="csv").logger
        self.assertIsNone(logger.usage_fields)
        payload = {"usage": {"gfx_busy_inst": {"xcp_0": [0, 17]}}}
        before = copy.deepcopy(payload)
        generic = logger.flatten_dict(copy.deepcopy(payload))
        logger.usage_fields = ("gfx_busy_inst",)
        logger._store_output_amdsmi(gpu_id=0, argument="values", data=payload)
        self.assertEqual(logger.output, {"gpu": 0, "gfx_busy_inst_xcp_0": [0, 17]})
        logger.multiple_device_output = [
            {"gpu": 0, "gfx_busy_inst": "N/A"},
            {"gpu": 1, "gfx_busy_inst_xcp_0": [0, 17]},
        ]
        rows_before = copy.deepcopy(logger.multiple_device_output)
        with contextlib.redirect_stdout(io.StringIO()):
            logger.print_output(multiple_device_enabled=True)
        self.assertEqual(logger.multiple_device_output, rows_before)
        self.assertEqual(payload, before)
        # Selection must not change the shared helper used by every other command.
        self.assertEqual(logger.flatten_dict(copy.deepcopy(payload)), generic)


class TestUsageOutput(_UsageFixture):
    def test_mixed_watch_json_file_keeps_cpu_core_records(self) -> None:
        self.helpers.is_amd_hsmp_initialized.return_value = True
        self.helpers.get_cpu_id_from_device_handle = mock.Mock(
            side_effect=lambda handle: handle - 20
        )
        self.helpers.get_core_id_from_device_handle = mock.Mock(
            side_effect=lambda handle: handle - 30
        )
        self.interface.amdsmi_get_cpu_prochot_status = mock.Mock(
            side_effect=lambda handle: handle - 20
        )
        self.interface.amdsmi_get_cpu_core_boostlimit = mock.Mock(
            side_effect=lambda handle: 4000 + handle - 30
        )
        for selection in (("--usage",), ("--usage-fields", "gfx_activity")):
            for cpu, core in ((True, False), (False, True), (True, True)):
                with self.subtest_directory(selection=selection, cpu=cpu, core=core) as directory:
                    path = Path(directory) / "mixed-watch.json"
                    commands = self.commands(format="json", destination=path)
                    commands.device_handles = [10, 11]
                    commands.cpu_handles = [20, 21]
                    commands.core_handles = [30, 31]
                    options = ["--json"]
                    if cpu:
                        options.append("--cpu-prochot")
                    if core:
                        options.append("--core-boost-limit")
                    args = self.parse(
                        *selection,
                        "--watch",
                        "1",
                        "--iterations",
                        "2",
                        "--file",
                        str(path),
                        *options,
                    )
                    self.interface.amdsmi_get_cpu_prochot_status.reset_mock()
                    self.interface.amdsmi_get_cpu_core_boostlimit.reset_mock()
                    self.run_metric(commands, args)
                    self.assertEqual(
                        self.interface.amdsmi_get_cpu_prochot_status.call_args_list,
                        [mock.call(20), mock.call(21)] if cpu else [],
                    )
                    self.assertEqual(
                        self.interface.amdsmi_get_cpu_core_boostlimit.call_args_list,
                        [mock.call(30), mock.call(31)] if core else [],
                    )
                    output = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(
                        set(output),
                        {"gpu_data"}
                        | ({"cpu_data"} if cpu else set())
                        | ({"core_data"} if core else set()),
                    )
                    if cpu:
                        self.assertEqual(
                            output["cpu_data"],
                            [{"cpu": i, "prochot": {"prochot_status": i}} for i in (0, 1)],
                        )
                    if core:
                        self.assertEqual(
                            output["core_data"],
                            [{"core": i, "boost_limit": {"value": 4000 + i}} for i in (0, 1)],
                        )
                    self.assertEqual([row["gpu"] for row in output["gpu_data"]], [0, 1, 0, 1])

    def test_nic_and_switch_watch_files_include_every_sample(self) -> None:
        devices = (
            ("metric_nic", "nic", "brcm_nic", "amdsmi_get_nic_metrics_info", ("nic_temperature",)),
            ("metric_switch", "switch", "brcm_switch", "amdsmi_get_switch_metrics_info", ()),
        )
        for method, device, label, getter, extra_flags in devices:
            setattr(self.interface, getter, mock.Mock(return_value={}))
            setattr(self.helpers, f"get_{device}_id_from_device_handle", mock.Mock(return_value=0))
            for output_format, stop in itertools.product(
                ("json", "csv", "human_readable"), ("iterations", "interrupt")
            ):
                with self.subtest_directory(device=device, format=output_format, stop=stop) as d:
                    path = Path(d) / "watch-output"
                    commands = self.commands(format=output_format, destination=path)
                    flags = {
                        name: False
                        for name in (f"{device}_power", f"{device}_errors", *extra_flags)
                    }
                    flags[f"{device}_power"] = True
                    args = argparse.Namespace(
                        **{device: 50},
                        **flags,
                        watch=1,
                        watch_time=None,
                        iterations=2 if stop == "iterations" else None,
                    )
                    sleep = [None, KeyboardInterrupt()] if stop == "interrupt" else None
                    with mock.patch.object(self.helper_module.time, "sleep", side_effect=sleep):
                        with contextlib.redirect_stdout(io.StringIO()):
                            if stop == "interrupt":
                                with self.assertRaises(KeyboardInterrupt):
                                    getattr(commands, method)(args)
                            else:
                                getattr(commands, method)(args)
                    text = path.read_text(encoding="utf-8")
                    if output_format == "json":
                        self.assertEqual([row[label] for row in json.loads(text)], [0, 0])
                    elif output_format == "csv":
                        rows = csv.DictReader(io.StringIO(text))
                        self.assertEqual([row[label] for row in rows], ["0", "0"])
                    else:
                        self.assertEqual(text.count("TIMESTAMP:"), 2)

    def test_cancelled_watch_file_flushes_completed_samples(self) -> None:
        init_path = self.cli_dir / "amdsmi_init.py"
        source = ast.parse(init_path.read_text(encoding="utf-8"))
        handler = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "signal_handler"
        )
        namespace = {"sys": sys, "logging": logging}
        # Execute only the real handler, without device initialization or signal registration.
        exec(
            compile(ast.Module(body=[handler], type_ignores=[]), str(init_path), "exec"), namespace
        )
        for output_format in ("json", "csv", "human_readable"):
            for cancellation in (signal.SIGINT, signal.SIGTERM, KeyboardInterrupt()):
                for devices, selection in itertools.product(
                    ([10], [10, 11]), (("--usage",), ("--usage-fields", "gfx_activity"))
                ):
                    with self.subtest_directory(
                        format=output_format,
                        cancellation=cancellation,
                        devices=devices,
                        selection=selection,
                    ) as directory:
                        path = Path(directory) / "watch-output"
                        commands = self.commands(format=output_format, destination=path)
                        commands.device_handles = devices
                        options = (
                            [] if output_format == "human_readable" else ["--" + output_format]
                        )
                        args = self.parse(*selection, "--watch", "1", "--file", str(path), *options)
                        if isinstance(cancellation, KeyboardInterrupt):
                            effect = cancellation
                            exception_type = KeyboardInterrupt
                        else:

                            def send_signal(_interval, signum=cancellation):
                                namespace["signal_handler"](signum, None)

                            effect = send_signal
                            exception_type = SystemExit
                        with mock.patch.object(
                            self.helper_module.time, "sleep", side_effect=effect
                        ):
                            with self.assertRaises(exception_type) as caught:
                                self.run_metric(commands, args)
                        if exception_type is SystemExit:
                            self.assertEqual(caught.exception.code, 0)
                        else:
                            self.assertIs(caught.exception, cancellation)
                        text = path.read_text(encoding="utf-8")
                        if output_format == "json":
                            rows = json.loads(text)
                            self.assertEqual(
                                [row["gpu"] for row in rows], [device - 10 for device in devices]
                            )
                            for row in rows:
                                self.assertIsInstance(row["timestamp"], int)
                                self.assertEqual(
                                    row["usage"]["gfx_activity"], {"value": 0, "unit": "%"}
                                )
                        elif output_format == "csv":
                            rows = list(csv.DictReader(io.StringIO(text)))
                            self.assertEqual(
                                [row["gpu"] for row in rows],
                                [str(device - 10) for device in devices],
                            )
                            for row in rows:
                                self.assertTrue(row["timestamp"].isdigit())
                                self.assertEqual(row["gfx_activity"], "0")
                        else:
                            self.assertEqual(text.count("TIMESTAMP:"), len(devices))
                            self.assertEqual(text.count("GFX_ACTIVITY: 0 %"), len(devices))

    def test_all_sections_apu_retains_selected_unavailable_fields(self) -> None:
        for name, value in vars(self.real_interface).items():
            if name.startswith(("AmdSmi", "AMDSMI_")):
                setattr(self.interface, name, value)
            elif name.startswith("amdsmi_") and name not in (
                "amdsmi_get_gpu_metrics_info",
                "amdsmi_get_gpu_activity",
            ):
                setattr(self.interface, name, mock.Mock(side_effect=_LibraryError(name)))
        self.interface.AmdSmiLibraryException = _LibraryError
        self.metrics.update({"is_apu": True, "apu_metrics.average_ipu_reads": 0})
        self.activity["gfx_activity"] = "N/A"
        sections = (
            "--mem-usage",
            "--power",
            "--clock",
            "--temperature",
            "--voltage",
            "--pcie",
            "--ecc",
            "--ecc-blocks",
            "--base-board",
            "--gpu-board",
            "--fan",
            "--voltage-curve",
            "--overdrive",
            "--perf-level",
            "--xgmi-err",
            "--energy",
            "--throttle",
        )
        for output_format in ("json", "human_readable", "csv"):
            with self.subTest(output_format=output_format):
                options = [] if output_format == "human_readable" else ["--" + output_format]
                commands = self.commands(format=output_format)
                args = self.parse(
                    "--usage-fields", "gfx_activity,apu_average_ipu_reads", *sections, *options
                )
                try:
                    text = self.run_metric(commands, args)
                except KeyError as error:
                    self.fail(f"Selected field deleted before serialization: {error}")
                if output_format == "json":
                    gpu = json.loads(text)["gpu_data"][0]
                    self.assertEqual(
                        gpu["usage"],
                        {
                            "gfx_activity": "N/A",
                            "apu_average_ipu_reads": {"value": 0, "unit": "MB/s"},
                        },
                    )
                    # Selection must not change which sections the APU dump hides.
                    for section in ("pcie", "ecc_blocks", "energy"):
                        self.assertNotIn(section, gpu)
                elif output_format == "csv":
                    rows = list(csv.DictReader(io.StringIO(text)))
                    self.assertEqual(rows[0]["gfx_activity"], "N/A")
                    self.assertEqual(rows[0]["apu_average_ipu_reads"], "0")
                else:
                    self.assertIn("GFX_ACTIVITY: N/A", text)
                    self.assertIn("APU_AVERAGE_IPU_READS: 0 MB/s", text)

    def test_json_envelope_keeps_all_selected_gpus(self) -> None:
        commands = self.commands()
        commands.device_handles = [10, 11]
        args = self.parse("--usage-fields", "umc_activity,gfx_activity", "--json")
        result = json.loads(self.run_metric(commands, args))
        self.assertEqual(list(result), ["gpu_data"])
        self.assertEqual([row["gpu"] for row in result["gpu_data"]], [0, 1])
        for row in result["gpu_data"]:
            self.assertEqual(list(row["usage"]), ["umc_activity", "gfx_activity"])
            self.assertEqual(row["usage"]["umc_activity"], {"value": 100, "unit": "%"})
            self.assertEqual(row["usage"]["gfx_activity"], {"value": 0, "unit": "%"})
        self.assertEqual(args.usage_fields, ("umc_activity", "gfx_activity"))
        self.assertEqual(args.gpu, [10, 11])

    def test_human_output_retains_identity_order_and_units(self) -> None:
        commands = self.commands(format="human_readable")
        text = self.run_metric(
            commands, self.parse("--usage-fields", "umc_activity,gfx_activity,mm_activity")
        )
        self.assertIn("GPU: 0", text)
        self.assertIn("UMC_ACTIVITY: 100 %", text)
        self.assertIn("GFX_ACTIVITY: 0 %", text)
        self.assertIn("MM_ACTIVITY: N/A", text)
        self.assertLess(text.index("UMC_ACTIVITY"), text.index("GFX_ACTIVITY"))
        self.assertNotIn("POWER:", text)

    def test_watch_json_file_includes_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.json"
            commands = self.commands(destination=path)
            commands.device_handles = [10, 11]
            args = self.parse(
                "--usage-fields",
                "gfx_activity,mm_activity",
                "--json",
                "--watch",
                "1",
                "--iterations",
                "2",
                "--file",
                str(path),
            )
            before = copy.deepcopy(self.metrics)
            stdout = self.run_metric(commands, args)
            self.assertEqual(stdout, "'CTRL' + 'C' to stop watching output:\n")
            self.assertEqual(len(commands.logger.watch_output), 4)
            self.assertEqual(args.usage_fields, ("gfx_activity", "mm_activity"))
            self.assertEqual(args.gpu, [10, 11])
            self.assertEqual(self.metrics, before)
            # Same gpu_data document as a plain --usage watch.
            rows = json.loads(path.read_text(encoding="utf-8"))["gpu_data"]
            self.assertEqual([row["gpu"] for row in rows], [0, 1, 0, 1])
            for row in rows:
                self.assertEqual(list(row["usage"]), ["gfx_activity", "mm_activity"])
                self.assertEqual(row["usage"]["gfx_activity"], {"value": 0, "unit": "%"})
                self.assertEqual(row["usage"]["mm_activity"], "N/A")

    def test_watch_csv_file_includes_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.csv"
            commands = self.commands(format="csv", destination=path)
            commands.device_handles = [10, 11]
            args = self.parse(
                "--usage-fields",
                "gfx_activity,mm_activity",
                "--csv",
                "--watch",
                "1",
                "--iterations",
                "2",
                "--file",
                str(path),
            )
            stdout = self.run_metric(commands, args)
            self.assertEqual(stdout, "'CTRL' + 'C' to stop watching output:\n")
            reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
            rows = list(reader)
            self.assertEqual(reader.fieldnames, ["timestamp", "gpu", "gfx_activity", "mm_activity"])
            self.assertEqual([row["gpu"] for row in rows], ["0", "1", "0", "1"])
            for row in rows:
                self.assertTrue(row["timestamp"].isdigit())
                self.assertEqual(row["gfx_activity"], "0")
                self.assertEqual(row["mm_activity"], "N/A")
            self.assertEqual(args.usage_fields, ("gfx_activity", "mm_activity"))

    def test_watch_human_file_includes_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.txt"
            commands = self.commands(format="human_readable", destination=path)
            commands.device_handles = [10, 11]
            args = self.parse(
                "--usage-fields",
                "gfx_activity,mm_activity",
                "--watch",
                "1",
                "--iterations",
                "2",
                "--file",
                str(path),
            )
            stdout = self.run_metric(commands, args)
            self.assertEqual(stdout, "'CTRL' + 'C' to stop watching output:\n")
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("GFX_ACTIVITY: 0 %"), 4)
            self.assertEqual(text.count("MM_ACTIVITY: N/A"), 4)
            self.assertEqual(text.count("TIMESTAMP:"), 4)
            self.assertEqual(text.count("GPU: 0"), 2)
            self.assertEqual(text.count("GPU: 1"), 2)
            self.assertEqual(args.usage_fields, ("gfx_activity", "mm_activity"))

    def test_single_gpu_watch_time_keeps_selector(self) -> None:
        commands = self.commands(format="csv")
        args = self.parse(
            "--usage-fields",
            "gfx_activity",
            "--csv",
            "--watch",
            "1",
            "--watch_time",
            "100",
            "--iterations",
            "2",
        )
        self.interface.amdsmi_get_gpu_activity.side_effect = [
            {"gfx_activity": 0, "umc_activity": 0, "mm_activity": 0},
            {"gfx_activity": 17, "umc_activity": 0, "mm_activity": 0},
        ]
        with mock.patch.object(self.helper_module.time, "time", return_value=1000):
            text = self.run_metric(commands, args)
        self.assertEqual([row["gfx_activity"] for row in commands.logger.watch_output], [0, 17])
        self.assertEqual(args.usage_fields, ("gfx_activity",))
        frames = text.split("\n", 1)[1].strip().split("\n\n")
        self.assertEqual(len(frames), 2)
        for frame, value in zip(frames, ("0", "17")):
            reader = csv.DictReader(io.StringIO(frame))
            self.assertEqual(reader.fieldnames, ["timestamp", "gpu", "gfx_activity"])
            self.assertEqual(
                list(reader), [{"timestamp": "1000", "gpu": "0", "gfx_activity": value}]
            )
        self.helper_module.time.sleep.assert_called_once_with(1)

    def test_watch_csv_tracks_changed_nested_availability(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.side_effect = [
            {"xcp_stats.gfx_busy_inst": "N/A"},
            {"xcp_stats.gfx_busy_inst": [[0, 100]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.csv"
            commands = self.commands(format="csv", destination=path)
            args = self.parse(
                "--usage-fields",
                "gfx_busy_inst,gfx_activity",
                "--partition",
                "--csv",
                "--watch",
                "1",
                "--iterations",
                "2",
                "--file",
                str(path),
            )
            stdout = self.run_metric(commands, args)
            self.assertEqual(stdout, "'CTRL' + 'C' to stop watching output:\n")
            reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
            rows = list(reader)
            self.assertEqual(
                reader.fieldnames,
                ["timestamp", "gpu", "gfx_busy_inst", "gfx_busy_inst_xcp_0", "gfx_activity"],
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["gfx_busy_inst_xcp_0"], "N/A")
            self.assertEqual(rows[1]["gfx_busy_inst_xcp_0"], "[0, 100]")
            for row in rows:
                self.assertEqual(row["gfx_busy_inst"], "N/A")
                self.assertEqual(row["gfx_activity"], "0")
                self.assertEqual(row["gpu"], "0")

    def test_watch_stopped_before_first_sample_keeps_existing_file(self) -> None:
        self.interface.amdsmi_get_gpu_activity.side_effect = KeyboardInterrupt
        for selection in (("--usage",), ("--usage-fields", "gfx_activity")):
            for output_format in ("json", "csv", "human_readable"):
                with self.subtest_directory(selection=selection, format=output_format) as directory:
                    path = Path(directory) / "watch-output"
                    path.write_text("existing user data\n", encoding="utf-8")
                    commands = self.commands(format=output_format, destination=path)
                    options = [] if output_format == "human_readable" else ["--" + output_format]
                    args = self.parse(
                        *selection, "--watch", "1", "--file", str(path), "--append", *options
                    )
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_metric(commands, args)
                    self.assertEqual(path.read_text(encoding="utf-8"), "existing user data\n")

    def test_unselected_watch_files_include_every_sample(self) -> None:
        for output_format in ("json", "csv", "human_readable"):
            with self.subtest_directory(format=output_format) as directory:
                path = Path(directory) / "watch-output"
                commands = self.commands(format=output_format, destination=path)
                commands.device_handles = [10, 11]
                options = [] if output_format == "human_readable" else ["--" + output_format]
                args = self.parse(
                    "--usage", "--watch", "1", "--iterations", "2", "--file", str(path), *options
                )
                self.run_metric(commands, args)
                text = path.read_text(encoding="utf-8")
                if output_format == "json":
                    # Normal completion keeps the existing gpu_data document.
                    rows = json.loads(text)["gpu_data"]
                    self.assertEqual([row["gpu"] for row in rows], [0, 1, 0, 1])
                elif output_format == "csv":
                    rows = list(csv.DictReader(io.StringIO(text)))
                    self.assertEqual([row["gpu"] for row in rows], ["0", "1", "0", "1"])
                else:
                    self.assertEqual(text.count("TIMESTAMP:"), 4)
                    self.assertEqual(text.count("GFX_ACTIVITY: 0 %"), 4)

    def test_unselected_watch_csv_is_stable_and_does_not_mutate_rows(self) -> None:
        self.interface.amdsmi_get_gpu_partition_metrics_info.side_effect = [
            {"xcp_stats.gfx_busy_inst": "N/A"},
            {"xcp_stats.gfx_busy_inst": [[0, 100]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watch.csv"
            commands = self.commands(format="csv", destination=path)
            args = self.parse(
                "--usage",
                "--partition",
                "--csv",
                "--watch",
                "1",
                "--iterations",
                "2",
                "--file",
                str(path),
            )
            self.run_metric(commands, args)
            reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            fields = reader.fieldnames
            # First-seen column order, so a key that appears later is appended.
            self.assertEqual(fields[-1], "gfx_busy_inst_xcp_0")
            self.assertLess(fields.index("gfx_activity"), fields.index("gfx_busy_inst"))
            self.assertEqual(rows[0]["gfx_busy_inst"], "N/A")
            self.assertEqual(rows[0]["gfx_busy_inst_xcp_0"], "N/A")
            self.assertEqual(rows[1]["gfx_busy_inst_xcp_0"], "[0, 100]")
            self.assertNotIn("gfx_busy_inst_xcp_0", commands.logger.watch_output[0])
            self.assertNotIn("gfx_busy_inst", commands.logger.watch_output[1])

    def test_non_watch_files_keep_complete_output(self) -> None:
        for output_format in ("json", "csv", "human_readable"):
            with self.subtest_directory(output_format=output_format) as directory:
                path = Path(directory) / "usage-output"
                commands = self.commands(format=output_format, destination=path)
                commands.device_handles = [10, 11]
                options = [] if output_format == "human_readable" else ["--" + output_format]
                args = self.parse(
                    "--usage-fields",
                    "gfx_activity,apu_average_ipu_reads",
                    "--file",
                    str(path),
                    *options,
                )
                self.assertEqual(self.run_metric(commands, args), "")
                text = path.read_text(encoding="utf-8")
                if output_format == "json":
                    output = json.loads(text)
                    self.assertEqual(list(output), ["gpu_data"])
                    rows = output["gpu_data"]
                    self.assertEqual([row["gpu"] for row in rows], [0, 1])
                    for row in rows:
                        self.assertEqual(
                            row["usage"],
                            {
                                "gfx_activity": {"value": 0, "unit": "%"},
                                "apu_average_ipu_reads": "N/A",
                            },
                        )
                elif output_format == "csv":
                    reader = csv.DictReader(io.StringIO(text))
                    rows = list(reader)
                    self.assertEqual(
                        reader.fieldnames, ["gpu", "gfx_activity", "apu_average_ipu_reads"]
                    )
                    self.assertEqual([row["gpu"] for row in rows], ["0", "1"])
                    for row in rows:
                        self.assertEqual(row["gfx_activity"], "0")
                        self.assertEqual(row["apu_average_ipu_reads"], "N/A")
                else:
                    self.assertEqual(text.count("GFX_ACTIVITY: 0 %"), 2)
                    self.assertEqual(text.count("APU_AVERAGE_IPU_READS: N/A"), 2)
                    self.assertEqual(text.count("GPU: 0"), 1)
                    self.assertEqual(text.count("GPU: 1"), 1)

    def test_apu_missing_selection_survives_with_power(self) -> None:
        self.metrics["is_apu"] = True
        result = self.run_gpu(
            self.parse("--usage-fields", "apu_average_ipu_reads,apu_average_mm_activity", "--power")
        )
        self.assertEqual(
            result["usage"], {"apu_average_ipu_reads": "N/A", "apu_average_mm_activity": "N/A"}
        )
        self.assertIn("power", result)

    def test_watch_json_stdout_keeps_existing_stream(self) -> None:
        for options in (("--usage",), ("--usage-fields", "gfx_activity,mm_activity")):
            with self.subTest(options=options):
                commands = self.commands()
                commands.device_handles = [10, 11]
                args = self.parse(*options, "--json", "--watch", "1", "--iterations", "2")
                text = self.run_metric(commands, args)
                banner, text = text.split("\n", 1)
                self.assertEqual(banner, "'CTRL' + 'C' to stop watching output:")
                documents = []
                while text.strip():
                    document, end = json.JSONDecoder().raw_decode(text.lstrip())
                    documents.append(document)
                    text = text.lstrip()[end:]
                self.assertEqual(len(documents), 3)
                for rows in documents[:2]:
                    self.assertEqual([row["gpu"] for row in rows], [0, 1])
                    for row in rows:
                        self.assertIsInstance(row["timestamp"], int)
                        self.assertEqual(row["usage"]["gfx_activity"], {"value": 0, "unit": "%"})
                        self.assertEqual(row["usage"]["mm_activity"], "N/A")
                        self.assertEqual(len(row["usage"]), 2 if args.usage_fields else 8)
                self.assertEqual(list(documents[2]), ["gpu_data"])
                self.assertEqual([row["gpu"] for row in documents[2]["gpu_data"]], [0, 1, 0, 1])
                self.assertTrue(all("timestamp" not in row for row in documents[2]["gpu_data"]))

    def test_watch_human_stdout_keeps_each_sample(self) -> None:
        for options in (("--usage",), ("--usage-fields", "gfx_activity,mm_activity")):
            with self.subTest(options=options):
                commands = self.commands(format="human_readable")
                commands.device_handles = [10, 11]
                args = self.parse(*options, "--watch", "1", "--iterations", "2")
                text = self.run_metric(commands, args)
                self.assertEqual(text.count("TIMESTAMP:"), 4)
                self.assertEqual(text.count("GPU: 0"), 2)
                self.assertEqual(text.count("GPU: 1"), 2)
                self.assertEqual(text.count("GFX_ACTIVITY: 0 %"), 4)
                self.assertEqual(text.count("MM_ACTIVITY: N/A"), 4)
                self.assertEqual(text.count("VCN_BUSY: 23 %"), 0 if args.usage_fields else 4)
                self.assertNotIn("POWER:", text)

    def test_legacy_non_watch_files_match_stdout(self) -> None:
        for output_format in ("json", "csv", "human_readable"):
            with self.subtest_directory(output_format=output_format) as directory:
                options = [] if output_format == "human_readable" else ["--" + output_format]
                commands = self.commands(format=output_format)
                commands.device_handles = [10, 11]
                expected = self.run_metric(commands, self.parse("--usage", *options))
                self.assertIn(
                    "vcn_busy" if output_format != "human_readable" else "VCN_BUSY", expected
                )
                path = Path(directory) / "legacy-output"
                commands = self.commands(format=output_format, destination=path)
                commands.device_handles = [10, 11]
                args = self.parse("--usage", *options, "--file", str(path))
                self.assertEqual(self.run_metric(commands, args), "")
                self.assertEqual(
                    path.read_text(encoding="utf-8").strip().splitlines(),
                    expected.strip().splitlines(),
                )
                self.assertIsNone(args.usage_fields)
