#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only XGMI output checks for compact metric rows and physical statuses."""

import argparse
import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

from amdsmi import amdsmi_wrapper
from amdsmi.amdsmi_interface import AMDSMI_MAX_NUM_XGMI_LINKS

_CLI_DIR = find_cli_dir(*cli_search_order(Path(__file__).parent))
_BDFS = ("0000:01:00.0", "0000:02:00.0")


class _FakeLibraryException(Exception):
    def get_error_info(self) -> str:
        return str(self)


class TestCliXgmiLinkStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI not found")
        cls.interface = fake_module(
            "amdsmi.amdsmi_interface",
            AMDSMI_MAX_NUM_XGMI_LINKS=AMDSMI_MAX_NUM_XGMI_LINKS,
            amdsmi_wrapper=amdsmi_wrapper,
        )
        exception = fake_module(
            "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
        )
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=cls.interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": cls.interface,
                "amdsmi.amdsmi_exception": exception,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
            },
        )
        cls.xgmi_module = load_cli_module(
            "xgmi_under_test", Path(_CLI_DIR) / "subcommands" / "xgmi.py"
        )
        cls.logger_module = load_cli_module(
            "xgmi_logger_under_test", Path(_CLI_DIR) / "amdsmi_logger.py"
        )

    def setUp(self) -> None:
        self.bdfs = list(_BDFS)
        self.metrics = {
            "num_links": 1,
            "links": [
                {"bdf": _BDFS[1], "read": 10, "write": 20, "bit_rate": 32, "max_bandwidth": 64}
            ]
            + [
                {"bdf": "0000:00:00.0", "read": 0, "write": 0, "bit_rate": 0, "max_bandwidth": 0}
                for _ in range(AMDSMI_MAX_NUM_XGMI_LINKS - 1)
            ],
        }
        self.statuses = ["D", "U"] + ["X"] * (AMDSMI_MAX_NUM_XGMI_LINKS - 2)
        self.interface.amdsmi_get_gpu_device_bdf = lambda handle: self.bdfs[handle]
        self.interface.amdsmi_get_processor_handle_from_bdf = self.bdfs.index
        self.interface.amdsmi_get_link_metrics = mock.Mock(return_value=self.metrics)
        self.interface.amdsmi_get_gpu_xgmi_link_status = mock.Mock(
            return_value={"status": self.statuses}
        )
        self.interface.amdsmi_topo_get_link_type = mock.Mock(
            return_value={"type": amdsmi_wrapper.AMDSMI_LINK_TYPE_XGMI}
        )

    def _run_xgmi(
        self,
        fmt: str = "json",
        *,
        metric: bool = False,
        link_status: bool = True,
        source_status: bool = False,
    ) -> str:
        commands = self.xgmi_module.XgmiCommands()
        commands.helpers = mock.Mock()
        commands.helpers._get_socket_counts.return_value = (2, 2, 0)
        commands.helpers.is_primary_partition.return_value = True
        commands.helpers.get_gpu_id_from_device_handle.side_effect = lambda handle: handle
        commands.helpers.unit_format.side_effect = lambda logger, value, unit: (
            {"value": value, "unit": unit} if logger.is_json_format() and value != "N/A" else value
        )
        commands.logger = self.logger_module.AMDSMILogger(format=fmt, helpers=commands.helpers)
        commands.group_check_printed = True
        args = argparse.Namespace(
            gpu=[0, 1], metric=metric, link_status=link_status, source_status=source_status
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            commands.xgmi(args)
        return output.getvalue()

    def test_compacted_peer_does_not_inherit_skipped_physical_status(self) -> None:
        output = json.loads(self._run_xgmi())
        self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_padded_bdf_is_not_a_measured_peer(self) -> None:
        self.bdfs[1] = "0000:00:00.0"
        output = json.loads(self._run_xgmi(metric=True, link_status=False))
        peer = output["xgmi_metric"][0][0]["link_metrics"]["links"][1]
        self.assertEqual(peer["read"], "N/A")
        self.assertEqual(peer["write"], "N/A")

    def test_no_counted_links_has_no_metrics(self) -> None:
        for count, empty in ((0, False), (None, False), (0, True)):
            with self.subTest(count=count, empty=empty):
                if count is None:
                    self.metrics.pop("num_links", None)
                else:
                    self.metrics["num_links"] = count
                if empty:
                    self.metrics["links"] = []
                output = json.loads(self._run_xgmi(metric=True, link_status=False))
                metrics = output["xgmi_metric"][0][0]["link_metrics"]
                self.assertEqual(metrics["bit_rate"], "N/A")
                self.assertEqual(metrics["max_bandwidth"], "N/A")
                self.assertEqual(metrics["links"][1]["read"], "N/A")
                self.assertEqual(metrics["links"][1]["write"], "N/A")
                status = json.loads(self._run_xgmi())["link_status"][0]["link_status"]
                self.assertEqual(status, ["SELF", "N/A"])

    def test_interior_compaction_with_equal_short_status_count_is_unavailable(self) -> None:
        self.metrics["num_links"] = AMDSMI_MAX_NUM_XGMI_LINKS - 1
        self.metrics["links"][0]["bdf"] = "0000:03:00.0"
        self.metrics["links"][1]["bdf"] = _BDFS[1]
        self.statuses[1:3] = ["D", "U"]
        for status_count in (AMDSMI_MAX_NUM_XGMI_LINKS, AMDSMI_MAX_NUM_XGMI_LINKS - 1):
            with self.subTest(status_count=status_count):
                self.interface.amdsmi_get_gpu_xgmi_link_status.return_value = {
                    "status": self.statuses[:status_count]
                }
                output = json.loads(self._run_xgmi())
                self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_full_capacity_preserves_multiple_link_statuses(self) -> None:
        self.metrics["num_links"] = AMDSMI_MAX_NUM_XGMI_LINKS
        self.metrics["links"][2]["bdf"] = _BDFS[1]
        self.metrics["links"][-1]["bdf"] = _BDFS[1]
        self.statuses[-1] = "U"
        for middle_status, expected in (("X", "D/X/U"), ("N/A", "D/U")):
            with self.subTest(middle_status=middle_status):
                self.statuses[2] = middle_status
                output = json.loads(self._run_xgmi())
                self.assertEqual(output["link_status"][0]["link_status"], ["SELF", expected])

    def test_self_is_not_replaced_by_a_measured_link_status(self) -> None:
        self.metrics["num_links"] = AMDSMI_MAX_NUM_XGMI_LINKS
        self.metrics["links"][0]["bdf"] = _BDFS[0]
        output = json.loads(self._run_xgmi())
        self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_full_capacity_requires_full_physical_status_span(self) -> None:
        self.metrics["num_links"] = AMDSMI_MAX_NUM_XGMI_LINKS
        for status_count in (0, 1, AMDSMI_MAX_NUM_XGMI_LINKS - 1):
            with self.subTest(status_count=status_count):
                self.interface.amdsmi_get_gpu_xgmi_link_status.return_value = {
                    "status": self.statuses[:status_count]
                }
                output = json.loads(self._run_xgmi())
                self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_compacted_and_dense_status_in_human_and_csv_output(self) -> None:
        for count, expected in ((1, "N/A"), (AMDSMI_MAX_NUM_XGMI_LINKS, "D")):
            self.metrics["num_links"] = count
            for fmt in ("human_readable", "csv"):
                with self.subTest(count=count, fmt=fmt):
                    output = self._run_xgmi(fmt)
                    if fmt == "csv":
                        row = next(csv.DictReader(io.StringIO(output)))
                        self.assertEqual(row[_BDFS[0]], "SELF")
                        self.assertEqual(row[_BDFS[1]], expected)
                    else:
                        row = next(line for line in output.splitlines() if _BDFS[0] in line)
                        self.assertEqual(row.split()[-2:], ["SELF", expected])

    def test_multiple_counted_links_accumulate_traffic(self) -> None:
        self.metrics["num_links"] = 2
        self.metrics["links"][1].update(bdf=_BDFS[1], read=3, write=5)
        self.metrics["links"][2].update(bdf=_BDFS[1], read=100, write=200)
        output = json.loads(self._run_xgmi(metric=True, link_status=False))
        metrics = output["xgmi_metric"][0][0]["link_metrics"]
        self.assertEqual(metrics["bit_rate"], {"value": 32, "unit": "Gb/s"})
        self.assertEqual(metrics["max_bandwidth"], {"value": 64, "unit": "Gb/s"})
        self.assertEqual(metrics["link_type"], "XGMI")
        self.assertEqual(metrics["links"][1]["read"], {"value": 13, "unit": "KB"})
        self.assertEqual(metrics["links"][1]["write"], {"value": 25, "unit": "KB"})

    def test_status_failure_and_unsupported_values_stay_unavailable(self) -> None:
        self.metrics["num_links"] = AMDSMI_MAX_NUM_XGMI_LINKS
        for error in (None, _FakeLibraryException("not supported")):
            with self.subTest(error=error):
                self.interface.amdsmi_get_gpu_xgmi_link_status.side_effect = error
                self.interface.amdsmi_get_gpu_xgmi_link_status.return_value = {
                    "status": ["N/A"] * AMDSMI_MAX_NUM_XGMI_LINKS
                }
                output = json.loads(self._run_xgmi())
                self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_metric_failure_is_unavailable(self) -> None:
        self.interface.amdsmi_get_link_metrics.side_effect = _FakeLibraryException("not supported")
        output = json.loads(self._run_xgmi(metric=True, link_status=False))
        metrics = output["xgmi_metric"][0][0]["link_metrics"]
        self.assertEqual(metrics["bit_rate"], "N/A")
        self.assertEqual(metrics["max_bandwidth"], "N/A")
        self.assertEqual(metrics["links"][1]["read"], "N/A")
        self.assertEqual(metrics["links"][1]["write"], "N/A")
        output = json.loads(self._run_xgmi())
        self.assertEqual(output["link_status"][0]["link_status"], ["SELF", "N/A"])

    def test_source_status_preserves_physical_slots(self) -> None:
        for link_status in (False, True):
            with self.subTest(link_status=link_status):
                output = json.loads(self._run_xgmi(source_status=True, link_status=link_status))
                self.assertEqual(output["link_port_status"][0]["link_status"], self.statuses)
