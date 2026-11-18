#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi fabric`` labels the fabric bandwidth in MB/s.

The library passes the driver's value through unchanged, so the unit exists
only in the CLI's labels: the human-readable text and the JSON ``unit``. The
CSV column ``bandwidth_mb_s`` and the value itself stay as they are.

Stubs the library and runs the real ``fabric`` command and logger.
"""

import argparse
import csv
import io
import json
import os
import unittest
from contextlib import redirect_stdout

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
FABRIC_PATH = os.path.join(_CLI_DIR, "subcommands", "fabric.py") if _CLI_DIR else None
LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else None

_BANDWIDTH = 12500
_UNSET = (1 << 32) - 1
_FABRIC_INFO = {
    "bdf": "0000:01:00.1",
    "version": _UNSET,
    "accelerator_id": 1,
    "fabric_type": "UALOE",
    "bandwidth": _BANDWIDTH,
    "latency": 0,
    "ppod_id": list(range(16)),
    "ppod_size": 4,
    "vpod_id": 1,
    "vpod_size": 4,
    "local_accelerators": [0, 1, 2, 3] + [_UNSET] * 12,
    "vpod_active_accelerators": [_UNSET] * 32,
    "addr_mode": "SOURCE_ALIASING",
    "accel_state": "ACTIVE",
}


class _FakeHelpers:
    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0


class TestCliFabricBandwidthUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interface = fake_module(
            "amdsmi.amdsmi_interface",
            amdsmi_get_gpu_device_bdf=lambda _handle: "0000:01:00.0",
            amdsmi_get_gpu_fabric_info=lambda _handle: dict(_FABRIC_INFO),
        )
        exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=Exception)
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi.amdsmi_exception": exception,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
            },
        )
        cls.fabric_module = load_cli_module("fabric_bandwidth_unit_under_test", FABRIC_PATH)
        cls.logger_module = load_cli_module("logger_bandwidth_unit_under_test", LOGGER_PATH)

    def _fabric_topology(self, output_format):
        """Run `amd-smi fabric --topology` on one GPU and return what it printed."""
        commands = self.fabric_module.FabricCommands()
        commands.helpers = _FakeHelpers()
        commands.logger = self.logger_module.AMDSMILogger(
            format=output_format, helpers=commands.helpers
        )
        args = argparse.Namespace(gpu=object(), topology=True, telemetry=False)
        printed = io.StringIO()
        with redirect_stdout(printed):
            commands.fabric(args)
        return printed.getvalue()

    def test_human_readable_labels_megabytes(self):
        printed = self._fabric_topology("human_readable")

        self.assertIn(f"BANDWIDTH: {_BANDWIDTH} MB/s", printed)

    def test_json_unit_is_megabytes(self):
        document = json.loads(self._fabric_topology("json"))

        bandwidth = document[0]["fabric"]["fabric_info"]["bandwidth"]
        self.assertEqual(bandwidth, {"value": _BANDWIDTH, "unit": "MB/s"})

    def test_csv_keeps_column_and_value(self):
        rows = list(csv.DictReader(io.StringIO(self._fabric_topology("csv"))))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bandwidth_mb_s"], str(_BANDWIDTH))
