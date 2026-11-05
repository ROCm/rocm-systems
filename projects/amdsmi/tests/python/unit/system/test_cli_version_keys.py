#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi version`` JSON and CSV output always carries every version key.

Scripts index ``amdgpu_version``, ``amd_hsmp_driver_version`` and
``nic_driver_version`` directly. On a host without the amd_hsmp/hsmp_acpi
driver (most Radeon workstations) the CPU query is skipped, and the key must
still be emitted as ``N/A`` rather than dropped; the same holds for any key
whose query was not requested. The command runs against a mocked library and
the real ``AMDSMILogger``, so these tests need no GPU, CPU driver or root.
"""

import csv
import json
import os
import pathlib
import tempfile
import types
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
_VERSION_PATH = os.path.join(_CLI_DIR, "subcommands", "version.py") if _CLI_DIR else None
_LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else None

_VERSION_KEYS = (
    "tool",
    "version",
    "amdsmi_library_version",
    "rocm_version",
    "amdgpu_version",
    "amd_hsmp_driver_version",
    "nic_driver_version",
)


class _FakeLibraryException(Exception):
    def get_error_info(self, detailed=True):
        return "AMDSMI_STATUS_NOT_SUPPORTED"


def _fake_amdsmi_modules():
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_get_lib_version=lambda: {"major": 27, "minor": 0, "release": 0},
        amdsmi_get_rocm_version=lambda: (True, "10.0.0"),
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_driver_info=lambda _h: {"driver_version": "7.1.3"},
        amdsmi_get_cpu_handles=lambda: {"processor_handles": ["cpu0"]},
        amdsmi_get_cpu_hsmp_driver_version=lambda _h: {
            "hsmp_driver_major_ver_num": 2,
            "hsmp_driver_minor_ver_num": 5,
        },
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "_version": fake_module("_version", __version__="27.0.0+test"),
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi_helpers": fake_module(
            "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
        ),
    }


class TestCliVersionKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _VERSION_PATH or not os.path.isfile(_VERSION_PATH):
            raise unittest.SkipTest(f"amd-smi CLI version.py not found (looked in {_CLI_DIR})")
        stub_modules(cls, _fake_amdsmi_modules())
        cls.version_module = load_cli_module("amdsmi_version_under_test", _VERSION_PATH)
        cls.logger_module = load_cli_module("amdsmi_logger_under_test", _LOGGER_PATH)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out_path = pathlib.Path(tmp.name) / "version.out"

    def _run(self, fmt, hsmp_initialized, gpu=None, cpu=None, nic=None):
        helpers = types.SimpleNamespace(
            check_required_groups=lambda: None,
            is_amd_hsmp_initialized=lambda: hsmp_initialized,
            get_ainic_handles=lambda: [],
        )
        commands = self.version_module.VersionCommands()
        commands.helpers = helpers
        commands.group_check_printed = True
        commands.logger = self.logger_module.AMDSMILogger(
            format=fmt, destination=self.out_path, helpers=helpers
        )
        args = types.SimpleNamespace(gpu_version=gpu, cpu_version=cpu, nic_version=nic)
        commands.version(args)
        return self.out_path.read_text(encoding="utf-8")

    def _json_record(self, **kwargs):
        output = json.loads(self._run("json", **kwargs))
        self.assertIsInstance(output, list, "version --json must emit a list")
        self.assertEqual(len(output), 1, "version --json must emit one record")
        return output[0]

    def _csv_record(self, **kwargs):
        rows = list(csv.DictReader(self._run("csv", **kwargs).splitlines()))
        self.assertEqual(len(rows), 1, "version --csv must emit one header and one row")
        return rows[0]

    def test_json_without_hsmp_driver_reports_na(self):
        record = self._json_record(hsmp_initialized=False)
        for key in _VERSION_KEYS:
            self.assertIn(key, record, f"version --json dropped '{key}' without amd_hsmp")
        self.assertEqual(record["amd_hsmp_driver_version"], "N/A")
        self.assertEqual(record["amdgpu_version"], "7.1.3")

    def test_csv_without_hsmp_driver_reports_na(self):
        record = self._csv_record(hsmp_initialized=False)
        for key in _VERSION_KEYS:
            self.assertIn(key, record, f"version --csv dropped '{key}' without amd_hsmp")
        self.assertEqual(record["amd_hsmp_driver_version"], "N/A")

    def test_json_with_hsmp_driver_reports_version(self):
        record = self._json_record(hsmp_initialized=True)
        self.assertEqual(record["amd_hsmp_driver_version"], "2.5")

    def test_json_gpu_version_only_keeps_unqueried_keys(self):
        record = self._json_record(hsmp_initialized=True, gpu=True)
        for key in _VERSION_KEYS:
            self.assertIn(key, record, f"version --gpu_version --json dropped '{key}'")
        self.assertEqual(record["amdgpu_version"], "7.1.3")
        self.assertEqual(record["amd_hsmp_driver_version"], "N/A")
        self.assertEqual(record["nic_driver_version"], "N/A")
