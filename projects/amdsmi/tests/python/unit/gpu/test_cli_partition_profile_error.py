#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi partition`` when the profile query fails.

The Python wrapper absorbs AMDSMI_STATUS_NOT_SUPPORTED, so the CLI only sees an
exception for other errors. The row for that GPU must show N/A instead of
crashing, reusing another GPU's partition ID, or reporting partition 0.
"""

import os
import types
import unittest

try:
    from common.common import (
        amdsmi_path,
        cli_search_order,
        find_cli_dir,
        load_cli_module,
        stub_modules,
    )
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None
    cli_search_order = None
    find_cli_dir = None

_CLI_DIR = (
    find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
    if find_cli_dir and cli_search_order
    else None
)
PARTITION_PATH = os.path.join(_CLI_DIR, "subcommands", "partition.py") if _CLI_DIR else ""


class _FakeLibraryException(Exception):
    def get_error_code(self):
        return -1

    def get_error_info(self, detailed=True):
        return "mock error"


def _build_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    exception.AmdSmiLibraryException = _FakeLibraryException
    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
    }


class _Logger:
    def __init__(self):
        self.table_header = ""
        self.table_title = ""
        self.multiple_device_output = []
        self.rows = []
        self.tables = []

    def is_json_format(self):
        return False

    def is_human_readable_format(self):
        return False

    def print_output(self, *args, **kwargs):
        self.rows.extend(self.multiple_device_output)
        self.tables.append(list(self.multiple_device_output))

    def clear_multiple_devices_output(self):
        self.multiple_device_output = []


class _Helpers:
    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, handle):
        return int(handle[-1])


class TestPartitionProfileError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARTITION_PATH or not os.path.isfile(PARTITION_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI partition.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = load_cli_module(
            "partition_current_under_test", PARTITION_PATH, sys_path_dir=_CLI_DIR
        )

    def _current(self, failing_gpus):
        def _profile(handle):
            if handle in failing_gpus:
                raise _FakeLibraryException()
            return {
                "partition_id": [int(handle[-1])],
                "partition_profile": {"profile_type": "SPX", "profile_index": 0},
            }

        self.interface.amdsmi_get_gpu_accelerator_partition_profile = _profile
        self.interface.amdsmi_get_gpu_memory_partition = lambda _h: "NPS1"

        cmd = self.module.PartitionCommands()
        cmd.logger = _Logger()
        cmd.helpers = _Helpers()
        cmd.group_check_printed = True
        args = types.SimpleNamespace(
            gpu=["gpu0", "gpu1"], current=True, memory=False, accelerator=False
        )
        cmd.partition(args)
        return {row["gpu_id"]: row for row in cmd.logger.rows}

    def test_first_gpu_error_reports_na(self):
        rows = self._current({"gpu0"})
        self.assertEqual(rows[0]["partition_id"], "N/A")
        self.assertEqual(rows[0]["accelerator_type"], "N/A")
        self.assertEqual(rows[1]["partition_id"], "1")

    def test_later_gpu_error_does_not_reuse_previous_partition_id(self):
        rows = self._current({"gpu1"})
        self.assertEqual(rows[0]["partition_id"], "0")
        self.assertEqual(rows[1]["partition_id"], "N/A")

    def test_accelerator_error_reports_na(self):
        def _profile(handle):
            if handle == "gpu0":
                raise _FakeLibraryException()
            return {
                "partition_id": [1],
                "partition_profile": {"profile_type": "SPX", "profile_index": 0},
            }

        def _config(_handle):
            raise _FakeLibraryException()

        self.interface.amdsmi_get_gpu_accelerator_partition_profile = _profile
        self.interface.amdsmi_get_gpu_accelerator_partition_profile_config = _config

        cmd = self.module.PartitionCommands()
        cmd.logger = _Logger()
        cmd.helpers = _Helpers()
        cmd.group_check_printed = True
        args = types.SimpleNamespace(
            gpu=["gpu0", "gpu1"], current=False, memory=False, accelerator=True
        )
        cmd.partition(args)
        rows = {row["gpu_id"]: row for row in cmd.logger.tables[0]}
        self.assertEqual(rows[0]["partition_id"], "N/A")
        self.assertEqual(rows[1]["partition_id"], "1")
