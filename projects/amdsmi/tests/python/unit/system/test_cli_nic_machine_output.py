#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for AI-NIC output in machine-readable ``amd-smi`` runs.

``--json`` and ``--csv`` output must stay one parseable document. ``list``
used to print the GPU records and then a second document for each AI-NIC, and
``static --nic ... --json`` never flushed the NIC records it had collected. Both
break any consumer that parses stdout or the ``--file`` target. The subcommand
mixins are loaded with their collaborators stubbed, so no GPU or NIC is needed.
"""

import argparse
import os
import unittest

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
LIST_PATH = os.path.join(_CLI_DIR, "subcommands", "list_devices.py") if _CLI_DIR else None
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None


class _Handle:
    """Stands in for a device handle; the CLI only compares ``.value``."""

    def __init__(self, value):
        self.value = value


class _FakeHelpers:
    def __init__(self, is_ainic_initialized=True):
        self._is_ainic_initialized = is_ainic_initialized

    def is_ainic_initialized(self):
        return self._is_ainic_initialized


class _FakeLogger:
    def __init__(self, fmt):
        self._fmt = fmt
        self.output = {}
        self.store_nic_json_output = []
        self.combine_calls = 0
        self.no_ainics_calls = 0

    def is_json_format(self):
        return self._fmt == "json"

    def is_csv_format(self):
        return self._fmt == "csv"

    def is_human_readable_format(self):
        return self._fmt == "human"

    def clear_multiple_devices_output(self):
        pass

    def combine_arrays_to_json(self):
        self.combine_calls += 1

    def print_no_ainics_found(self):
        self.no_ainics_calls += 1


def _stub_cli_imports(test_cls):
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=Exception)
    interface = fake_module("amdsmi.amdsmi_interface")
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    stub_modules(
        test_cls,
        {
            "amdsmi": amdsmi_pkg,
            "amdsmi.amdsmi_interface": interface,
            "amdsmi.amdsmi_exception": exception,
            "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
            "amdsmi_cli_exceptions": fake_module(
                "amdsmi_cli_exceptions",
                AmdSmiInvalidParameterException=type(
                    "AmdSmiInvalidParameterException", (Exception,), {}
                ),
            ),
        },
    )


class TestListDevicesNicOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LIST_PATH or not os.path.isfile(LIST_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI list_devices.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        _stub_cli_imports(cls)
        cls.module = load_cli_module("list_devices_under_test", LIST_PATH)

    def _make_commands(self, fmt):
        gpu = _Handle(1)
        nic = _Handle(2)
        commands = self.module.ListDevicesCommands()
        commands.helpers = _FakeHelpers()
        commands.logger = _FakeLogger(fmt)
        commands.device_handles_gpus = [gpu]
        commands.device_handles_ainics = [nic]
        commands.listed_gpus = []
        commands.listed_ainics = []
        commands.list_gpu = lambda args, multiple_devices=False, gpu=None: (
            commands.listed_gpus.append(gpu)
        )
        commands.list_ainic = lambda args, multiple_devices=False, nic=None: (
            commands.listed_ainics.append(nic)
        )
        return commands, nic

    def test_json_lists_gpus_without_a_second_nic_document(self):
        commands, _nic = self._make_commands("json")
        commands.list_devices(argparse.Namespace(gpu=None, nic=None))
        self.assertEqual(len(commands.listed_gpus), 1)
        self.assertEqual(commands.listed_ainics, [])

    def test_csv_lists_gpus_without_a_second_nic_table(self):
        commands, _nic = self._make_commands("csv")
        commands.list_devices(argparse.Namespace(gpu=None, nic=None))
        self.assertEqual(len(commands.listed_gpus), 1)
        self.assertEqual(commands.listed_ainics, [])

    def test_human_readable_still_lists_gpus_then_nics(self):
        commands, nic = self._make_commands("human")
        commands.list_devices(argparse.Namespace(gpu=None, nic=None))
        self.assertEqual(len(commands.listed_gpus), 1)
        self.assertEqual(commands.listed_ainics, [[nic]])

    def test_json_with_explicit_nic_selector_lists_the_nic(self):
        commands, nic = self._make_commands("json")
        commands.list_devices(argparse.Namespace(gpu=None, nic=[nic]))
        self.assertEqual(commands.listed_gpus, [])
        self.assertEqual(commands.listed_ainics, [[nic]])


class TestStaticNicJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        _stub_cli_imports(cls)
        cls.module = load_cli_module("static_nic_under_test", STATIC_PATH)

    def _make_commands(self, fmt):
        commands = self.module.StaticCommands()
        commands.helpers = _FakeHelpers()
        commands.logger = _FakeLogger(fmt)
        commands.device_handles_ainics = [_Handle(2)]

        def store_nic_record(args, multiple_devices=False, nic=None):
            commands.logger.store_nic_json_output.append({"ai_nic": 0})

        commands._static_ainic = store_nic_record
        return commands

    def test_json_nic_query_flushes_collected_records_once(self):
        commands = self._make_commands("json")
        args = argparse.Namespace(nic=commands.device_handles_ainics)
        self.assertTrue(commands._static_nics(args, False, None))
        self.assertEqual(commands.logger.store_nic_json_output, [{"ai_nic": 0}])
        self.assertEqual(commands.logger.combine_calls, 1)

    def test_json_nic_query_with_no_nics_does_not_add_a_second_document(self):
        commands = self._make_commands("json")
        args = argparse.Namespace(nic=[])
        self.assertTrue(commands._static_nics(args, False, None))
        self.assertEqual(commands.logger.no_ainics_calls, 1)
        self.assertEqual(commands.logger.combine_calls, 0)

    def test_human_readable_nic_query_does_not_emit_json(self):
        commands = self._make_commands("human")
        args = argparse.Namespace(nic=commands.device_handles_ainics)
        self.assertTrue(commands._static_nics(args, False, None))
        self.assertEqual(commands.logger.combine_calls, 0)


if __name__ == "__main__":
    unittest.main()
