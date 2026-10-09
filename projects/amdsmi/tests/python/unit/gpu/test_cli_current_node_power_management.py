#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``CURRENT_NODE_POWER`` in ``amd-smi node -p``.

Drives the real ``NodeCommands.node()`` with the ``amdsmi`` package stubbed and
checks the new ``current_node_power`` field from ``amdsmi_get_npm_info()``:
human-readable, JSON, and CSV output, "N/A" pass-through, the
``AmdSmiLibraryException`` fallback, and that the CSV ``node`` column is emitted
only with ``-p``.
"""

import argparse
import importlib.util
import os
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
NODE_PATH = os.path.join(_CLI_DIR, "subcommands", "node.py") if _CLI_DIR else ""


class _FakeLibraryException(Exception):
    def __init__(self, message="mock error"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


_DEFAULT_NPM_INFO = {
    "limit": 6000,
    "status": 0,  # AMDSMI_NPM_STATUS_ENABLED
    "ubb_power_threshold": 6200,
    "max_node_power_limit": 6400,
    "current_node_power": 5800,
}


def _build_fake_modules():
    wrapper = fake_module("amdsmi.amdsmi_wrapper", AMDSMI_NPM_STATUS_DISABLED=1)
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        amdsmi_get_npm_info=lambda _h: dict(_DEFAULT_NPM_INFO),
        amdsmi_get_tray_info=lambda: {"max_acc_per_tray": 4, "tray_type": "OAM"},
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module(
        "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception, amdsmi_wrapper=wrapper
    )

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


def _load_node_module():
    spec = importlib.util.spec_from_file_location("node_under_test_power", NODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def __init__(self, output_format="human", destination="stdout"):
        self._format = output_format
        self.destination = destination
        self.output = None

    def is_json_format(self):
        return self._format == "json"

    def is_csv_format(self):
        return self._format == "csv"

    def is_human_readable_format(self):
        return self._format == "human"

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass


class _FakeHelpers:
    def check_required_groups(self):
        pass

    def unit_format(self, logger, value, unit):
        # Mirrors the real AMDSMIHelpers.unit_format's output-mode branches
        # so the CLI-under-test's formatting choices are exercised
        # faithfully, not just stubbed to a passthrough.
        if value == "N/A":
            return "N/A"
        if logger.is_json_format():
            return {"value": value, "unit": unit} if unit else value
        if logger.is_csv_format():
            return value
        if logger.is_human_readable_format():
            return f"{value} {unit}".rstrip() if unit else f"{value}".rstrip()
        return f"{value}"


def _build_node_args(**overrides):
    defaults = dict(
        nodes=None, power_management=True, base_board_temps=False, gtt=False, tray=False
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_node(node_module, output_format="human", destination="stdout", **arg_overrides):
    helpers = _FakeHelpers()
    logger = _FakeLogger(output_format=output_format, destination=destination)
    commands = object.__new__(node_module.NodeCommands)
    commands.logger = logger
    commands.helpers = helpers
    commands.group_check_printed = True
    commands.node_handle = object()

    commands.node(_build_node_args(**arg_overrides))
    return logger, commands


class TestNodePowerManagementCurrentNodePower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(NODE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI node.py not found at {NODE_PATH}")
        modules = _build_fake_modules()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]

    def setUp(self):
        # Reload fresh each test so per-test monkeypatches on the shared
        # interface module (e.g. amdsmi_get_npm_info raising) don't leak.
        self.interface.amdsmi_get_npm_info = lambda _h: dict(_DEFAULT_NPM_INFO)
        self.node_module = _load_node_module()

    def test_current_node_power_human_readable(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            _run_node(self.node_module, output_format="human", destination="stdout")

        self.assertIn("CURRENT_NODE_POWER: 5800 W", buf.getvalue())

    def test_current_node_power_json_format(self):
        _, commands = _run_node(self.node_module, output_format="json", destination="file")
        self.assertEqual(
            commands.logger.output["node"]["power_management"]["current_node_power"],
            {"value": 5800, "unit": "W"},
        )

    def test_current_node_power_csv_format(self):
        _, commands = _run_node(self.node_module, output_format="csv", destination="file")
        # CSV path stores raw values, no unit_format applied.
        self.assertEqual(commands.logger.output["current_node_power"], 5800)
        self.assertEqual(commands.logger.output["node"], 0)
        self.assertEqual(next(iter(commands.logger.output)), "node")

    def test_csv_omits_node_column_without_power_management(self):
        _, commands = _run_node(
            self.node_module,
            output_format="csv",
            destination="file",
            power_management=False,
            tray=True,
        )
        self.assertNotIn("node", commands.logger.output)
        self.assertEqual(commands.logger.output["tray_type"], "OAM")

    def test_current_node_power_na_key_absent_from_library_dict(self):
        # If current_node_power is missing from the dict entirely (e.g. an
        # older amdsmi build predating this field), the CLI's .get(..., "N/A")
        # default must apply rather than raising a KeyError.
        npm_info_without_current_node_power = dict(_DEFAULT_NPM_INFO)
        del npm_info_without_current_node_power["current_node_power"]
        self.interface.amdsmi_get_npm_info = lambda _h: npm_info_without_current_node_power

        _, commands = _run_node(self.node_module, output_format="json", destination="file")

        self.assertEqual(
            commands.logger.output["node"]["power_management"]["current_node_power"], "N/A"
        )

    def test_current_node_power_na_sentinel_value(self):
        # amdsmi_get_npm_info()'s Python wrapper already converts the
        # UINT32_MAX sentinel to the string "N/A" before the dict reaches the
        # CLI; confirm that value passes through unchanged rather than being
        # formatted as "N/A W".
        npm_info_na = dict(_DEFAULT_NPM_INFO)
        npm_info_na["current_node_power"] = "N/A"
        self.interface.amdsmi_get_npm_info = lambda _h: npm_info_na

        _, commands = _run_node(self.node_module, output_format="json", destination="file")

        self.assertEqual(
            commands.logger.output["node"]["power_management"]["current_node_power"], "N/A"
        )

    def test_current_node_power_defaults_to_na_on_library_exception(self):
        # When amdsmi_get_npm_info() itself raises, the whole npm_dict
        # (seeded with "N/A" defaults, current_node_power included) is left
        # untouched -- no partial/crashed state.
        def _raise(_h):
            raise _FakeLibraryException("npm info unavailable")

        self.interface.amdsmi_get_npm_info = _raise

        _, commands = _run_node(self.node_module, output_format="json", destination="file")

        power_management = commands.logger.output["node"]["power_management"]
        self.assertEqual(power_management["current_node_power"], "N/A")
        self.assertEqual(power_management["limit"], "N/A")
