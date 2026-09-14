#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``CURRENT_NODE_POWER`` field in ``amd-smi node -p``.

current_node_power moved from amdsmi_power_info_t (per-GPU, amd-smi metric
--power) to amdsmi_npm_info_t (per-node, amd-smi node -p /
amdsmi_get_npm_info()) -- see the "Set npm power limit" design doc. This file
replaces the previous test_metric_power_node_power.py, which exercised the
now-removed amd-smi metric --power NODE_POWER surface.

``NodeCommands.node()`` in ``amdsmi_cli/subcommands/node.py`` now seeds
``npm_dict["current_node_power"] = "N/A"`` and, on a successful
``amdsmi_get_npm_info()`` call, overwrites it with
``npm_info.get("current_node_power", "N/A")`` -- directly mirroring the
pre-existing ``threshold``/``limit`` fields it sits next to. These tests
drive the real ``NodeCommands.node()`` dispatch (stubbing only the
underlying ``amdsmi`` package, per the
``test_vcn_busy_navi.py``/``test_cli_set_clk_limit.py`` stub-and-importlib
pattern) and assert ``current_node_power`` is present with the correct
format/value across all output modes:

* human-readable: unit-suffixed string printed as ``CURRENT_NODE_POWER: 5800 W``.
* JSON/file output: ``{"value": 5800, "unit": "W"}`` (unit_format's json
  branch), nested under ``power_management``.
* CSV: the raw value (unit_format is not applied on the CSV path).

Also covers the "N/A" sentinel pass-through (key absent from the library's
dict, or explicitly UINT32_MAX-turned-"N/A" upstream in
``amdsmi_get_npm_info()``'s Python wrapper) and the
``AmdSmiLibraryException`` fallback path (whole npm_dict stays at its "N/A"
defaults, current_node_power included).
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "node.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
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


def _install_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    wrapper.AMDSMI_NPM_STATUS_DISABLED = 1
    interface.amdsmi_wrapper = wrapper
    interface.amdsmi_get_npm_info = lambda _h: dict(_DEFAULT_NPM_INFO)

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    amdsmi_pkg.amdsmi_wrapper = wrapper

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


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


def _run_node(node_module, output_format="human", destination="stdout"):
    helpers = _FakeHelpers()
    logger = _FakeLogger(output_format=output_format, destination=destination)
    commands = object.__new__(node_module.NodeCommands)
    commands.logger = logger
    commands.helpers = helpers
    commands.group_check_printed = True
    commands.node_handle = object()

    commands.node(_build_node_args())
    return logger, commands


class TestNodePowerManagementCurrentNodePower(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
    )

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(NODE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI node.py not found at {NODE_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

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
        # CSV rows always lead with a device identifier, mirroring gpu/cpu/nic.
        self.assertEqual(commands.logger.output["node"], 0)

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


if __name__ == "__main__":
    unittest.main()
