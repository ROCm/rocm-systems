#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi static --bus`` PCIe sentinel handling.

WSL2 reports ``max_pcie_speed``/``pcie_interface_version`` as the string
``"N/A"`` sentinel (rather than a numeric max-uint sentinel) when the PCIe
static info is unavailable. ``StaticCommands.static_gpu`` used to do
``bus_info["max_pcie_speed"] % 1000``, which raised
``TypeError: not all arguments converted during string formatting`` because
Python treats ``%`` on a ``str`` as string formatting, not modulo. These tests
drive ``static_gpu`` with the C library, logger, and helpers stubbed so they
run without GPU hardware, and lock in that the bus/PCIe section no longer
crashes and reports ``N/A`` in both human-readable and JSON output.
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
STATIC_PATH = os.path.join(_REPO_ROOT, "amdsmi_cli", "subcommands", "static.py")


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


def _install_fake_modules(pcie_static):
    """Register a stub ``amdsmi`` package plus the sibling CLI modules.

    ``pcie_static`` is returned verbatim from ``amdsmi_get_pcie_info``.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"
    interface.amdsmi_get_pcie_info = lambda _handle: {"pcie_static": pcie_static}
    exception.AmdSmiLibraryException = _FakeLibraryException

    def _raise_lib_exc(_handle):
        raise exception.AmdSmiLibraryException("mock: pci bandwidth unavailable")

    # Bus info also fetches pci bandwidth for pcie_levels; keep it out of scope
    # for these PCIe-speed/version-focused tests by having it degrade to N/A
    # the same way the real library does when unsupported.
    interface.amdsmi_get_gpu_pci_bandwidth = _raise_lib_exc

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception

    # ``static.py`` imports these sibling names at load time; the bus path
    # never instantiates them (the test injects a fake helpers object).
    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = object
    sys.modules["amdsmi_helpers"] = helpers_mod

    exceptions_mod = types.ModuleType("amdsmi_cli_exceptions")
    exceptions_mod.AmdSmiInvalidParameterException = type(
        "AmdSmiInvalidParameterException", (Exception,), {}
    )
    sys.modules["amdsmi_cli_exceptions"] = exceptions_mod

    return interface


def _load_static_module():
    spec = importlib.util.spec_from_file_location("static_under_test", STATIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload ``static_gpu`` stores per GPU."""

    def __init__(self, fmt):
        self._fmt = fmt
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self._fmt == "json"

    def is_csv_format(self):
        return self._fmt == "csv"

    def is_human_readable_format(self):
        return self._fmt == "human"

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass


class _FakeHelpers:
    """Minimal helpers stub for the single-GPU, baremetal-off bus path.

    ``unit_format`` mirrors the real ``AMDSMIHelpers.unit_format`` (see
    ``amdsmi_cli/amdsmi_helpers.py``) closely enough to exercise the same
    N/A-passthrough and per-format branches ``static_gpu`` relies on.
    """

    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def check_required_groups(self):
        pass

    def is_linux(self):
        return True

    def is_baremetal(self):
        return False

    def is_virtual_os(self):
        return True

    def is_hypervisor(self):
        return False

    def unit_format(self, logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(logger, val, unit) for val in value]
        if value == "N/A":
            return "N/A"
        if logger.is_json_format():
            return {"value": value, "unit": unit} if unit else value
        if logger.is_csv_format():
            return value
        if logger.is_human_readable_format():
            return f"{value} {unit}".rstrip() if unit else f"{value}".rstrip()
        return f"{value}"


def _build_args():
    """Namespace with bus on and every other static section off."""
    return argparse.Namespace(
        gpu=object(),
        asic=False,
        bus=True,
        vbios=False,
        driver=False,
        ras=False,
        vram=False,
        cache=False,
        board=False,
        process_isolation=False,
        clock=False,
        mem_carveout=False,
        partition=False,
    )


class TestCliStaticBusPcieNA(unittest.TestCase):
    """``max_pcie_speed``/``pcie_interface_version`` as the string ``"N/A"``
    sentinel (the WSL2 case) must not crash ``static_gpu`` and must render as
    ``N/A`` rather than a bogus formatted value."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI static.py not found at {STATIC_PATH}")
        cls.interface = _install_fake_modules(
            {
                "max_pcie_width": "N/A",
                "max_pcie_speed": "N/A",
                "pcie_interface_version": "N/A",
                "slot_type": "N/A",
            }
        )
        cls.static_module = _load_static_module()

    def _run_bus(self, fmt):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True

        commands.static_gpu(_build_args())

        if fmt == "json":
            self.assertTrue(commands.logger.store_gpu_json_output)
            static_dict = commands.logger.store_gpu_json_output[-1]
        else:
            static_dict = commands.logger.captured_values
        self.assertIsNotNone(static_dict, "static_gpu stored no values payload")
        self.assertIn("bus", static_dict)
        return static_dict["bus"]

    def test_human_readable_na_does_not_crash(self):
        # Reproduces the WSL2 crash: max_pcie_speed == "N/A" used to hit
        # "N/A" % 1000, raising TypeError. Must resolve to plain "N/A".
        bus_info = self._run_bus("human")
        self.assertEqual(bus_info["max_pcie_speed"], "N/A")
        self.assertEqual(bus_info["pcie_interface_version"], "N/A")

    def test_json_na_does_not_crash(self):
        bus_info = self._run_bus("json")
        self.assertEqual(bus_info["max_pcie_speed"], "N/A")
        self.assertEqual(bus_info["pcie_interface_version"], "N/A")


class TestCliStaticBusPcieValid(unittest.TestCase):
    """Sanity check: a normal numeric PCIe reading still formats correctly
    after the ``unit_format`` refactor."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI static.py not found at {STATIC_PATH}")
        cls.interface = _install_fake_modules(
            {
                "max_pcie_width": 16,
                "max_pcie_speed": 16000,
                "pcie_interface_version": 4,
                "slot_type": "OAM",
            }
        )
        cls.static_module = _load_static_module()

    def _run_bus(self, fmt):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True

        commands.static_gpu(_build_args())

        if fmt == "json":
            self.assertTrue(commands.logger.store_gpu_json_output)
            static_dict = commands.logger.store_gpu_json_output[-1]
        else:
            static_dict = commands.logger.captured_values
        self.assertIsNotNone(static_dict, "static_gpu stored no values payload")
        return static_dict["bus"]

    def test_human_readable_formats_speed_and_gen(self):
        bus_info = self._run_bus("human")
        self.assertEqual(bus_info["max_pcie_speed"], "16 GT/s")
        self.assertEqual(bus_info["pcie_interface_version"], "Gen 4")

    def test_json_formats_speed_as_value_unit_dict(self):
        bus_info = self._run_bus("json")
        self.assertEqual(bus_info["max_pcie_speed"], {"value": 16, "unit": "GT/s"})
        self.assertEqual(bus_info["pcie_interface_version"], "Gen 4")
