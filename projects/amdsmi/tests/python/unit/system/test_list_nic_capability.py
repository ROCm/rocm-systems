#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""list_ainic capability/mode rendering tests.

A fwctl-only NIC has no host netdev, so its permanent address (MAC) is
meaningless. list_ainic must surface a synthesized MODE, expose the decoded
CAPABILITY bits, and omit the permanent-address row for such cards in the
human-readable output (while keeping the column, blanked, in CSV for alignment).

Loaded in isolation with a stubbed ``amdsmi`` so the branch is exercised
without a built libamd_smi.so or NIC hardware.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from typing import List
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_CLI_DIR = os.path.join(_REPO_ROOT, "amdsmi_cli")
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "list_devices.py")
_INTERFACE_PATH = os.path.join(_REPO_ROOT, "py-interface", "amdsmi_interface.py")


def _load_list_devices():
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")
    amdsmi_stub.amdsmi_interface = types.ModuleType("amdsmi.amdsmi_interface")
    with mock.patch.dict(sys.modules, {"amdsmi": amdsmi_stub}):
        spec = importlib.util.spec_from_file_location("list_devices_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _load_decode_nic_capabilities():
    """Compile just ``_decode_nic_capabilities`` from the shipped interface source.

    Importing the whole module needs a built libamd_smi.so; extracting the one
    function lets us exercise the real bit-to-name decode against a fake wrapper.
    """
    with open(_INTERFACE_PATH, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_nic_capabilities":
            wrapper = types.SimpleNamespace(
                amdsmi_nic_capability_bits_t__enumvalues={
                    1: "AMDSMI_NIC_CAP_FWCTL",
                    2: "AMDSMI_NIC_CAP_NETDEV",
                }
            )
            namespace = {"amdsmi_wrapper": wrapper, "List": List}
            exec(compile(ast.Module([node], []), _INTERFACE_PATH, "exec"), namespace)
            return namespace["_decode_nic_capabilities"]
    raise AssertionError("_decode_nic_capabilities not found in amdsmi_interface.py")


def _summary(capability):
    return {
        "bdf": "0000:a1:00.0",
        "UUID": "0a:1b:2c:3d:4e:5f",
        "Permanent Address": "0a:1b:2c:3d:4e:5f",
        "Product Name": "DSC3-200",
        "Part Number": "A1B2C3",
        "Serial Number": "SN12345",
        "Vendor Name": "AMD Pensando Systems, Inc.",
        "Capability": capability,
    }


class _StoreCapture:
    """Captures store_ainic_output(handle, argument, data) as {argument: data}."""

    def __init__(self):
        self.stored = {}

    def __call__(self, _handle, argument, data):
        self.stored[argument] = data


class TestListNicCapability(unittest.TestCase):
    def _run_list_ainic(self, capability, csv=False):
        module = _load_list_devices()
        handle = types.SimpleNamespace(value=1)

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.handle_ainics.return_value = (False, handle)
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        capture = _StoreCapture()
        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = csv
        command.logger.store_ainic_output.side_effect = capture

        module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(return_value=_summary(capability))

        args = types.SimpleNamespace(nic=handle)
        command.list_ainic(args, nic=handle)
        return capture.stored

    def test_fwctl_only_omits_permanent_address_human(self):
        stored = self._run_list_ainic(["FWCTL"], csv=False)
        self.assertNotIn("permanent_address", stored)
        self.assertEqual(stored["mode"], "fwctl-only")
        self.assertEqual(stored["capability"], ["FWCTL"])

    def test_netdev_keeps_permanent_address_human(self):
        stored = self._run_list_ainic(["FWCTL", "NETDEV"], csv=False)
        self.assertEqual(stored["permanent_address"], "0a:1b:2c:3d:4e:5f")
        self.assertEqual(stored["mode"], "netdev")
        self.assertEqual(stored["capability"], ["FWCTL", "NETDEV"])

    def test_fwctl_only_blanks_permanent_address_csv(self):
        stored = self._run_list_ainic(["FWCTL"], csv=True)
        # CSV keeps the column for alignment, blanked, and joins bits.
        self.assertEqual(stored["permanent_address"], "")
        self.assertEqual(stored["mode"], "fwctl-only")
        self.assertEqual(stored["capability"], "FWCTL")

    def test_netdev_csv_joins_capability(self):
        stored = self._run_list_ainic(["FWCTL", "NETDEV"], csv=True)
        self.assertEqual(stored["permanent_address"], "0a:1b:2c:3d:4e:5f")
        self.assertEqual(stored["mode"], "netdev")
        self.assertEqual(stored["capability"], "FWCTL;NETDEV")

    def test_no_capability_bits_reports_unknown_human(self):
        stored = self._run_list_ainic([], csv=False)
        self.assertNotIn("permanent_address", stored)
        self.assertEqual(stored["mode"], "unknown")
        self.assertEqual(stored["capability"], [])

    def test_no_capability_bits_reports_unknown_csv(self):
        stored = self._run_list_ainic([], csv=True)
        self.assertEqual(stored["permanent_address"], "")
        self.assertEqual(stored["mode"], "unknown")
        self.assertEqual(stored["capability"], "")

    def test_library_failure_renders_na_row(self):
        module = _load_list_devices()
        handle = types.SimpleNamespace(value=1)

        class _LibError(Exception):
            def get_error_info(self):
                return "boom"

        module.amdsmi_exception.AmdSmiLibraryException = _LibError

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.handle_ainics.return_value = (False, handle)
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        capture = _StoreCapture()
        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = False
        command.logger.store_ainic_output.side_effect = capture

        module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(side_effect=_LibError())

        args = types.SimpleNamespace(nic=handle)
        # Must not raise UnboundLocalError; the row degrades to N/A.
        command.list_ainic(args, nic=handle)

        self.assertEqual(capture.stored["bdf"], "N/A")
        self.assertEqual(capture.stored["vendor_name"], "N/A")
        self.assertEqual(capture.stored["mode"], "unknown")
        self.assertNotIn("permanent_address", capture.stored)


class TestDecodeNicCapabilities(unittest.TestCase):
    """Exercises the bit-to-name decode used to build the CAPABILITY field."""

    @classmethod
    def setUpClass(cls):
        cls.decode = staticmethod(_load_decode_nic_capabilities())

    def test_single_bit_strips_prefix(self):
        self.assertEqual(self.decode(1), ["FWCTL"])

    def test_multiple_bits_expand_in_enum_order(self):
        self.assertEqual(self.decode(3), ["FWCTL", "NETDEV"])

    def test_zero_bitmask_is_empty(self):
        self.assertEqual(self.decode(0), [])

    def test_unknown_bit_is_ignored(self):
        # A bit with no enum entry must not appear in the decoded list.
        self.assertEqual(self.decode(1 | (1 << 5)), ["FWCTL"])


if __name__ == "__main__":
    unittest.main()
