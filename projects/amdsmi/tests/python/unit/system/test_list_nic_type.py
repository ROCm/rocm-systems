#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""list --nic shows the NIC type right after the BDF.

The subcommand module is loaded in isolation with a stubbed ``amdsmi``, so the rendering is
exercised without a built libamd_smi.so or NIC hardware.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

from common.common import cli_search_order, find_cli_dir

# cli_search_order() decides whether the install or this checkout wins; None means no CLI found.
_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))
_MODULE_PATH = Path(_CLI_DIR) / "subcommands" / "list_devices.py" if _CLI_DIR else None

if _MODULE_PATH is None or not _MODULE_PATH.is_file():
    raise unittest.SkipTest(f"amd-smi CLI list_devices.py not found (looked in {_CLI_DIR})")


def _load_list_devices() -> types.ModuleType:
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")
    amdsmi_stub.amdsmi_interface = types.ModuleType("amdsmi.amdsmi_interface")
    with mock.patch.dict(sys.modules, {"amdsmi": amdsmi_stub}):
        spec = importlib.util.spec_from_file_location("list_devices_type_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _summary(nic_type: str, capability: List[str]) -> Dict[str, object]:
    return {
        "bdf": "0000:a1:00.0",
        "Type": nic_type,
        "UUID": "0a:1b:2c:3d:4e:5f",
        "Permanent Address": "0a:1b:2c:3d:4e:5f",
        "Product Name": "DSC3-200",
        "Part Number": "A1B2C3",
        "Serial Number": "SN12345",
        "Vendor Name": "AMD Pensando Systems, Inc.",
        "Capability": capability,
    }


class _LibraryError(Exception):
    def get_error_info(self) -> str:
        return "boom"


class TestListNicType(unittest.TestCase):
    def _run_list_ainic(self, info: Optional[Dict[str, object]], csv: bool) -> Dict[str, object]:
        """Run list_ainic and return {argument: data} in the order it was stored."""
        module = _load_list_devices()
        module.amdsmi_exception.AmdSmiLibraryException = _LibraryError
        handle = types.SimpleNamespace(value=1)

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.handle_ainics.return_value = (False, handle)
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        stored: Dict[str, object] = {}
        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = csv
        command.logger.store_ainic_output.side_effect = lambda _handle, argument, data: (
            stored.update({argument: data})
        )
        if info is None:
            module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(side_effect=_LibraryError())
        else:
            module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(return_value=info)

        command.list_ainic(types.SimpleNamespace(nic=handle), nic=handle)
        return stored

    def test_type_is_stored_in_human_output(self) -> None:
        stored = self._run_list_ainic(_summary("UALoE", ["FWCTL"]), csv=False)
        self.assertEqual(stored["type"], "UALoE")

    def test_type_is_stored_in_csv_output(self) -> None:
        stored = self._run_list_ainic(_summary("AINIC", ["FWCTL", "NETDEV"]), csv=True)
        self.assertEqual(stored["type"], "AINIC")

    def test_type_follows_the_bdf_in_human_output(self) -> None:
        stored = self._run_list_ainic(_summary("AINIC", ["FWCTL"]), csv=False)
        self.assertEqual(list(stored)[:2], ["bdf", "type"])

    def test_type_follows_the_bdf_in_csv_output(self) -> None:
        stored = self._run_list_ainic(_summary("AINIC", ["FWCTL"]), csv=True)
        self.assertEqual(list(stored)[:2], ["nic_bdf", "type"])

    def test_another_vendor_shows_the_label_it_was_given(self) -> None:
        stored = self._run_list_ainic(_summary("Broadcom Inc.", ["NETDEV"]), csv=False)
        self.assertEqual(stored["type"], "Broadcom Inc.")

    def test_library_failure_renders_na_type(self) -> None:
        stored = self._run_list_ainic(None, csv=False)
        self.assertEqual(stored["type"], "N/A")
