#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""list_ainic root-hint tests.

PCI VPD is mode 0600, so an unprivileged run renders PRODUCT_NAME, PART_NUMBER
and SERIAL_NUMBER as "N/A" -- indistinguishable from a card that genuinely has
no VPD. list_ainic must emit one note explaining that, and must stay silent when
the fields populated or the caller is already root.

Loaded in isolation with a stubbed ``amdsmi`` so the branch is exercised
without a built libamd_smi.so or NIC hardware.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_MODULE_PATH = os.path.join(_REPO_ROOT, "amdsmi_cli", "subcommands", "list_devices.py")


def _load_list_devices():
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")
    amdsmi_stub.amdsmi_interface = types.ModuleType("amdsmi.amdsmi_interface")
    with mock.patch.dict(sys.modules, {"amdsmi": amdsmi_stub}):
        spec = importlib.util.spec_from_file_location("list_devices_hint_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _summary(product, part, serial):
    return {
        "bdf": "0000:a1:00.0",
        "UUID": "N/A",
        "Permanent Address": "N/A",
        "Product Name": product,
        "Part Number": part,
        "Serial Number": serial,
        "Vendor Name": "AMD Pensando Systems, Inc.",
        "Capability": ["FWCTL"],
    }


class TestListNicVpdHint(unittest.TestCase):
    def _run_list_ainic(self, summary, euid, human=True):
        module = _load_list_devices()
        handle = types.SimpleNamespace(value=1)

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.handle_ainics.return_value = (False, handle)
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = False
        command.logger.is_human_readable_format.return_value = human

        module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(return_value=summary)

        args = types.SimpleNamespace(nic=handle)
        captured = io.StringIO()
        with mock.patch("os.geteuid", return_value=euid):
            with contextlib.redirect_stderr(captured):
                command.list_ainic(args, nic=handle)
        return captured.getvalue()

    def test_unprivileged_na_identity_emits_hint(self):
        err = self._run_list_ainic(_summary("N/A", "N/A", "N/A"), euid=1000)
        self.assertIn("root", err)
        self.assertIn("VPD", err)

    def test_root_run_stays_silent(self):
        # As root an "N/A" is the card's real answer, so the hint would mislead.
        err = self._run_list_ainic(_summary("N/A", "N/A", "N/A"), euid=0)
        self.assertEqual(err, "")

    def test_populated_identity_stays_silent(self):
        err = self._run_list_ainic(_summary("Salina", "DSC3", "FPK26"), euid=1000)
        self.assertEqual(err, "")

    def test_partial_identity_emits_hint(self):
        # One suppressed field is enough; VPD is read as a whole.
        err = self._run_list_ainic(_summary("Salina", "DSC3", "N/A"), euid=1000)
        self.assertIn("root", err)

    def test_library_failure_stays_silent(self):
        # The failure path fills every identity field with "N/A", which is not
        # evidence about VPD permissions.
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

        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = False
        command.logger.is_human_readable_format.return_value = True

        module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(side_effect=_LibError())

        args = types.SimpleNamespace(nic=handle)
        captured = io.StringIO()
        with mock.patch("os.geteuid", return_value=1000):
            with contextlib.redirect_stderr(captured):
                command.list_ainic(args, nic=handle)
        self.assertEqual(captured.getvalue(), "")

    def test_multiple_nics_emit_one_note(self):
        # Every suppressed row sets the pending flag, but the note is per
        # invocation: two "N/A" NICs must not produce two notes.
        module = _load_list_devices()
        handles = [types.SimpleNamespace(value=1), types.SimpleNamespace(value=2)]

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = False
        command.logger.is_human_readable_format.return_value = True

        # The real helper recurses into list_ainic once per NIC, then reports
        # that it handled them; each inner row returns before printing.
        outer_call_count = 0

        def _handle_ainics(_args, _logger, callback):
            nonlocal outer_call_count
            outer_call_count += 1
            if outer_call_count > 1:
                return (False, _args.nic)
            for nic_handle in handles:
                callback(
                    types.SimpleNamespace(nic=nic_handle), multiple_devices=True, nic=nic_handle
                )
            return (True, None)

        command.helpers.handle_ainics.side_effect = _handle_ainics

        module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(
            return_value=_summary("N/A", "N/A", "N/A")
        )

        args = types.SimpleNamespace(nic=handles[0])
        captured = io.StringIO()
        with mock.patch("os.geteuid", return_value=1000):
            with contextlib.redirect_stderr(captured):
                command.list_ainic(args, nic=handles[0])
        self.assertEqual(captured.getvalue().count(module._VPD_ROOT_HINT), 1)

    def test_hint_does_not_leak_into_next_invocation(self):
        # The pending flag lives on the command object, so a suppressed run must
        # clear it or the next run reports a VPD problem it never saw.
        module = _load_list_devices()
        handle = types.SimpleNamespace(value=1)

        command = module.ListDevicesCommands.__new__(module.ListDevicesCommands)
        command.group_check_printed = True
        command.helpers = mock.Mock()
        command.helpers.handle_ainics.return_value = (False, handle)
        command.helpers.get_ainic_id_from_device_handle.return_value = 0

        command.logger = mock.Mock()
        command.logger.is_csv_format.return_value = False
        command.logger.is_human_readable_format.return_value = True

        args = types.SimpleNamespace(nic=handle)
        with mock.patch("os.geteuid", return_value=1000):
            module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(
                return_value=_summary("N/A", "N/A", "N/A")
            )
            with contextlib.redirect_stderr(io.StringIO()) as first:
                command.list_ainic(args, nic=handle)

            module.amdsmi_interface.amdsmi_get_ainic_info = mock.Mock(
                return_value=_summary("Salina", "DSC3", "FPK26")
            )
            with contextlib.redirect_stderr(io.StringIO()) as second:
                command.list_ainic(args, nic=handle)

        self.assertIn("VPD", first.getvalue())
        self.assertEqual(second.getvalue(), "")

    def test_machine_readable_output_stays_silent(self):
        # A note on stderr is harmless, but JSON/CSV consumers get no prose.
        err = self._run_list_ainic(_summary("N/A", "N/A", "N/A"), euid=1000, human=False)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
