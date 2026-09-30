#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""firmware --nic prefetch: multiple NICs are queried concurrently in a single
prefetch pass, and the per-device path consumes the prefetched result instead
of re-querying. The single-NIC path is untouched (no prefetch, no behavior
change).

Loads subcommands/firmware.py standalone with a stubbed ``amdsmi`` package so
the control flow is exercised without a built libamd_smi.so or NIC hardware.
"""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

_CLI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "amdsmi_cli")
)
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "firmware.py")


def _load_firmware():
    """Import subcommands/firmware.py standalone with a stubbed amdsmi.

    Loading by file path skips subcommands/__init__.py (which pulls in every
    sibling command); stubbing amdsmi removes the built-library dependency.
    """
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")

    class _AmdSmiLibraryException(Exception):
        def get_error_info(self):
            return "stub error"

    amdsmi_stub.amdsmi_exception.AmdSmiLibraryException = _AmdSmiLibraryException

    interface_stub = types.ModuleType("amdsmi.amdsmi_interface")
    amdsmi_stub.amdsmi_interface = interface_stub

    with mock.patch.dict(
        sys.modules,
        {
            "amdsmi": amdsmi_stub,
            "amdsmi.amdsmi_exception": amdsmi_stub.amdsmi_exception,
            "amdsmi.amdsmi_interface": interface_stub,
        },
    ):
        spec = importlib.util.spec_from_file_location("firmware_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, amdsmi_stub


def _handle_ainics(args, logger, subcommand):
    """Faithful stand-in for AMDSMIHelpers.handle_ainics's dispatch shape."""
    if isinstance(args.nic, list):
        if len(args.nic) > 1:
            for handle in args.nic:
                subcommand(args, multiple_devices=True, nic=handle)
            logger.print_output(multiple_device_enabled=True)
            return True, args.nic
        elif len(args.nic) == 1:
            return False, args.nic[0]
        return True, args.nic
    return False, args.nic


class TestFirmwareNicPrefetch(unittest.TestCase):
    def _make_command(self, module, device_handles):
        class _Command(module.FirmwareCommands):
            def __init__(self):
                self.device_handles_ainics = device_handles
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.side_effect = lambda h: (
                    device_handles.index(h)
                )
                self.helpers.handle_ainics.side_effect = _handle_ainics
                self.logger = mock.Mock()

        return _Command()

    def test_multiple_nics_query_each_exactly_once(self):
        module, amdsmi_stub = _load_firmware()
        handles = [mock.Mock(value=i) for i in range(3)]
        command = self._make_command(module, handles)

        get_fw_info = mock.Mock(side_effect=lambda h: {"FW": f"v{h.value}"})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_fw_info = get_fw_info

        args = types.SimpleNamespace(nic=None, fw_list=None)
        command.firmware_nic(args)

        self.assertEqual(get_fw_info.call_count, len(handles))
        stored = [c.args[2] for c in command.logger.store_ainic_output.call_args_list]
        self.assertEqual(stored, [{"FW": "v0"}, {"FW": "v1"}, {"FW": "v2"}])

    def test_single_nic_has_no_prefetch(self):
        module, amdsmi_stub = _load_firmware()
        handle = mock.Mock(value=0)
        command = self._make_command(module, [handle])

        get_fw_info = mock.Mock(return_value={"FW": "v0"})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_fw_info = get_fw_info

        args = types.SimpleNamespace(nic=handle, fw_list=None)
        command.firmware_nic(args)

        get_fw_info.assert_called_once_with(handle)
        self.assertFalse(hasattr(args, "_nic_fw_info_prefetch"))

    def test_prefetch_failure_reports_empty_dict(self):
        module, amdsmi_stub = _load_firmware()
        handles = [mock.Mock(value=i) for i in range(2)]
        command = self._make_command(module, handles)

        def _raise_on_first(h):
            if h.value == 0:
                raise amdsmi_stub.amdsmi_exception.AmdSmiLibraryException()
            return {"FW": "v1"}

        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_fw_info = mock.Mock(side_effect=_raise_on_first)

        args = types.SimpleNamespace(nic=None, fw_list=None)
        command.firmware_nic(args)

        stored = [c.args[2] for c in command.logger.store_ainic_output.call_args_list]
        self.assertEqual(stored, [{}, {"FW": "v1"}])


if __name__ == "__main__":
    unittest.main()
