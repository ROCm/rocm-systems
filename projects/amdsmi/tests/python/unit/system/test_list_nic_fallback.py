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
"""list_devices routing tests: an explicit --nic that matches nothing must not
silently list GPUs.

These load the CLI module in isolation with a stubbed ``amdsmi`` package so the
control-flow branch is exercised without a built libamd_smi.so or NIC hardware;
the NIC-less branch never fires on a GPU/NIC-equipped CI host.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

_CLI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "amdsmi_cli")
)
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "list_devices.py")


def _load_list_devices():
    """Import subcommands/list_devices.py standalone with a stubbed amdsmi.

    Loading by file path skips subcommands/__init__.py (which pulls in every
    sibling command); stubbing amdsmi removes the built-library dependency.
    """
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")
    amdsmi_stub.amdsmi_interface = types.ModuleType("amdsmi.amdsmi_interface")
    with mock.patch.dict(sys.modules, {"amdsmi": amdsmi_stub}):
        spec = importlib.util.spec_from_file_location("list_devices_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestListNicFallback(unittest.TestCase):
    def _make_command(self):
        module = _load_list_devices()

        class _Command(module.ListDevicesCommands):
            def __init__(self):
                # Two GPUs present, no NICs -> any --nic query matches nothing.
                self.device_handles_gpus = [object(), object()]
                self.device_handles_ainics = []
                self.helpers = mock.Mock()
                self.helpers.is_ainic_initialized.return_value = True
                self.logger = mock.Mock()
                self.logger.is_human_readable_format.return_value = True
                self.list_gpu = mock.Mock()
                self.list_ainic = mock.Mock()

        return _Command()

    def test_explicit_nic_no_match_does_not_list_gpus(self):
        command = self._make_command()
        # Explicit --nic with a handle that is absent from device_handles_ainics.
        args = types.SimpleNamespace(gpu=None, nic=[types.SimpleNamespace(value=99)])

        command.list_devices(args)

        command.list_gpu.assert_not_called()
        command.list_ainic.assert_not_called()

    def test_bare_list_still_lists_gpus(self):
        command = self._make_command()
        # Bare `amd-smi list`: no --nic given, GPUs must still be listed.
        args = types.SimpleNamespace(gpu=None, nic=None)

        command.list_devices(args)

        command.list_gpu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
