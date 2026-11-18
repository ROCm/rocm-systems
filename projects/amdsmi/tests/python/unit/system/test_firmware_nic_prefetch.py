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

import ast
import importlib.util
import logging
import os
import sys
import threading
import types
import unittest
from unittest import mock

from common.common import amdsmi_path, cli_search_order, find_cli_dir

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "firmware.py") if _CLI_DIR else None
_HELPERS_PATH = os.path.join(_CLI_DIR, "amdsmi_helpers.py") if _CLI_DIR else None

_NIC_PROCESSOR_TYPE = 7
_NIC_TYPE_NAME = "AMD_NIC"
_OVERLAP_TIMEOUT_SECONDS = 2

if not _MODULE_PATH or not os.path.isfile(_MODULE_PATH):
    raise unittest.SkipTest(
        f"amd-smi CLI firmware.py not found (looked in {_CLI_DIR or amdsmi_path})"
    )


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


def _real_handle_ainics():
    """``AMDSMIHelpers.handle_ainics`` compiled from the shipped source.

    Importing amdsmi_helpers needs the whole CLI import chain, so the one method is extracted and
    bound to a stand-in ``self``. Only ``run_device_subcommand`` and the processor-type lookup are
    faked.
    """
    with open(_HELPERS_PATH, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    helpers_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AMDSMIHelpers"
    )
    method = next(
        node
        for node in helpers_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "handle_ainics"
    )

    interface = types.SimpleNamespace(
        amdsmi_get_processor_type=lambda handle: {"processor_type": _NIC_TYPE_NAME},
        AmdSmiProcessorType=lambda value: types.SimpleNamespace(name=_NIC_TYPE_NAME),
        amdsmi_wrapper=types.SimpleNamespace(AMDSMI_PROCESSOR_TYPE_AMD_NIC=_NIC_PROCESSOR_TYPE),
    )
    namespace = {"amdsmi_interface": interface, "logging": logging}
    exec(compile(ast.Module([method], []), _HELPERS_PATH, "exec"), namespace)

    helpers = types.SimpleNamespace(
        run_device_subcommand=lambda subcommand, args, **device_kwarg: subcommand(
            args, multiple_devices=True, **device_kwarg
        )
    )
    return types.MethodType(namespace["handle_ainics"], helpers)


class TestFirmwareNicPrefetch(unittest.TestCase):
    def _make_command(self, module, device_handles):
        class _Command(module.FirmwareCommands):
            def __init__(self):
                self.device_handles_ainics = device_handles
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.side_effect = lambda h: (
                    device_handles.index(h)
                )
                self.helpers.handle_ainics.side_effect = _real_handle_ainics()
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

    def test_prefetch_queries_the_nics_concurrently(self):
        module, amdsmi_stub = _load_firmware()
        handles = [mock.Mock(value=i) for i in range(2)]
        command = self._make_command(module, handles)
        both_queries_running = threading.Barrier(2, timeout=_OVERLAP_TIMEOUT_SECONDS)

        def _query_after_the_other_starts(handle):
            both_queries_running.wait()
            return {"FW": f"v{handle.value}"}

        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_fw_info = mock.Mock(
            side_effect=_query_after_the_other_starts
        )

        args = types.SimpleNamespace(nic=None, fw_list=None)
        command.firmware_nic(args)

        stored = [c.args[2] for c in command.logger.store_ainic_output.call_args_list]
        self.assertEqual(stored, [{"FW": "v0"}, {"FW": "v1"}])

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

    def test_prefetch_pool_is_capped(self):
        module, amdsmi_stub = _load_firmware()
        handles = [mock.Mock(value=i) for i in range(12)]
        command = self._make_command(module, handles)
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_fw_info = mock.Mock(return_value={})
        real_pool = module.concurrent.futures.ThreadPoolExecutor
        requested = []

        def _recording_pool(max_workers):
            requested.append(max_workers)
            return real_pool(max_workers=max_workers)

        args = types.SimpleNamespace(nic=None, fw_list=None)
        with mock.patch.object(module.concurrent.futures, "ThreadPoolExecutor", _recording_pool):
            command.firmware_nic(args)

        self.assertEqual(len(requested), 1)
        self.assertLessEqual(requested[0], 8)
