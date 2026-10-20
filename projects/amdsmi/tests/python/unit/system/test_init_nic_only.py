#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CLI init gating tests for GPU/CPU-less hosts.

The old init counted a live NIC driver toward init success, so a NIC-only host
(no amdgpu/amd_hsmp) came up. The vendor-registry rewrite dropped the per-vendor
NIC probe and must not regress that: a host with a discovered NIC but no GPU/CPU
must still initialize, while a host with nothing to manage must still exit.

Loaded with a stubbed ``amdsmi`` so the gate is exercised without a built
libamd_smi.so or any hardware.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

from common.common import amdsmi_path, cli_search_order, find_cli_dir

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_CLI_DIR, "amdsmi_init.py") if _CLI_DIR else None

if not _MODULE_PATH or not os.path.isfile(_MODULE_PATH):
    raise unittest.SkipTest(
        f"amd-smi CLI amdsmi_init.py not found (looked in {_CLI_DIR or amdsmi_path})"
    )


def _make_amdsmi_stub():
    class AmdSmiException(Exception):
        pass

    class AmdSmiLibraryException(AmdSmiException):
        def __init__(self, err_code=0):
            self.err_code = err_code

    class AmdSmiParameterException(AmdSmiException):
        pass

    exc = types.ModuleType("amdsmi.amdsmi_exception")
    exc.AmdSmiException = AmdSmiException
    exc.AmdSmiLibraryException = AmdSmiLibraryException
    exc.AmdSmiParameterException = AmdSmiParameterException

    iface = types.ModuleType("amdsmi.amdsmi_interface")
    iface.AmdSmiInitFlags = types.SimpleNamespace(
        INIT_ALL_PROCESSORS=0xFFFFFFFF, INIT_AMD_GPUS=1, INIT_AMD_CPUS=2, INIT_AMD_NICS=4
    )
    iface.AmdSmiProcessorType = types.SimpleNamespace(AMD_AINIC=object())
    iface.AmdSmiLibraryException = AmdSmiLibraryException
    iface.AmdSmiParameterException = AmdSmiParameterException
    iface.amdsmi_wrapper = types.SimpleNamespace(
        AMDSMI_STATUS_NOT_INIT=1, AMDSMI_STATUS_DRIVER_NOT_LOADED=2
    )
    iface.amdsmi_init = mock.Mock()
    iface.amdsmi_shut_down = mock.Mock()  # registered via atexit at import
    iface.amdsmi_get_socket_handles = mock.Mock(return_value=[object()])
    iface.amdsmi_get_processor_handles_by_type = mock.Mock(return_value={"processor_handles": []})

    amdsmi = types.ModuleType("amdsmi")
    amdsmi.amdsmi_exception = exc
    amdsmi.amdsmi_interface = iface
    return amdsmi


def _load_amdsmi_init(amdsmi_stub):
    patched = {
        "amdsmi": amdsmi_stub,
        "amdsmi.amdsmi_exception": amdsmi_stub.amdsmi_exception,
        "amdsmi.amdsmi_interface": amdsmi_stub.amdsmi_interface,
    }
    with mock.patch.dict(sys.modules, patched):
        spec = importlib.util.spec_from_file_location("amdsmi_init_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        try:
            # The module runs amdsmi_cli_init() at import; on a driverless test
            # host that legitimately exits, which we swallow to reach the funcs.
            spec.loader.exec_module(module)
        except SystemExit:
            pass
    return module


class TestInitNicOnlyGating(unittest.TestCase):
    def _patch_gates(self, module, gpu, hsmp, nic_present):
        for name, value in (
            ("check_amdgpu_driver", gpu),
            ("check_amd_hsmp_driver", hsmp),
            ("_any_nic_present", nic_present),
        ):
            patcher = mock.patch.object(module, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run_init(self, gpu, hsmp, nic_present):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        self._patch_gates(module, gpu, hsmp, nic_present)
        return module.amdsmi_cli_init()

    def test_nic_only_host_initializes(self):
        flag = self._run_init(gpu=False, hsmp=False, nic_present=True)
        self.assertTrue(flag & 4)  # INIT_AMD_NICS requested
        self.assertFalse(flag & 1)  # no GPU
        self.assertFalse(flag & 2)  # no CPU

    def test_empty_host_exits_with_drivers_not_loaded(self):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        self._patch_gates(module, gpu=False, hsmp=False, nic_present=False)
        with self.assertLogs(level="ERROR") as logs, self.assertRaises(SystemExit) as ctx:
            module.amdsmi_cli_init()
        # -1 would reach the shell as 255, the library's unknown-error code.
        self.assertEqual(ctx.exception.code, int(module.AmdSmiExitCode.DRIVERS_NOT_LOADED))
        self.assertIn("Drivers not loaded", logs.output[0])

    def test_gpu_host_initializes_without_nic(self):
        flag = self._run_init(gpu=True, hsmp=False, nic_present=False)
        self.assertTrue(flag & 1)  # GPU
        self.assertTrue(flag & 4)  # NIC still always requested

    def test_gpu_host_skips_nic_probe(self):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        self._patch_gates(module, gpu=True, hsmp=False, nic_present=False)
        module.amdsmi_cli_init()
        # GPU present short-circuits the gate; the NIC probe is not needed.
        module._any_nic_present.assert_not_called()

    def _init_raising(self, err_code):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        self._patch_gates(module, gpu=True, hsmp=False, nic_present=True)
        lib_exc = module.amdsmi_interface.AmdSmiLibraryException(err_code)
        patcher = mock.patch.object(module.amdsmi_interface, "amdsmi_init", side_effect=lib_exc)
        patcher.start()
        self.addCleanup(patcher.stop)
        return module

    def test_library_not_init_exits_with_drivers_not_loaded(self):
        module = self._init_raising(1)  # AMDSMI_STATUS_NOT_INIT in the stub
        with self.assertLogs(level="ERROR") as logs, self.assertRaises(SystemExit) as ctx:
            module.amdsmi_cli_init()
        self.assertEqual(ctx.exception.code, int(module.AmdSmiExitCode.DRIVERS_NOT_LOADED))
        self.assertIn("Drivers not loaded", logs.output[0])

    def test_library_driver_not_loaded_exits_with_drivers_not_loaded(self):
        module = self._init_raising(2)  # AMDSMI_STATUS_DRIVER_NOT_LOADED in the stub
        with self.assertLogs(level="ERROR") as logs, self.assertRaises(SystemExit) as ctx:
            module.amdsmi_cli_init()
        self.assertEqual(ctx.exception.code, int(module.AmdSmiExitCode.DRIVERS_NOT_LOADED))
        self.assertIn("Drivers not loaded", logs.output[0])

    def test_other_library_failure_is_raised_not_translated(self):
        module = self._init_raising(99)
        with self.assertRaises(module.amdsmi_interface.AmdSmiLibraryException) as ctx:
            module.amdsmi_cli_init()
        self.assertEqual(ctx.exception.err_code, 99)


class TestAnyNicPresentHelper(unittest.TestCase):
    """Exercises the real _any_nic_present detector (not mocked).

    The gate tests above stub _any_nic_present, so without this the detector's
    own contract -- iterate sockets, read the processor_handles key, swallow a
    discovery exception -- would be untested, and a handle-shape change could
    regress the NIC-only gate while every test stayed green.
    """

    def _load(self):
        return _load_amdsmi_init(_make_amdsmi_stub())

    def _patch_one_socket(self, module, processor_handles):
        for name, value in (
            ("amdsmi_get_socket_handles", [object()]),
            ("amdsmi_get_processor_handles_by_type", {"processor_handles": processor_handles}),
        ):
            patcher = mock.patch.object(module.amdsmi_interface, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_true_when_a_socket_has_nic_handles(self):
        module = self._load()
        self._patch_one_socket(module, [object()])
        self.assertTrue(module._any_nic_present())

    def test_false_when_no_socket_has_nic_handles(self):
        module = self._load()
        self._patch_one_socket(module, [])
        self.assertFalse(module._any_nic_present())

    def test_false_when_discovery_raises(self):
        module = self._load()
        with mock.patch.object(
            module.amdsmi_interface,
            "amdsmi_get_socket_handles",
            side_effect=module.amdsmi_exception.AmdSmiException("discovery failed"),
        ):
            self.assertFalse(module._any_nic_present())
