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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_MODULE_PATH = os.path.join(_REPO_ROOT, "amdsmi_cli", "amdsmi_init.py")


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
    def _run_init(self, gpu, hsmp, nic_present):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        with (
            mock.patch.object(module, "check_amdgpu_driver", return_value=gpu),
            mock.patch.object(module, "check_amd_hsmp_driver", return_value=hsmp),
            mock.patch.object(module, "_any_nic_present", return_value=nic_present),
        ):
            return module.amdsmi_cli_init()

    def test_nic_only_host_initializes(self):
        flag = self._run_init(gpu=False, hsmp=False, nic_present=True)
        self.assertTrue(flag & 4)  # INIT_AMD_NICS requested
        self.assertFalse(flag & 1)  # no GPU
        self.assertFalse(flag & 2)  # no CPU

    def test_empty_host_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_init(gpu=False, hsmp=False, nic_present=False)
        self.assertEqual(ctx.exception.code, -1)

    def test_gpu_host_initializes_without_nic(self):
        flag = self._run_init(gpu=True, hsmp=False, nic_present=False)
        self.assertTrue(flag & 1)  # GPU
        self.assertTrue(flag & 4)  # NIC still always requested

    def test_gpu_host_skips_nic_probe(self):
        module = _load_amdsmi_init(_make_amdsmi_stub())
        with (
            mock.patch.object(module, "check_amdgpu_driver", return_value=True),
            mock.patch.object(module, "check_amd_hsmp_driver", return_value=False),
            mock.patch.object(module, "_any_nic_present") as nic_probe,
        ):
            module.amdsmi_cli_init()
        # GPU present short-circuits the gate; the NIC probe is not needed.
        nic_probe.assert_not_called()


class TestAnyNicPresentHelper(unittest.TestCase):
    """Exercises the real _any_nic_present detector (not mocked).

    The gate tests above stub _any_nic_present, so without this the detector's
    own contract -- iterate sockets, read the processor_handles key, swallow a
    discovery exception -- would be untested, and a handle-shape change could
    regress the NIC-only gate while every test stayed green.
    """

    def _load(self):
        return _load_amdsmi_init(_make_amdsmi_stub())

    def test_true_when_a_socket_has_nic_handles(self):
        module = self._load()
        with (
            mock.patch.object(
                module.amdsmi_interface, "amdsmi_get_socket_handles", return_value=[object()]
            ),
            mock.patch.object(
                module.amdsmi_interface,
                "amdsmi_get_processor_handles_by_type",
                return_value={"processor_handles": [object()]},
            ),
        ):
            self.assertTrue(module._any_nic_present())

    def test_false_when_no_socket_has_nic_handles(self):
        module = self._load()
        with (
            mock.patch.object(
                module.amdsmi_interface, "amdsmi_get_socket_handles", return_value=[object()]
            ),
            mock.patch.object(
                module.amdsmi_interface,
                "amdsmi_get_processor_handles_by_type",
                return_value={"processor_handles": []},
            ),
        ):
            self.assertFalse(module._any_nic_present())

    def test_false_when_discovery_raises(self):
        module = self._load()
        with mock.patch.object(
            module.amdsmi_interface,
            "amdsmi_get_socket_handles",
            side_effect=module.amdsmi_exception.AmdSmiException("discovery failed"),
        ):
            self.assertFalse(module._any_nic_present())


if __name__ == "__main__":
    unittest.main()
