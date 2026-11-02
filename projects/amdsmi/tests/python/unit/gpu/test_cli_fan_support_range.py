#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the range ``amd-smi set --fan --help`` advertises.

``AMDSMIHelpers.get_fan_support()`` must advertise the range
``amdsmi_set_gpu_fan_speed()`` validates against: the gpu_od ``OD_RANGE`` where
the GPU exposes it, otherwise the hwmon 0-255. Since ROCm/rocm-systems#8684
``amdsmi_get_gpu_fan_speed_max()`` reports pwm1_max (255) on gpu_od GPUs too, so
the advertised range cannot be derived from it.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))

_STATUS_NOT_SUPPORTED = 2
_HWMON_PWM_MAX = 255

_FAN_MINIMUM_PWM = "FAN_MINIMUM_PWM:\n0\nOD_RANGE:\nMINIMUM_PWM: 15 100\n"


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED):
        super().__init__("mock error")
        self.err_code = err_code


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


def _fake_amdsmi():
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED,
        AMDSMI_STATUS_NO_PERM=10,
        AMDSMI_STATUS_INVAL=5,
        AMDSMI_STATUS_UNEXPECTED_SIZE=42,
        AMDSMI_STATUS_UNEXPECTED_DATA=43,
        AMDSMI_NPM_STATUS_DISABLED=0,
        AMDSMI_NPM_STATUS_ENABLED=1,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        AmdSmiInitFlags=_FakeInitFlags,
        AmdSmiLibraryException=_FakeLibraryException,
        AmdSmiParameterException=type("_FakeParameterException", (Exception,), {}),
        amdsmi_init=lambda _flag: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: ["gpu0"],
        amdsmi_get_gpu_fan_speed=lambda _handle, _sensor: 51,
        amdsmi_get_gpu_fan_speed_max=lambda _handle, _sensor: _HWMON_PWM_MAX,
        amdsmi_get_gpu_device_bdf=lambda _handle: "0000:03:00.0",
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
        "amdsmi_init": None,
        "amdsmi_helpers": None,
        "amdsmi_cli_exceptions": None,
        "BDF": None,
    }


class TestFanSupportRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        stub_modules(cls, _fake_amdsmi())

        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_helpers

        cls.helpers_cls = amdsmi_helpers.AMDSMIHelpers

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="amdsmi_fan_support_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _fan_support(self, gpu_od_path):
        fake_self = types.SimpleNamespace(
            detect_gpu_od=lambda _bdf: (gpu_od_path is not None, gpu_od_path),
            parse_gpu_od_fan_range=self.helpers_cls.parse_gpu_od_fan_range,
        )
        return self.helpers_cls.get_fan_support(fake_self)

    def test_gpu_od_advertises_od_range(self):
        gpu_od = os.path.join(self.tmp, "gpu_od")
        os.makedirs(os.path.join(gpu_od, "fan_ctrl"))
        with open(os.path.join(gpu_od, "fan_ctrl", "fan_minimum_pwm"), "w") as f:
            f.write(_FAN_MINIMUM_PWM)

        self.assertEqual(self._fan_support(gpu_od), "15-100 or 0-100%%")

    def test_gpu_od_without_fan_ctrl_falls_back_to_hwmon_range(self):
        gpu_od = os.path.join(self.tmp, "gpu_od")
        os.makedirs(gpu_od)

        self.assertEqual(self._fan_support(gpu_od), "0-255 or 0-100%%")

    def test_legacy_hwmon_advertises_pwm_range(self):
        self.assertEqual(self._fan_support(None), "0-255 or 0-100%%")
