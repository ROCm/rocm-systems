#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the vcn_busy sysfs fallback on non-XCP devices.

On Navi-class GPUs (no XCP partitions, num_partition == "N/A",
gpu_partition_metrics is None), ``metric_gpu --usage`` now calls
``amdsmi_get_vcn_busy_percent`` to populate ``engine_usage["vcn_busy"]``
instead of leaving it at the "N/A" default.

These tests exercise:

* Successful read: the returned integer is stored as-is.
* Library exception: ``vcn_busy`` degrades to "N/A" without propagating.
* Partition path not taken: the new sysfs call is not issued when
  ``gpu_partition_metrics`` is present (XCP devices must not fall through).
* Socket-level-metrics path not taken: the sysfs call is not issued when
  ``num_partition != "N/A"`` (older metric-version path with XCP data).
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

from common.common import ModuleIsolationMixin, amdsmi_path

_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
METRIC_PATH = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli", "subcommands", "metric.py")


class _FakeClkType:
    GFX = "GFX"
    MEM = "MEM"
    VCLK0 = "VCLK0"
    DCLK0 = "DCLK0"
    SOC = "SOC"
    DF = "DF"


class _FakeLibraryException(Exception):
    def __init__(self, message="mock error"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


def _raise_lib_exc(*_args, **_kwargs):
    raise _FakeLibraryException("vcn_busy sysfs unavailable")


_UNSET = object()


def _restore_attr(obj, name, original):
    if original is _UNSET:
        delattr(obj, name)
    else:
        setattr(obj, name, original)


def _patch(testcase, obj, **overrides):
    for name, value in overrides.items():
        original = getattr(obj, name, _UNSET)
        setattr(obj, name, value)
        testcase.addCleanup(_restore_attr, obj, name, original)


def _install_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_NUM_GFX_CLKS = 8
    interface.AMDSMI_MAX_NUM_CLKS = 4
    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AmdSmiClkType = _FakeClkType

    interface.amdsmi_get_clock_info = lambda _h, _t: {
        "min_clk": 400,
        "max_clk": 2100,
        "clk_deep_sleep": "DISABLED",
    }
    interface.amdsmi_get_gpu_metrics_info = lambda _h: {
        "vcn_activity": "N/A",
        "jpeg_activity": "N/A",
    }
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: {}
    interface.amdsmi_get_gpu_partition_metrics_info = lambda _h: None
    interface.amdsmi_get_gpu_activity = lambda _h: {"gfx_activity": 30}
    interface.amdsmi_get_vcn_busy_percent = lambda _h: 42

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    return interface


# Everything ``metric.py`` imports at load time.
_STUBBED_MODULES = ("amdsmi", "amdsmi.amdsmi_interface", "amdsmi.amdsmi_exception")


def _load_metric_module():
    spec = importlib.util.spec_from_file_location("metric_under_test_vcn", METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def __init__(self):
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False

    def is_human_readable_format(self):
        return True

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass

    def store_watch_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def __init__(self, num_partition="N/A"):
        self._num_partition = num_partition

    def is_hypervisor(self):
        return False

    def is_windows(self):
        return False

    def is_baremetal(self):
        return True

    def is_linux(self):
        return True

    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {"num_partition": self._num_partition}

    def unit_format(self, logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(logger, v, unit) for v in value]
        if value == "N/A":
            return "N/A"
        if unit:
            return f"{value} {unit}".rstrip()
        return f"{value}".rstrip()


def _build_usage_args(**overrides):
    defaults = dict(
        gpu=object(),
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        partition=False,
        clock=False,
        usage=True,
        power=False,
        temperature=False,
        voltage=False,
        pcie=False,
        ecc=False,
        ecc_blocks=False,
        base_board=False,
        gpu_board=False,
        mem_usage=False,
        fan=False,
        voltage_curve=False,
        overdrive=False,
        perf_level=False,
        xgmi_err=False,
        energy=False,
        throttle=False,
        violation=False,
        schedule=False,
        guard=False,
        guest_data=False,
        fb_usage=False,
        xgmi=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_usage(metric_module, interface, helpers=None, args=None):
    if helpers is None:
        helpers = _FakeHelpers(num_partition="N/A")
    if args is None:
        args = _build_usage_args()

    commands = object.__new__(metric_module.MetricCommands)
    commands.logger = _FakeLogger()
    commands.helpers = helpers
    commands.group_check_printed = True
    commands.device_handles = []

    commands.metric_gpu(args)
    return commands.logger.captured_values


class TestVcnBusyNaviFallback(ModuleIsolationMixin, unittest.TestCase):
    """The new sysfs fallback for vcn_busy on non-XCP devices."""

    ISOLATED_MODULES = _STUBBED_MODULES

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(METRIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI metric.py not found at {METRIC_PATH}")
        super().setUpClass()
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()

    def test_navi_vcn_busy_reads_sysfs(self):
        # gpu_partition_metrics is None and num_partition is "N/A": the new
        # sysfs getter is called and its value stored in usage["vcn_busy"].
        _patch(self, self.interface, amdsmi_get_vcn_busy_percent=lambda _h: 55)

        captured = _run_usage(self.metric_module, self.interface)

        self.assertIsNotNone(captured)
        self.assertIn("usage", captured)
        self.assertEqual(captured["usage"]["vcn_busy"], "55 %")

    def test_navi_vcn_busy_zero_is_valid(self):
        # 0% is a real reading; it must not be treated as N/A.
        _patch(self, self.interface, amdsmi_get_vcn_busy_percent=lambda _h: 0)

        captured = _run_usage(self.metric_module, self.interface)

        self.assertEqual(captured["usage"]["vcn_busy"], "0 %")

    def test_navi_vcn_busy_library_exception_degrades_to_na(self):
        # When the sysfs read raises AmdSmiLibraryException, vcn_busy must
        # fall back to "N/A" and not propagate the exception.
        _patch(self, self.interface, amdsmi_get_vcn_busy_percent=_raise_lib_exc)

        captured = _run_usage(self.metric_module, self.interface)

        self.assertIsNotNone(captured)
        self.assertEqual(captured["usage"]["vcn_busy"], "N/A")

    def test_xcp_partition_metrics_path_skips_sysfs(self):
        # When gpu_partition_metrics is non-None (XCP device), the sysfs getter
        # must not be called; vcn_busy comes from partition metrics instead.
        calls = []

        def _spy(_h):
            calls.append(_h)
            return 99

        _patch(
            self,
            self.interface,
            amdsmi_get_vcn_busy_percent=_spy,
            amdsmi_get_gpu_partition_metrics_info=lambda _h: {
                "xcp_stats.gfx_busy_inst": [],
                "xcp_stats.jpeg_busy": [],
                "xcp_stats.vcn_busy": [70, 80],
            },
        )

        captured = _run_usage(
            self.metric_module, self.interface, args=_build_usage_args(partition=True)
        )

        self.assertEqual(calls, [], "sysfs getter must not be called on XCP devices")
        vcn = captured["usage"]["vcn_busy"]
        self.assertIsInstance(vcn, dict)
        self.assertIn("xcp_0", vcn)

    def test_socket_level_metrics_path_skips_sysfs(self):
        # When num_partition != "N/A" (socket-level XCP path), vcn_busy is
        # populated from gpu_metric data and the sysfs getter is not called.
        calls = []

        def _spy(_h):
            calls.append(_h)
            return 99

        _patch(
            self,
            self.interface,
            amdsmi_get_vcn_busy_percent=_spy,
            amdsmi_get_gpu_metrics_info=lambda _h: {
                "vcn_activity": "N/A",
                "jpeg_activity": "N/A",
                "xcp_stats.gfx_busy_inst": [10, 20],
                "xcp_stats.jpeg_busy": [5, 6],
                "xcp_stats.vcn_busy": [30, 40],
            },
        )

        # num_partition=2 triggers the socket-level elif branch
        helpers = _FakeHelpers(num_partition=2)
        captured = _run_usage(self.metric_module, self.interface, helpers=helpers)

        self.assertEqual(calls, [], "sysfs getter must not be called on socket-level XCP path")
        vcn = captured["usage"]["vcn_busy"]
        self.assertIsInstance(vcn, dict)
        self.assertIn("xcp_0", vcn)
        self.assertIn("xcp_1", vcn)


if __name__ == "__main__":
    unittest.main()
