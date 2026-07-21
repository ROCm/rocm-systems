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

"""Mock-based unit tests for the ``amd-smi metric --partition`` clock logic.

These tests drive ``MetricCommands.metric_gpu`` with the C library, logger, and
helpers fully stubbed, so they run without GPU hardware or the compiled
``amdsmi`` package. They exercise the partition-scoped clock assembly path that
builds the per-AID (``aid_<N>``) and per-XCP (``xcp_<N>``) clock entries, and
lock the following behaviors in place:

* AID iteration spans every VCLK position, so a hole in the middle of the array
  does not drop trailing valid AIDs.
* An XCP that reports no GFX clock produces no ``xcp_<N>`` entry (no limits or
  lock state attached to a missing value).
* GFX lock state is ``N/A`` when the lock-status word is absent, ``DISABLED``
  when the word is present and the bit is clear (0 is a valid reading).

A second class covers the virtual-OS case: the parser omits ``--partition`` on
non-baremetal platforms, so ``args`` has no ``partition`` attribute and
``metric_gpu`` must not read it.
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path

# The amd-smi CLI ships alongside the amdsmi package: ``common`` resolves
# ``amdsmi_path`` to ``<rocm>/share/amd_smi`` and the CLI installs to the sibling
# ``<rocm>/libexec/amdsmi_cli``. ``setUpClass`` skips the suite if it is absent
# (e.g. an unusual layout where only the package, not the CLI, is present).
_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
METRIC_PATH = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli", "subcommands", "metric.py")


class _FakeClkType:
    """Stand-in for ``amdsmi_interface.AmdSmiClkType`` enum members."""

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
    raise _FakeLibraryException("mock error")


_UNSET = object()


def _restore_attr(interface, name, original):
    if original is _UNSET:
        delattr(interface, name)
    else:
        setattr(interface, name, original)


def _patch_interface(testcase, interface, **overrides):
    """Set interface attributes for one test and restore them on cleanup."""
    for name, value in overrides.items():
        original = getattr(interface, name, _UNSET)
        setattr(interface, name, value)
        testcase.addCleanup(_restore_attr, interface, name, original)


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``metric.py`` imports cleanly.

    Returns the fake ``amdsmi_interface`` module so individual tests can swap in
    per-case return values for the C-library entry points.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_NUM_GFX_CLKS = 8
    interface.AMDSMI_MAX_NUM_CLKS = 4
    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AmdSmiClkType = _FakeClkType

    # Default clock-limit payload reused by every AmdSmiClkType lookup.
    def _get_clock_info(_handle, _clk_type):
        return {"min_clk": 400, "max_clk": 2100, "clk_deep_sleep": "DISABLED"}

    interface.amdsmi_get_clock_info = _get_clock_info
    interface.amdsmi_get_gpu_metrics_info = lambda _handle: {}
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: {}
    # Set per-test; default keeps the partition path inert.
    interface.amdsmi_get_gpu_partition_metrics_info = lambda _handle: None

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    return interface


def _load_metric_module():
    spec = importlib.util.spec_from_file_location("metric_under_test", METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload that ``metric_gpu`` stores per GPU."""

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
        return {"num_partition": 1}

    def unit_format(self, logger, value, unit):
        # Mirror the human-readable branch of the real helper: "N/A" passes
        # through, everything else becomes "<value> <unit>".
        if isinstance(value, list):
            return [self.unit_format(logger, v, unit) for v in value]
        if value == "N/A":
            return "N/A"
        if unit:
            return f"{value} {unit}".rstrip()
        return f"{value}".rstrip()


class _FakeHelpersVirtualOS(_FakeHelpers):
    """Virtual-OS stub: ``is_baremetal()`` is False, so ``--partition`` is unregistered."""

    def is_baremetal(self):
        return False

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {"num_partition": "N/A"}


def _build_args(**overrides):
    """Namespace with every attribute ``metric_gpu`` touches, clock+partition on."""
    defaults = dict(
        gpu=object(),  # non-None, non-list placeholder device handle
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        partition=True,
        clock=True,
        usage=False,
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


def _build_virtual_os_args(**overrides):
    """Namespace mirroring argparse on a Linux virtual OS (no baremetal-only flags)."""
    defaults = dict(
        gpu=object(),  # non-None, non-list placeholder device handle
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        clock=True,
        usage=False,
        power=False,
        temperature=False,
        voltage=False,
        pcie=False,
        ecc=False,
        ecc_blocks=False,
        base_board=False,
        gpu_board=False,
        mem_usage=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliMetricPartitionClock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(METRIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI metric.py not found at {METRIC_PATH}")
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()

    def _run_clock_partition(self, partition_metrics):
        """Drive ``metric_gpu`` for ``--clock --partition`` and return ``clocks``."""
        self.interface.amdsmi_get_gpu_partition_metrics_info = lambda _handle: partition_metrics

        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        commands.device_handles = []

        commands.metric_gpu(_build_args())

        captured = commands.logger.captured_values
        self.assertIsNotNone(captured, "metric_gpu did not store a values payload")
        self.assertIn("clock", captured)
        return captured["clock"]

    def test_sparse_vclk_keeps_trailing_aids(self):
        # Hole at index 0 and 2; a trailing valid AID sits at index 3. Counting
        # only non-"N/A" entries (the old behavior) would size the loop to 2 and
        # drop aid_3 entirely.
        partition_metrics = {
            "current_gfxclks": "N/A",
            "current_vclk0s": ["N/A", 900, "N/A", 850],
            "current_dclk0s": ["N/A", 800, "N/A", 750],
            "current_socclks": ["N/A", 700, "N/A", 650],
        }

        clocks = self._run_clock_partition(partition_metrics)

        self.assertNotIn("aid_0", clocks)
        self.assertIn("aid_1", clocks)
        self.assertIn("aid_3", clocks)
        self.assertEqual(clocks["aid_1"]["vclk"], "900 MHz")
        self.assertEqual(clocks["aid_3"]["vclk"], "850 MHz")
        self.assertEqual(clocks["aid_3"]["dclk"], "750 MHz")
        self.assertEqual(clocks["aid_3"]["socclk"], "650 MHz")

    def test_xcp_without_gfx_clk_is_omitted(self):
        # Middle XCP reports no GFX clock: it must not appear with phantom
        # limits/lock state derived from a missing value.
        partition_metrics = {
            "current_gfxclks": [1500, "N/A", 1400],
            "gfxclk_lock_status": 0,
            "current_vclk0s": "N/A",
        }

        clocks = self._run_clock_partition(partition_metrics)

        self.assertIn("xcp_0", clocks)
        self.assertNotIn("xcp_1", clocks)
        self.assertIn("xcp_2", clocks)
        self.assertEqual(clocks["xcp_0"]["gfx_clk"], "1500 MHz")
        self.assertEqual(clocks["xcp_2"]["gfx_clk"], "1400 MHz")
        # The present XCPs still carry limits since they have a real value.
        self.assertEqual(clocks["xcp_0"]["gfx_min_clk"], "400 MHz")
        self.assertEqual(clocks["xcp_0"]["gfx_max_clk"], "2100 MHz")

    def test_xcp_lock_unknown_is_na(self):
        # No "gfxclk_lock_status" key: lock state is unknown, not DISABLED.
        partition_metrics = {"current_gfxclks": [1500], "current_vclk0s": "N/A"}

        clocks = self._run_clock_partition(partition_metrics)

        self.assertIn("xcp_0", clocks)
        self.assertEqual(clocks["xcp_0"]["gfx_clk_locked"], "N/A")
        # Top-level gfx skeleton entry is likewise left unknown.
        self.assertEqual(clocks["gfx_0"]["clk_locked"], "N/A")

    def test_lock_status_zero_is_disabled(self):
        # A lock-status word of 0 is a valid "all unlocked" reading and must
        # resolve to DISABLED, not be skipped as if it were missing.
        partition_metrics = {
            "current_gfxclks": [1500],
            "gfxclk_lock_status": 0,
            "current_vclk0s": "N/A",
        }

        clocks = self._run_clock_partition(partition_metrics)

        self.assertEqual(clocks["xcp_0"]["gfx_clk_locked"], "DISABLED")
        self.assertEqual(clocks["gfx_0"]["clk_locked"], "DISABLED")

    def test_lock_status_bitmask_per_domain(self):
        # Bit 1 set -> gfx_1 / xcp_1 locked, the others unlocked.
        partition_metrics = {
            "current_gfxclks": [1500, 1450, 1400],
            "gfxclk_lock_status": 0b010,
            "current_vclk0s": "N/A",
        }

        clocks = self._run_clock_partition(partition_metrics)

        self.assertEqual(clocks["gfx_0"]["clk_locked"], "DISABLED")
        self.assertEqual(clocks["gfx_1"]["clk_locked"], "ENABLED")
        self.assertEqual(clocks["gfx_2"]["clk_locked"], "DISABLED")
        self.assertEqual(clocks["xcp_0"]["gfx_clk_locked"], "DISABLED")
        self.assertEqual(clocks["xcp_1"]["gfx_clk_locked"], "ENABLED")
        self.assertEqual(clocks["xcp_2"]["gfx_clk_locked"], "DISABLED")

    def test_happy_path_full_partition_metrics(self):
        # Dense arrays: every AID and XCP entry is present and formatted.
        partition_metrics = {
            "current_gfxclks": [1500, 1450],
            "gfxclk_lock_status": 0,
            "current_vclk0s": [900, 950],
            "current_dclk0s": [800, 850],
            "current_socclks": [700, 750],
            "current_socclks_mid": [600, 650],
        }

        clocks = self._run_clock_partition(partition_metrics)

        self.assertEqual(clocks["aid_0"]["vclk"], "900 MHz")
        self.assertEqual(clocks["aid_1"]["vclk"], "950 MHz")
        self.assertEqual(clocks["aid_0"]["dclk"], "800 MHz")
        self.assertEqual(clocks["aid_0"]["socclk"], "700 MHz")
        self.assertEqual(clocks["xcp_0"]["gfx_clk"], "1500 MHz")
        self.assertEqual(clocks["xcp_1"]["gfx_clk"], "1450 MHz")
        self.assertEqual(clocks["socclks_mid"]["mid_0"], "600 MHz")
        self.assertEqual(clocks["socclks_mid"]["mid_1"], "650 MHz")

    def _run_baremetal_section(self, args):
        """Drive ``metric_gpu`` on baremetal and return the stored values dict."""
        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        commands.device_handles = []
        commands.metric_gpu(args)
        return commands.logger.captured_values

    def test_usage_partition_branch_uses_partition_metrics(self):
        # partition=True drives the fetch; gpu_partition_metrics is non-None, so
        # the simplified usage condition must still route to partition-scoped
        # xcp data. The socket-level else branch is covered by tests that leave
        # amdsmi_get_gpu_partition_metrics_info returning None.
        _patch_interface(
            self,
            self.interface,
            amdsmi_get_gpu_metrics_info=lambda _h: {"vcn_activity": "N/A", "jpeg_activity": "N/A"},
            amdsmi_get_gpu_activity=lambda _h: {"gfx_activity": 50},
            amdsmi_get_gpu_partition_metrics_info=lambda _h: {
                "xcp_stats.gfx_busy_inst": [[10, 20]]
            },
        )

        captured = self._run_baremetal_section(_build_args(clock=False, usage=True, partition=True))

        self.assertIn("usage", captured)
        self.assertIsInstance(captured["usage"], dict)
        self.assertIn("xcp_0", captured["usage"]["gfx_busy_inst"])

    def test_temperature_partition_branch_uses_partition_metrics(self):
        # Same invariant for the temperature section: partition metrics present
        # drive mid/xcd temperatures without reading args.partition.
        _patch_interface(
            self,
            self.interface,
            amdsmi_get_gpu_metrics_info=lambda _h: {},
            amdsmi_get_temp_metric=lambda *_: 50,
            AmdSmiTemperatureType=types.SimpleNamespace(
                EDGE="EDGE", HOTSPOT="HOTSPOT", VRAM="VRAM"
            ),
            AmdSmiTemperatureMetric=types.SimpleNamespace(CURRENT="CURRENT", CRITICAL="CRITICAL"),
            amdsmi_get_gpu_partition_metrics_info=lambda _h: {
                "temperature_mid": [40, 41],
                "xcp_stats.temperature_xcd": [55, 56],
            },
        )

        captured = self._run_baremetal_section(
            _build_args(clock=False, temperature=True, partition=True)
        )

        self.assertIn("temperature", captured)
        self.assertIn("xcp_0", captured["temperature"]["xcd"])


class TestCliMetricPartitionVirtualOS(unittest.TestCase):
    """``amd-smi metric`` must not crash on a virtual OS where ``--partition``
    is never registered, so ``args`` has no ``partition`` attribute."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(METRIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI metric.py not found at {METRIC_PATH}")
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()
        # fclk is unavailable so the non-partition clock exception handler fires.
        cls.interface.amdsmi_get_clk_freq = _raise_lib_exc
        # Temperature enum access is not guarded, so the enums must exist; the
        # sensor fetches degrade to N/A.
        cls.interface.amdsmi_get_gpu_activity = _raise_lib_exc
        cls.interface.amdsmi_get_temp_metric = _raise_lib_exc
        cls.interface.AmdSmiTemperatureType = types.SimpleNamespace(
            EDGE="EDGE", HOTSPOT="HOTSPOT", VRAM="VRAM"
        )
        cls.interface.AmdSmiTemperatureMetric = types.SimpleNamespace(
            CURRENT="CURRENT", CRITICAL="CRITICAL"
        )

    def _run_metric(self, args):
        partition_calls = []

        def _tracking_partition_metrics(handle):
            partition_calls.append(handle)
            return None

        self.interface.amdsmi_get_gpu_partition_metrics_info = _tracking_partition_metrics

        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpersVirtualOS()
        commands.group_check_printed = True
        commands.device_handles = []

        commands.metric_gpu(args)
        return commands.logger.captured_values, partition_calls

    def test_metric_without_partition_attr_does_not_crash(self):
        # Reproduces the AttributeError: on a virtual OS the parser omits
        # --partition, so args has no 'partition'. metric_gpu must not read it.
        args = _build_virtual_os_args()
        self.assertFalse(hasattr(args, "partition"))

        captured, partition_calls = self._run_metric(args)

        self.assertIsNotNone(captured, "metric_gpu did not store a values payload")
        self.assertIn("clock", captured)
        self.assertIsInstance(captured["clock"], dict)
        # partition is not a valid platform arg here, so the partition-metrics
        # API must never be probed.
        self.assertEqual(partition_calls, [])

    def test_usage_and_temperature_sections_do_not_read_partition(self):
        # The usage and temperature partition branches were simplified to key off
        # gpu_partition_metrics (None here). Let the usage section run to
        # completion (real activity + metrics data) so a re-introduced
        # args.partition read would surface instead of being swallowed by the
        # section's broad except; both sections must produce a dict payload and
        # never probe the partition-metrics API on a virtual OS.
        _patch_interface(
            self,
            self.interface,
            amdsmi_get_gpu_activity=lambda _h: {"gfx_activity": 50},
            amdsmi_get_gpu_metrics_info=lambda _h: {"vcn_activity": "N/A", "jpeg_activity": "N/A"},
        )
        args = _build_virtual_os_args(clock=False, usage=True, temperature=True)
        self.assertFalse(hasattr(args, "partition"))

        captured, partition_calls = self._run_metric(args)

        self.assertIsNotNone(captured, "metric_gpu did not store a values payload")
        self.assertIsInstance(captured["usage"], dict)
        self.assertIn("temperature", captured)
        self.assertEqual(partition_calls, [])


if __name__ == "__main__":
    unittest.main()
