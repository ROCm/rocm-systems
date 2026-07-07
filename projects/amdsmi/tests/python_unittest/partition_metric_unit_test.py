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
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_PATH = os.path.normpath(
    os.path.join(THIS_DIR, "..", "..", "amdsmi_cli", "subcommands", "metric.py")
)


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


def _build_args(**overrides):
    """Namespace with every attribute ``metric_gpu`` touches, clock+partition on."""
    defaults = dict(
        gpu=object(),  # non-None, non-list sentinel device handle
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


class PartitionClockMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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


if __name__ == "__main__":
    unittest.main()
