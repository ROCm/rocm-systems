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

"""Mock-based unit tests for the ``amd-smi monitor --decoder`` (DEC%) path.

These tests drive ``MonitorCommands.monitor`` with the C library, logger, and
helpers fully stubbed, so they run without GPU hardware or the compiled
``amdsmi`` package. They lock in how the decoder column sources its value on
MI3x ASICs, where only decoding (not encoding) is supported.

On an MI300X SR-IOV guest the per-instance ``vcn_activity`` array reads all
``"N/A"``, while the partition-scoped ``xcp_stats.vcn_busy[partition_id]`` still
carries real per-instance data (e.g. ``[0, 0, 0, 0]``). The monitor decoder
must fall back to that partition array so ``DEC%`` shows ``0`` (or the true
value), never ``N/A``, whenever partition data is present. The behaviors locked
in here:

* Guest fallback to zero: ``vcn_activity`` all ``"N/A"`` but ``vcn_busy[0]`` is
  ``[0, 0, 0, 0]`` yields ``DEC% == "0 %"`` (the reported defect), not ``N/A``.
* Guest fallback to a real average: ``vcn_busy[0] == [10, 20, 30, 40]`` yields
  ``DEC% == "25 %"``, proving real activity propagates through the fallback.
* Non-zero partition id: the fallback indexes the active partition row, not
  always row 0.
* Bare-metal direct path: a populated ``vcn_activity`` is used as-is without
  consulting the partition array.
* Genuine absence: both sources all ``"N/A"`` leaves ``DEC%`` as ``N/A`` (no
  fabricated zero).
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path


def _resolve_monitor_path():
    """Locate ``monitor.py``: prefer the in-tree source, fall back to install.

    Running from a checkout exercises the source under review; the installed
    ``<rocm>/libexec/amdsmi_cli`` copy is used when the tests run from the
    packaged ``amd-smi-lib-tests`` location.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    in_tree = os.path.normpath(
        os.path.join(here, "..", "..", "..", "..", "amdsmi_cli", "subcommands", "monitor.py")
    )
    rocm_root = os.path.dirname(os.path.dirname(amdsmi_path))
    installed = os.path.join(rocm_root, "libexec", "amdsmi_cli", "subcommands", "monitor.py")
    for path in (in_tree, installed):
        if os.path.isfile(path):
            return path
    return None


MONITOR_PATH = _resolve_monitor_path()


class _FakeLibraryException(Exception):
    def __init__(self, message="mock error"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


class _FakeSIUnit:
    MICRO = "micro"


class _FakeAMDSMIHelpers:
    """Minimal stand-in so ``from amdsmi_helpers import AMDSMIHelpers`` resolves.

    ``monitor.py`` only touches ``AMDSMIHelpers.SI_Unit.MICRO`` in the power-cap
    branch, which the decoder-only path never enters.
    """

    SI_Unit = _FakeSIUnit


def _install_fake_modules():
    """Register stub ``amdsmi`` and ``amdsmi_helpers`` modules for a clean import.

    Returns the fake ``amdsmi_interface`` module so each test can swap in
    per-case metric payloads.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.amdsmi_get_gpu_metrics_info = lambda _handle: {}
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: {}
    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = _FakeAMDSMIHelpers

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi_helpers"] = helpers_mod
    return interface


def _load_monitor_module():
    spec = importlib.util.spec_from_file_location("monitor_under_test", MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload that ``monitor`` stores per GPU."""

    def __init__(self):
        self.captured_values = None
        self.table_header = ""
        self.destination = "stdout"

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False

    def is_human_readable_format(self):
        return True

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def store_multiple_device_output(self):
        pass

    def store_watch_output(self, *args, **kwargs):
        pass

    def print_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    """Stub helpers exposing only what the decoder path reads."""

    def __init__(self, partition_id=0, num_partition=1):
        self._partition_id = partition_id
        self._num_partition = num_partition

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def is_virtual_os(self):
        return True

    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {
            "metric_version": "N/A",
            "partition_id": self._partition_id,
            "num_partition": self._num_partition,
            "num_xcp": self._num_partition,
        }

    @staticmethod
    def average_flattened_ints(data, context="data"):
        # Mirrors AMDSMIHelpers.average_flattened_ints: flatten one level, keep
        # ints, drop "N/A" strings, and average; empty -> "N/A".
        if not isinstance(data, (list, tuple)):
            return "N/A"
        flat = [
            v
            for value in data
            for v in (value if isinstance(value, list) else [value])
            if isinstance(v, int)
        ]
        return round(sum(flat) / len(flat)) if flat else "N/A"


def _build_args(**overrides):
    """Namespace with decoder selected and every other monitor section off."""
    defaults = dict(
        gpu=[object()],  # single-element handle list; monitor() unwraps it
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        default_output=False,
        power_usage=False,
        temperature=False,
        base_board_temps=False,
        gpu_board_temps=False,
        gfx=False,
        mem=False,
        encoder=False,
        decoder=True,
        ecc=False,
        vram_usage=False,
        pcie=False,
        violation=False,
        process=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliMonitorDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MONITOR_PATH is None:
            raise unittest.SkipTest("amd-smi CLI monitor.py not found (in-tree or installed)")
        cls.interface = _install_fake_modules()
        cls.monitor_module = _load_monitor_module()

    def _run_decoder(self, metrics, partition_id=0, num_partition=1):
        """Drive ``monitor`` for ``--decoder`` and return the DEC% value."""
        self.interface.amdsmi_get_gpu_metrics_info = lambda _handle: metrics

        commands = object.__new__(self.monitor_module.MonitorCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers(partition_id=partition_id, num_partition=num_partition)
        commands.group_check_printed = True
        commands.device_handles = []

        commands.monitor(_build_args())

        captured = commands.logger.captured_values
        self.assertIsNotNone(captured, "monitor did not store a values payload")
        self.assertIn("decoder", captured)
        return captured["decoder"]

    def test_guest_vcn_activity_na_falls_back_to_partition_zero(self):
        # SR-IOV guest: per-instance vcn_activity is all "N/A" but the active
        # partition row of xcp_stats.vcn_busy carries real zeros. DEC% must be
        # 0, not N/A -- the reported defect.
        metrics = {
            "vcn_activity": ["N/A", "N/A", "N/A", "N/A"],
            "xcp_stats.vcn_busy": [
                [0, 0, 0, 0],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
                ["N/A", "N/A", "N/A", "N/A"],
            ],
            "current_vclk0": 29,
            "current_dclk0": 22,
        }

        self.assertEqual(self._run_decoder(metrics), "0 %")

    def test_guest_fallback_reports_real_average(self):
        # Same fallback path, but the partition row holds real activity: the
        # averaged value must propagate, proving the column is not pinned to 0.
        metrics = {
            "vcn_activity": ["N/A", "N/A", "N/A", "N/A"],
            "xcp_stats.vcn_busy": [[10, 20, 30, 40]] + [["N/A"] * 4 for _ in range(7)],
            "current_vclk0": 29,
            "current_dclk0": 22,
        }

        self.assertEqual(self._run_decoder(metrics), "25 %")

    def test_fallback_uses_active_partition_row(self):
        # A GPU whose active partition is not 0 must read its own row; row 0 is
        # "N/A" here, so pinning to row 0 would regress to N/A.
        metrics = {
            "vcn_activity": ["N/A", "N/A", "N/A", "N/A"],
            "xcp_stats.vcn_busy": [["N/A", "N/A", "N/A", "N/A"], [4, 4, 4, 4]]
            + [["N/A"] * 4 for _ in range(6)],
            "current_vclk0": 29,
            "current_dclk0": 22,
        }

        self.assertEqual(self._run_decoder(metrics, partition_id=1), "4 %")

    def test_baremetal_direct_vcn_activity_used(self):
        # Bare metal: vcn_activity is populated, so it is used directly without
        # consulting the partition array.
        metrics = {
            "vcn_activity": [0, 0, 0, 0],
            "xcp_stats.vcn_busy": [["N/A"] * 4 for _ in range(8)],
            "current_vclk0": 29,
            "current_dclk0": 22,
        }

        self.assertEqual(self._run_decoder(metrics), "0 %")

    def test_no_decode_data_stays_na(self):
        # Neither source has data: DEC% must remain N/A rather than a fabricated
        # 0.
        metrics = {
            "vcn_activity": ["N/A", "N/A", "N/A", "N/A"],
            "xcp_stats.vcn_busy": [["N/A"] * 4 for _ in range(8)],
            "current_vclk0": "N/A",
            "current_dclk0": "N/A",
        }

        self.assertEqual(self._run_decoder(metrics), "N/A")


if __name__ == "__main__":
    unittest.main()
