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

"""Hardware-free unit tests for partition-scoped ``amd-smi metric`` transforms.

These tests drive ``MetricCommands.metric_gpu()`` with the amdsmi_interface
library calls and driver-state helpers patched out, so the partition
data-shaping logic (per-XCP usage, per-AID/per-XCP clock breakdown, partition
temperature source/XCD count) is exercised without a GPU. Only the in-Python
transformation is under test; no real library call is made.
"""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch


def _resolve_amdsmi_path():
    return os.environ.get("AMDSMI_PATH") or os.path.join(
        os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm", "share/amd_smi"
    )


def _resolve_cli_source_dir():
    # The CLI sources live in the project tree, not the installed share dir.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "amdsmi_cli"))
    return candidate if os.path.isdir(candidate) else None


# Every metric category flag metric_gpu() inspects. All default off so a test
# can enable exactly one category and isolate its transform.
_METRIC_FLAGS = (
    "usage",
    "mem_usage",
    "power",
    "clock",
    "temperature",
    "voltage",
    "pcie",
    "ecc",
    "ecc_blocks",
    "base_board",
    "gpu_board",
    "fan",
    "voltage_curve",
    "overdrive",
    "perf_level",
    "xgmi_err",
    "energy",
    "throttle",
    "violation",
    "schedule",
    "guard",
    "guest_data",
    "fb_usage",
    "xgmi",
)

# Opaque stand-in for a device handle; never dereferenced because every library
# call that would consume it is patched.
_GPU_HANDLE = object()


class TestAmdSmiPartitionMetricTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli_dir = _resolve_cli_source_dir()
        if cli_dir is None:
            raise unittest.SkipTest("amdsmi_cli source directory not found")
        amdsmi_path = _resolve_amdsmi_path()
        if not os.path.isdir(amdsmi_path):
            raise unittest.SkipTest(f"amdsmi path '{amdsmi_path}' does not exist")

        sys.path.insert(0, cli_dir)
        sys.path.insert(0, amdsmi_path)
        try:
            from amdsmi import amdsmi_interface
            from amdsmi_helpers import AMDSMIHelpers
            from amdsmi_logger import AMDSMILogger
            from subcommands.metric import MetricCommands
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import the CLI metric command: {e}")

        cls.amdsmi_interface = amdsmi_interface
        cls.AMDSMIHelpers = AMDSMIHelpers
        cls.AMDSMILogger = AMDSMILogger
        cls.MetricCommands = MetricCommands

    def _make_args(self, category, partition):
        args = SimpleNamespace(
            gpu=_GPU_HANDLE,
            watch=None,
            watch_time=None,
            iterations=None,
            loglevel="INFO",
            partition=partition,
        )
        for flag in _METRIC_FLAGS:
            setattr(args, flag, False)
        setattr(args, category, True)
        return args

    def _invoke(
        self, category, partition, gpu_metric, partition_metrics, num_partition, interface_mocks
    ):
        """Run metric_gpu() for a single category in JSON mode and return the
        resulting values dict (the structure the CLI would serialize)."""
        cmd = self.MetricCommands()
        helpers = self.AMDSMIHelpers()
        logger = self.AMDSMILogger(format="json", helpers=helpers)
        cmd.helpers = helpers
        cmd.logger = logger
        cmd.group_check_printed = True
        cmd.device_handles = [_GPU_HANDLE]

        part_info = {
            "metric_version": 1.9,
            "partition_id": 0,
            "num_partition": num_partition,
            "num_xcp": num_partition,
        }
        args = self._make_args(category, partition)

        H = self.AMDSMIHelpers
        ai = self.amdsmi_interface
        with ExitStack() as stack:

            def p(obj, name, **kwargs):
                stack.enter_context(patch.object(obj, name, **kwargs))

            p(H, "is_hypervisor", return_value=False)
            p(H, "is_windows", return_value=False)
            p(H, "is_baremetal", return_value=True)
            p(H, "is_linux", return_value=True)
            p(H, "get_gpu_id_from_device_handle", return_value=0)
            p(H, "os_info", return_value=("TestOS", "0"))
            p(H, "_get_metric_version_and_partition_info", return_value=part_info)

            p(ai, "amdsmi_get_gpu_metrics_info", return_value=gpu_metric)
            if partition:
                p(ai, "amdsmi_get_gpu_partition_metrics_info", return_value=partition_metrics)
            for name, value in interface_mocks.items():
                p(ai, name, return_value=value)

            cmd.metric_gpu(args)

        return logger.store_gpu_json_output[-1]

    # ----- usage -----------------------------------------------------------

    def test_usage_partition_builds_per_xcp_dicts(self):
        gpu_metric = {"vcn_activity": [10, 20], "jpeg_activity": [1, 2]}
        partition_metrics = {
            "xcp_stats.gfx_busy_inst": [[11, 12], [13, 14]],
            "xcp_stats.jpeg_busy": [[1], [2]],
            "xcp_stats.vcn_busy": [[3], [4]],
        }
        result = self._invoke(
            "usage",
            partition=True,
            gpu_metric=gpu_metric,
            partition_metrics=partition_metrics,
            num_partition=2,
            interface_mocks={
                "amdsmi_get_gpu_activity": {
                    "gfx_activity": 50,
                    "umc_activity": 10,
                    "mm_activity": 5,
                }
            },
        )
        gfx = result["usage"]["gfx_busy_inst"]
        self.assertEqual(set(gfx), {"XCP_0", "XCP_1"})
        self.assertEqual(set(result["usage"]["jpeg_busy"]), {"XCP_0", "XCP_1"})
        self.assertEqual(set(result["usage"]["vcn_busy"]), {"XCP_0", "XCP_1"})
        # XCP_0 carries the first partition's gfx list, wrapped per-value.
        self.assertEqual([item["value"] for item in gfx["XCP_0"]], [11, 12])

    def test_usage_socket_uses_uppercase_xcp_keys(self):
        gpu_metric = {
            "vcn_activity": [10],
            "jpeg_activity": [1],
            "xcp_stats.gfx_busy_inst": [100, 200],
            "xcp_stats.jpeg_busy": [1, 2],
            "xcp_stats.vcn_busy": [3, 4],
        }
        result = self._invoke(
            "usage",
            partition=False,
            gpu_metric=gpu_metric,
            partition_metrics=None,
            num_partition=2,
            interface_mocks={
                "amdsmi_get_gpu_activity": {
                    "gfx_activity": 50,
                    "umc_activity": 10,
                    "mm_activity": 5,
                }
            },
        )
        self.assertEqual(set(result["usage"]["gfx_busy_inst"]), {"XCP_0", "XCP_1"})

    # ----- clock -----------------------------------------------------------

    def test_clock_partition_per_aid_and_per_xcp_breakdown(self):
        partition_metrics = {
            "current_vclk0s": [800, 810],
            "current_dclk0s": [400, 410],
            "current_socclks": [900, 910],
            "current_gfxclks": [1500, 1600],
            "gfxclk_lock_status": 0b10,  # XCP_0 unlocked, XCP_1 locked
        }
        result = self._invoke(
            "clock",
            partition=True,
            gpu_metric={},
            partition_metrics=partition_metrics,
            num_partition=2,
            interface_mocks={
                "amdsmi_get_clock_info": {
                    "min_clk": 100,
                    "max_clk": 2000,
                    "clk": 1500,
                    "sleep_clk": 0,
                    "clk_deep_sleep": 0,
                },
                "amdsmi_get_clk_freq": {
                    "num_supported": 1,
                    "current": 0,
                    "frequency": [1000000000],
                },
            },
        )
        clocks = result["clock"]
        self.assertEqual(clocks["AID_0"]["vclk"]["value"], 800)
        self.assertEqual(clocks["AID_1"]["vclk"]["value"], 810)
        self.assertEqual(clocks["AID_0"]["vclk_min_limit"]["value"], 100)
        self.assertEqual(clocks["AID_0"]["vclk_max_limit"]["value"], 2000)
        self.assertEqual(clocks["XCP_0"]["gfx_clk"]["value"], 1500)
        self.assertEqual(clocks["XCP_0"]["gfx_clk_locked"], "DISABLED")
        self.assertEqual(clocks["XCP_1"]["gfx_clk_locked"], "ENABLED")

    def test_clock_partition_na_xcp_omits_limits(self):
        # An XCP with no current gfx_clk must not surface device-wide limits
        # with no value; only its lock state is still reported.
        partition_metrics = {
            "current_gfxclks": [1500, "N/A"],
            "gfxclk_lock_status": 0b10,  # XCP_0 unlocked, XCP_1 locked
        }
        result = self._invoke(
            "clock",
            partition=True,
            gpu_metric={},
            partition_metrics=partition_metrics,
            num_partition=2,
            interface_mocks={
                "amdsmi_get_clock_info": {
                    "min_clk": 100,
                    "max_clk": 2000,
                    "clk": 1500,
                    "sleep_clk": 0,
                    "clk_deep_sleep": 0,
                },
                "amdsmi_get_clk_freq": {
                    "num_supported": 1,
                    "current": 0,
                    "frequency": [1000000000],
                },
            },
        )
        clocks = result["clock"]
        # XCP_0 reported a value, so limits attach.
        self.assertEqual(clocks["XCP_0"]["gfx_clk"]["value"], 1500)
        self.assertEqual(clocks["XCP_0"]["gfx_min_clk"]["value"], 100)
        # XCP_1 is N/A: no value, no phantom limits, lock state still present.
        self.assertNotIn("gfx_clk", clocks["XCP_1"])
        self.assertNotIn("gfx_min_clk", clocks["XCP_1"])
        self.assertNotIn("gfx_max_clk", clocks["XCP_1"])
        self.assertEqual(clocks["XCP_1"]["gfx_clk_locked"], "ENABLED")

    # ----- temperature -----------------------------------------------------

    def test_temperature_partition_source_and_full_xcd(self):
        partition_metrics = {
            "temperature_hbm_stacks": [60, 61],
            "temperature_mid": [55],
            "temperature_aid": [50, 51],
            "xcp_stats.temperature_xcd": [70, 71, 72],
        }
        result = self._invoke(
            "temperature",
            partition=True,
            gpu_metric={},
            partition_metrics=partition_metrics,
            num_partition=2,
            interface_mocks={"amdsmi_get_temp_metric": 65},
        )
        temps = result["temperature"]
        self.assertEqual(temps["aid"], [{"value": 50, "unit": "C"}, {"value": 51, "unit": "C"}])
        # Partition path reports every XCD entry, not just num_partition of them.
        self.assertEqual(set(temps["xcd"]), {"XCP_0", "XCP_1", "XCP_2"})

    def test_temperature_socket_caps_xcd_at_num_partition(self):
        gpu_metric = {
            "temperature_hbm_stacks": [60, 61],
            "temperature_mid": [55],
            "temperature_aid": [50, 51],
            "xcp_stats.temperature_xcd": [70, 71, 72],
        }
        result = self._invoke(
            "temperature",
            partition=False,
            gpu_metric=gpu_metric,
            partition_metrics=None,
            num_partition=2,
            interface_mocks={"amdsmi_get_temp_metric": 65},
        )
        # Socket path caps XCD entries at num_partition (2 of the 3 reported).
        self.assertEqual(set(result["temperature"]["xcd"]), {"XCP_0", "XCP_1"})


if __name__ == "__main__":
    unittest.main()
