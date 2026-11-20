#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi xgmi``'s CPU-column logic.

These drive ``XgmiCommands.xgmi`` with the C library, logger, and helpers
fully stubbed, so they run without GPU hardware or the compiled ``amdsmi``
package. They lock in three fixes to the GPU-CPU XGMI column logic:

* The GPU destination grid is hidden only when the CPU columns are actually
  replacing it, not on any other resolution failure (single GPU, PCIe-only
  host, a raised ``amdsmi_get_link_metrics()``).
* A GPU whose id and a connected CPU's package id are both ``0`` no longer
  collide on the same ``bdf_0`` tabular-output key.
* A ``UINT64_MAX`` read/write counter (an unpopulated ``links[]`` slot) in
  the CPU column renders as ``"N/A"``, not a nonsensical byte count.
"""

import argparse
import os
import types
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
XGMI_PATH = os.path.join(_CLI_DIR, "subcommands", "xgmi.py") if _CLI_DIR else None

_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class _FakeAmdSmiException(Exception):
    """Stands in for ``AmdSmiException``, the base xgmi.py catches."""


class _FakeLibraryException(_FakeAmdSmiException):
    def __init__(self, message="mock error"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


def _raise_lib_exc(*_args, **_kwargs):
    raise _FakeLibraryException("mock error")


class _FakeLogger:
    """``multiple_device_output`` is reassigned multiple times per ``xgmi()``
    call (tabular_output, then xgmi_values) -- record every assignment so a
    test can inspect the tabular one even though it isn't the final value."""

    def __init__(self):
        self.table_header = ""
        self.table_title = ""
        self.destination = "stdout"
        self.multiple_device_output_history = []

    @property
    def multiple_device_output(self):
        return (
            self.multiple_device_output_history[-1] if self.multiple_device_output_history else None
        )

    @multiple_device_output.setter
    def multiple_device_output(self, value):
        self.multiple_device_output_history.append(value)

    def is_human_readable_format(self):
        return True

    def is_csv_format(self):
        return False

    def is_json_format(self):
        return False

    def print_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def __init__(self, gpu_ids):
        self._gpu_ids = gpu_ids

    def _get_socket_counts(self):
        return (len(self._gpu_ids), len(self._gpu_ids), 0)

    def check_required_groups(self):
        pass

    def is_primary_partition(self, _handle):
        return True

    def get_gpu_id_from_device_handle(self, handle):
        return self._gpu_ids[handle]

    def unit_format(self, _logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(_logger, v, unit) for v in value]
        if value == "N/A":
            return "N/A"
        return f"{value} {unit}".rstrip() if unit else f"{value}"

    def convert_bytes_to_readable(self, byte_value, _to_str):
        return f"{byte_value}B"


def _build_fake_amdsmi(bdf_by_handle, get_cpu_links, get_link_metrics):
    """Register a stub ``amdsmi`` package so ``xgmi.py`` imports cleanly."""
    handle_by_bdf = {bdf: handle for handle, bdf in bdf_by_handle.items()}

    wrapper = types.SimpleNamespace(
        AMDSMI_LINK_TYPE_INTERNAL=0, AMDSMI_LINK_TYPE_PCIE=1, AMDSMI_LINK_TYPE_XGMI=2
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        amdsmi_wrapper=wrapper,
        amdsmi_get_gpu_device_bdf=lambda handle: bdf_by_handle[handle],
        amdsmi_get_processor_handle_from_bdf=lambda bdf: handle_by_bdf[bdf],
        amdsmi_get_gpu_cpu_links=get_cpu_links,
        amdsmi_get_link_metrics=get_link_metrics,
        amdsmi_topo_get_link_type=_raise_lib_exc,
    )
    exception = fake_module(
        "amdsmi.amdsmi_exception",
        AmdSmiException=_FakeAmdSmiException,
        AmdSmiLibraryException=_FakeLibraryException,
    )
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
    }


class TestCliXgmiCpuColumns(unittest.TestCase):
    def _run_xgmi(self, handles, gpu_ids, bdf_by_handle, get_cpu_links, get_link_metrics):
        if not XGMI_PATH or not os.path.isfile(XGMI_PATH):
            self.skipTest(f"amd-smi CLI xgmi.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi(bdf_by_handle, get_cpu_links, get_link_metrics)
        stub_modules(type(self), modules)
        xgmi_module = load_cli_module("xgmi_under_test", XGMI_PATH, sys_path_dir=_CLI_DIR)

        commands = object.__new__(xgmi_module.XgmiCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers(gpu_ids)
        commands.group_check_printed = True
        commands.device_handles = []

        args = argparse.Namespace(gpu=handles, metric=True, link_status=False, source_status=False)
        commands.xgmi(args)
        return commands

    def test_gpu_grid_hidden_when_cpu_columns_replace_it(self):
        # Single GPU, unresolvable GPU-GPU link metrics (simulates MI450,
        # where only the CPU link is real), one connected CPU package.
        gpu0 = object()
        commands = self._run_xgmi(
            handles=[gpu0],
            gpu_ids={gpu0: 0},
            bdf_by_handle={gpu0: "0000:01:00.0"},
            get_cpu_links=lambda _h: [{"package_id": 0, "link_type": 2}],
            get_link_metrics=_raise_lib_exc,
        )
        self.assertNotIn("GPU0", commands.logger.table_header)
        self.assertIn("CPU(PKG0)", commands.logger.table_header)

    def test_gpu_grid_kept_when_nothing_replaces_it(self):
        # Same unresolvable GPU-GPU link metrics, but no CPU link either --
        # the GPU grid must fall back to showing, not disappear entirely.
        gpu0 = object()
        commands = self._run_xgmi(
            handles=[gpu0],
            gpu_ids={gpu0: 0},
            bdf_by_handle={gpu0: "0000:01:00.0"},
            get_cpu_links=lambda _h: [],
            get_link_metrics=_raise_lib_exc,
        )
        self.assertIn("GPU0", commands.logger.table_header)
        self.assertNotIn("CPU(PKG", commands.logger.table_header)

    def test_tabular_keys_do_not_collide_between_gpu0_and_pkg0(self):
        # GPU id 0 and a connected CPU's package id 0 must not share a
        # tabular-output key: GPU0's own "Read"/"Write" row carries both a
        # "bdf_0" GPU column and a "pkg_0" CPU column, under distinct keys.
        gpu0, gpu1 = object(), object()
        bdf0, bdf1 = "0000:01:00.0", "0000:02:00.0"

        def _link_metrics(handle):
            if handle is gpu0:
                return {
                    "links": [
                        {
                            "bdf": bdf1,
                            "bit_rate": 100,
                            "max_bandwidth": 200,
                            "read": 1000,
                            "write": 2000,
                        }
                    ]
                }
            return {
                "links": [
                    {"bdf": bdf0, "bit_rate": 100, "max_bandwidth": 200, "read": 500, "write": 600}
                ]
            }

        def _cpu_links(handle):
            return [{"package_id": 0, "link_type": 2}] if handle is gpu0 else []

        commands = self._run_xgmi(
            handles=[gpu0, gpu1],
            gpu_ids={gpu0: 0, gpu1: 1},
            bdf_by_handle={gpu0: bdf0, gpu1: bdf1},
            get_cpu_links=_cpu_links,
            get_link_metrics=_link_metrics,
        )

        # The first assignment is the tabular rows; a later one overwrites it
        # with xgmi_values for JSON/CSV, so the history has to be used.
        tabular_output = commands.logger.multiple_device_output_history[0]
        # Per source GPU: [metrics row, Read row, Write row]; GPU0 is first.
        gpu0_read, gpu0_write = tabular_output[1], tabular_output[2]

        # Self-destination (GPU0 -> GPU0) never resolves against GPU0's own
        # metrics (which only lists a link to GPU1), so its GPU column is N/A.
        self.assertEqual(gpu0_read["bdf_0"], "N/A")
        self.assertEqual(gpu0_write["bdf_0"], "N/A")
        # GPU0's one links[] entry is a resolved GPU-GPU destination (to GPU1),
        # not a CPU link counter, and a resolved GPU-GPU link anywhere on the
        # host means no links[] entry can be safely attributed to a CPU link
        # (see xgmi.py's gpu_gpu_resolvable guard) -- so the CPU column is N/A
        # too, under its own distinct key, not merged into / overwriting the
        # GPU column above.
        self.assertEqual(gpu0_read["pkg_0"], "N/A")
        self.assertEqual(gpu0_write["pkg_0"], "N/A")

    def test_uint64_max_counter_renders_as_na(self):
        # links[] is a fixed-size array; an unpopulated slot reads back as
        # the driver's UINT64_MAX sentinel, which must map to "N/A". The
        # link's bdf deliberately doesn't match any known GPU (a CPU
        # destination isn't one), so gpu_gpu_resolvable stays False and this
        # entry's counter is read as the CPU link's, exercising the sentinel
        # check below rather than short-circuiting to "N/A" some other way.
        gpu0 = object()
        commands = self._run_xgmi(
            handles=[gpu0],
            gpu_ids={gpu0: 0},
            bdf_by_handle={gpu0: "0000:01:00.0"},
            get_cpu_links=lambda _h: [{"package_id": 0, "link_type": 2}],
            get_link_metrics=lambda _h: {
                "links": [
                    {
                        "bdf": "0000:ff:00.0",
                        "bit_rate": 100,
                        "max_bandwidth": 200,
                        "read": _UINT64_MAX,
                        "write": _UINT64_MAX,
                    }
                ]
            },
        )

        cpu_link = commands.logger.multiple_device_output[0]["link_metrics"]["cpu_links"][0]
        self.assertEqual(cpu_link["read"], "N/A")
        self.assertEqual(cpu_link["write"], "N/A")
