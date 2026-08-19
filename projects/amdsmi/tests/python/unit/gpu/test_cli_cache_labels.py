#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi static --cache`` label enrichment.

These tests drive ``StaticCommands.static_gpu`` with the C library, logger, and
helpers fully stubbed, so they run without GPU hardware. They lock in two
enrichments of the cache listing:

* each entry carries a ``cache_acronym`` derived from ``cache_level`` plus the
  DATA/INST flags (L1D / L1I / L2 / L3, with an ``L<level>`` fallback), and
* each entry carries a ``total_cache_size`` equal to the per-instance
  ``cache_size`` times ``num_cache_instance``.

``static.py`` is loaded from the source tree so the test exercises the code
under development rather than a possibly-stale installed copy.
"""

import argparse
import copy
import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path, find_cli_dir

# Locate the CLI dir (amdsmi_path first so an AMDSMI_PATH override selects the
# matching install; see common.find_cli_dir). None -> setUpClass skips.
_CLI_DIR = find_cli_dir(amdsmi_path, os.path.dirname(os.path.abspath(__file__)))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None

# Modules the fakes below replace. Snapshotted around the suite so a real copy
# loaded by a sibling test is never left shadowed by a stub.
_CLI_MODULES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi_helpers",
    "amdsmi_cli_exceptions",
)

# gfx950 (MI350) cache layout captured live: four L1 variants plus L2 and L3.
# Mirrors the per-entry dict shape returned by ``amdsmi_get_gpu_cache_info``.
_GFX950_CACHE = [
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 32,
        "cache_level": 1,
        "max_num_cu_shared": 1,
        "num_cache_instance": 256,
    },
    {
        "cache_properties": ["INST_CACHE", "SIMD_CACHE"],
        "cache_size": 64,
        "cache_level": 1,
        "max_num_cu_shared": 2,
        "num_cache_instance": 112,
    },
    {
        "cache_properties": ["INST_CACHE", "SIMD_CACHE"],
        "cache_size": 64,
        "cache_level": 1,
        "max_num_cu_shared": 1,
        "num_cache_instance": 32,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 16,
        "cache_level": 1,
        "max_num_cu_shared": 2,
        "num_cache_instance": 112,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 16,
        "cache_level": 1,
        "max_num_cu_shared": 1,
        "num_cache_instance": 32,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 4096,
        "cache_level": 2,
        "max_num_cu_shared": 256,
        "num_cache_instance": 1,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 229376,
        "cache_level": 3,
        "max_num_cu_shared": 256,
        "num_cache_instance": 1,
    },
]


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


def _install_fake_modules():
    """Register a stub ``amdsmi`` package plus the sibling CLI modules.

    Returns the fake ``amdsmi_interface`` so tests can swap the cache payload.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    def _get_cache_info(_handle):
        return {"cache": copy.deepcopy(_GFX950_CACHE)}

    interface.amdsmi_get_gpu_cache_info = _get_cache_info
    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception

    # ``static.py`` imports these sibling names at load time; the cache path
    # never instantiates them (the test injects a fake helpers object).
    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = object
    sys.modules["amdsmi_helpers"] = helpers_mod

    exceptions_mod = types.ModuleType("amdsmi_cli_exceptions")
    exceptions_mod.AmdSmiInvalidParameterException = type(
        "AmdSmiInvalidParameterException", (Exception,), {}
    )
    sys.modules["amdsmi_cli_exceptions"] = exceptions_mod

    return interface


def _load_static_module():
    spec = importlib.util.spec_from_file_location("static_under_test", STATIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload ``static_gpu`` stores per GPU."""

    def __init__(self, fmt):
        self._fmt = fmt
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self._fmt == "json"

    def is_csv_format(self):
        return self._fmt == "csv"

    def is_human_readable_format(self):
        return self._fmt == "human"

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass


class _FakeHelpers:
    """Minimal helpers stub for the single-GPU, baremetal-off cache path."""

    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def check_required_groups(self):
        pass

    def is_linux(self):
        return True

    def is_baremetal(self):
        return False

    def is_virtual_os(self):
        return True

    def is_hypervisor(self):
        return False


def _build_args():
    """Namespace with cache on and every other static section off."""
    return argparse.Namespace(
        gpu=object(),
        asic=False,
        bus=False,
        vbios=False,
        driver=False,
        ras=False,
        vram=False,
        cache=True,
        board=False,
        process_isolation=False,
        clock=False,
        mem_carveout=False,
        partition=False,
    )


class TestCliCacheLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI not found ({STATIC_PATH or _CLI_DIR}): static.py not present"
            )
        cls._saved_modules = {name: sys.modules.get(name) for name in _CLI_MODULES}
        cls.interface = _install_fake_modules()
        cls.static_module = _load_static_module()

    @classmethod
    def tearDownClass(cls):
        for name, mod in getattr(cls, "_saved_modules", {}).items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _run_cache(self, fmt):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True

        commands.static_gpu(_build_args())

        if fmt == "json":
            self.assertTrue(commands.logger.store_gpu_json_output)
            static_dict = commands.logger.store_gpu_json_output[-1]
        else:
            static_dict = commands.logger.captured_values
        self.assertIsNotNone(static_dict, "static_gpu stored no values payload")
        self.assertIn("cache_info", static_dict)
        return static_dict["cache_info"]

    def test_human_readable_labels_and_totals(self):
        cache_info = self._run_cache("human")

        # Acronyms derived from level + DATA/INST flags, one assertion per fixture entry.
        expected_acronyms = ["L1D", "L1I", "L1I", "L1D", "L1D", "L2", "L3"]
        for index, acronym in enumerate(expected_acronyms):
            self.assertEqual(cache_info[f"cache_{index}"].get("cache_acronym"), acronym)

        # Total size = per-instance cache_size * num_cache_instance, same unit.
        expected_totals = [
            "8192 KB",
            "7168 KB",
            "2048 KB",
            "1792 KB",
            "512 KB",
            "4096 KB",
            "229376 KB",
        ]
        for index, total in enumerate(expected_totals):
            self.assertEqual(cache_info[f"cache_{index}"].get("total_cache_size"), total)

        # Existing per-instance size formatting is unchanged.
        self.assertEqual(cache_info["cache_0"]["cache_size"], "32 KB")

    def test_json_labels_and_totals(self):
        cache_info = self._run_cache("json")

        acronyms = [entry.get("cache_acronym") for entry in cache_info]
        self.assertEqual(acronyms, ["L1D", "L1I", "L1I", "L1D", "L1D", "L2", "L3"])

        expected_totals = [8192, 7168, 2048, 1792, 512, 4096, 229376]
        for index, value in enumerate(expected_totals):
            self.assertEqual(
                cache_info[index].get("total_cache_size"), {"value": value, "unit": "KB"}
            )

        # Existing per-instance size wrapping is unchanged.
        self.assertEqual(cache_info[0]["cache_size"], {"value": 32, "unit": "KB"})
