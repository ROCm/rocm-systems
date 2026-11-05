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
  ``cache_size`` times ``num_cache_instance``, and
* each entry carries a ``cache_name`` that tells apart the level-1 data caches
  KFD reports with identical flags (vector L1, scalar L1, GL1).

``static.py`` is loaded through cli_search_order, so a plain checkout
exercises the code under development; an explicit AMDSMI_PATH pins that install.
"""

import argparse
import copy
import os
import unittest
from unittest import mock

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

# Locate the CLI dir; cli_search_order() decides whether the install or this
# checkout wins. None -> setUpClass skips.
_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None

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


def _cache_entry(props, size, level, cu_shared, instances):
    return {
        "cache_properties": [props, "SIMD_CACHE"],
        "cache_size": size,
        "cache_level": level,
        "max_num_cu_shared": cu_shared,
        "num_cache_instance": instances,
    }


# gfx1250 layout: seven level-1 entries, including three 256 KB GL1 groups.
_GFX1250_CACHE = [
    _cache_entry("DATA_CACHE", 32, 1, 1, 256),
    _cache_entry("INST_CACHE", 64, 1, 2, 112),
    _cache_entry("INST_CACHE", 64, 1, 1, 32),
    _cache_entry("DATA_CACHE", 16, 1, 2, 112),
    _cache_entry("DATA_CACHE", 16, 1, 1, 32),
    _cache_entry("DATA_CACHE", 256, 1, 9, 14),
    _cache_entry("DATA_CACHE", 256, 1, 7, 14),
    _cache_entry("DATA_CACHE", 256, 1, 8, 4),
    _cache_entry("DATA_CACHE", 4096, 2, 258, 1),
]

# gfx1100 (Navi31) layout captured live: one GL1 group plus L2 and L3.
_GFX1100_CACHE = [
    _cache_entry("DATA_CACHE", 32, 1, 1, 96),
    _cache_entry("INST_CACHE", 32, 1, 2, 48),
    _cache_entry("DATA_CACHE", 16, 1, 2, 48),
    _cache_entry("DATA_CACHE", 256, 1, 8, 12),
    _cache_entry("DATA_CACHE", 6144, 2, 96, 1),
    _cache_entry("DATA_CACHE", 98304, 3, 96, 1),
]


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


def _build_fake_modules():
    """Build a stub ``amdsmi`` package plus the sibling CLI modules.

    Returns the name -> module mapping for ``common.stub_modules``.
    """

    def _get_cache_info(_handle):
        return {"cache": copy.deepcopy(_GFX950_CACHE)}

    interface = fake_module("amdsmi.amdsmi_interface", amdsmi_get_gpu_cache_info=_get_cache_info)
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)

    # ``static.py`` imports these sibling names at load time; the cache path
    # never instantiates them (the test injects a fake helpers object).
    helpers_mod = fake_module("amdsmi_helpers", AMDSMIHelpers=object)
    exceptions_mod = fake_module(
        "amdsmi_cli_exceptions",
        AmdSmiInvalidParameterException=type("AmdSmiInvalidParameterException", (Exception,), {}),
    )

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi_helpers": helpers_mod,
        "amdsmi_cli_exceptions": exceptions_mod,
    }


def _load_static_module():
    return load_cli_module("static_under_test", STATIC_PATH)


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
        numa=False,
    )


class TestCliCacheLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_modules()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.static_module = _load_static_module()

    def _run_cache(self, fmt, layout=None):
        if layout is not None:
            patcher = mock.patch.object(
                self.interface,
                "amdsmi_get_gpu_cache_info",
                lambda _handle: {"cache": copy.deepcopy(layout)},
            )
            patcher.start()
            self.addCleanup(patcher.stop)
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

    def test_gfx950_cache_names(self):
        names = [entry.get("cache_name") for entry in self._run_cache("json")]
        self.assertEqual(
            names,
            [
                "L1 Vector Data",
                "L1 Instruction",
                "L1 Instruction",
                "L1 Scalar Data",
                "L1 Scalar Data",
                "L2 Data",
                "L3 Data",
            ],
            "vector and scalar L1 data caches must get distinct cache_name values",
        )

    def test_gfx1250_level1_caches_are_distinguishable(self):
        cache_info = self._run_cache("human", _GFX1250_CACHE)
        names = [cache_info[f"cache_{index}"].get("cache_name") for index in range(9)]
        self.assertEqual(
            names,
            [
                "L1 Vector Data",
                "L1 Instruction",
                "L1 Instruction",
                "L1 Scalar Data",
                "L1 Scalar Data",
                "GL1 Data",
                "GL1 Data",
                "GL1 Data",
                "L2 Data",
            ],
            "256 KB per-shader-array caches must be named GL1, not merged with scalar L1",
        )
        # cache_acronym is a released field; its values must not change.
        acronyms = [cache_info[f"cache_{index}"]["cache_acronym"] for index in range(9)]
        self.assertEqual(acronyms, ["L1D", "L1I", "L1I", "L1D", "L1D", "L1D", "L1D", "L1D", "L2"])

    def test_gfx1100_cache_names_json(self):
        names = [entry.get("cache_name") for entry in self._run_cache("json", _GFX1100_CACHE)]
        self.assertEqual(
            names,
            [
                "L1 Vector Data",
                "L1 Instruction",
                "L1 Scalar Data",
                "GL1 Data",
                "L2 Data",
                "L3 Data",
            ],
        )

    def test_level1_data_without_instruction_cache_stays_generic(self):
        # Vector/scalar/GL1 are told apart by their position around the L1
        # instruction cache; without one, do not guess.
        layout = [
            _cache_entry("DATA_CACHE", 16, 1, 1, 64),
            _cache_entry("DATA_CACHE", 128, 1, 8, 4),
            _cache_entry("DATA_CACHE", 4096, 2, 64, 1),
        ]
        names = [entry.get("cache_name") for entry in self._run_cache("json", layout)]
        self.assertEqual(names, ["L1 Data", "L1 Data", "L2 Data"])
