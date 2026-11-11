#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi static --cache`` label enrichment.

These tests drive ``StaticCommands.static_gpu`` with the C library, logger, and
helpers fully stubbed, so they run without GPU hardware. They lock in these
enrichments of the cache listing:

* each entry carries a ``cache_acronym`` derived from ``cache_level``, the
  DATA/INST flags and CU sharing (vL1D / sL1D / GL1 / L1I, else
  ``L<level>``),
* each entry carries a ``total_cache_size`` equal to the per-instance
  ``cache_size`` times ``num_cache_instance``, and
* ``cache_line_size`` is shown in bytes.

``static.py`` is loaded through cli_search_order, so a plain checkout
exercises the code under development; an explicit AMDSMI_PATH pins that install.
"""

import argparse
import copy
import os
import unittest

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

# A GPU with a per-shader-array GL1 (gfx10 style), one entry per cache kind as
# returned by ``amdsmi_get_gpu_cache_entries``: vector L1D, L1I, scalar L1D, GL1
# and L2. Harvesting leaves unpaired CUs and uneven shader arrays, so the SQC
# and GL1 sharing are ranges. The GL1 and L2 report no line size.
_GL1_CACHE = [
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 16,
        "cache_level": 1,
        "cache_line_size": 128,
        "min_num_cu_shared": 1,
        "max_num_cu_shared": 1,
        "num_cache_instance": 76,
    },
    {
        "cache_properties": ["INST_CACHE", "SIMD_CACHE"],
        "cache_size": 32,
        "cache_level": 1,
        "cache_line_size": 64,
        "min_num_cu_shared": 1,
        "max_num_cu_shared": 2,
        "num_cache_instance": 40,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 16,
        "cache_level": 1,
        "cache_line_size": 64,
        "min_num_cu_shared": 1,
        "max_num_cu_shared": 2,
        "num_cache_instance": 40,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 128,
        "cache_level": 1,
        "cache_line_size": 0,
        "min_num_cu_shared": 8,
        "max_num_cu_shared": 10,
        "num_cache_instance": 8,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_size": 4096,
        "cache_level": 2,
        "cache_line_size": 0,
        "min_num_cu_shared": 76,
        "max_num_cu_shared": 76,
        "num_cache_instance": 1,
    },
]


# A layout where every SQC serves one CU: the vector L1D, L1I and scalar L1D
# then share identical CU sharing and instance counts.
_GL1_ONE_CU_SQC_CACHE = copy.deepcopy(_GL1_CACHE)
for _entry in _GL1_ONE_CU_SQC_CACHE[1:3]:
    _entry.update(max_num_cu_shared=1, num_cache_instance=76)


# gfx90a and gfx950 as folded by amdsmi_get_gpu_cache_entries: no GL1, and
# gfx950 adds an L3.
def _cache(props, size, level, max_shared, instances):
    return {
        "cache_properties": [props, "SIMD_CACHE"],
        "cache_size": size,
        "cache_level": level,
        "cache_line_size": 64,
        "min_num_cu_shared": 1,
        "max_num_cu_shared": max_shared,
        "num_cache_instance": instances,
    }


_GFX90A_CACHE = [
    _cache("DATA_CACHE", 16, 1, 1, 104),
    _cache("INST_CACHE", 64, 1, 2, 56),
    _cache("DATA_CACHE", 16, 1, 2, 56),
    _cache("DATA_CACHE", 8192, 2, 104, 1),
]
_GFX950_CACHE = [
    _cache("DATA_CACHE", 32, 1, 1, 256),
    _cache("INST_CACHE", 64, 1, 2, 144),
    _cache("DATA_CACHE", 16, 1, 2, 144),
    _cache("DATA_CACHE", 4096, 2, 256, 1),
    _cache("DATA_CACHE", 229376, 3, 256, 1),
]

# Layout returned by the stub interface; tests may swap it.
_CACHE_FIXTURE = {"cache": _GL1_CACHE}


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


def _build_fake_modules():
    """Build a stub ``amdsmi`` package plus the sibling CLI modules.

    Returns the name -> module mapping for ``common.stub_modules``.
    """

    def _get_cache_entries(_handle):
        return {"cache": copy.deepcopy(_CACHE_FIXTURE["cache"])}

    interface = fake_module(
        "amdsmi.amdsmi_interface", amdsmi_get_gpu_cache_entries=_get_cache_entries
    )
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

    def _use_fixture(self, cache):
        previous = _CACHE_FIXTURE["cache"]
        _CACHE_FIXTURE["cache"] = cache
        self.addCleanup(_CACHE_FIXTURE.__setitem__, "cache", previous)

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
        expected_acronyms = ["vL1D", "L1I", "sL1D", "GL1", "L2"]
        for index, acronym in enumerate(expected_acronyms):
            self.assertEqual(cache_info[f"cache_{index}"].get("cache_acronym"), acronym)

        # Total size = per-instance cache_size * num_cache_instance, same unit.
        expected_totals = ["1216 KB", "1280 KB", "640 KB", "1024 KB", "4096 KB"]
        for index, total in enumerate(expected_totals):
            self.assertEqual(cache_info[f"cache_{index}"].get("total_cache_size"), total)

        # Per-instance size in KB, line size in bytes, N/A where the driver reports none.
        self.assertEqual(cache_info["cache_0"]["cache_size"], "16 KB")
        self.assertEqual(cache_info["cache_0"]["cache_line_size"], "128 B")
        self.assertEqual(cache_info["cache_2"]["cache_line_size"], "64 B")
        self.assertEqual(cache_info["cache_3"]["cache_line_size"], "N/A")
        self.assertEqual(cache_info["cache_4"]["cache_line_size"], "N/A")

    def test_json_labels_and_totals(self):
        cache_info = self._run_cache("json")

        acronyms = [entry.get("cache_acronym") for entry in cache_info]
        self.assertEqual(acronyms, ["vL1D", "L1I", "sL1D", "GL1", "L2"])

        expected_totals = [1216, 1280, 640, 1024, 4096]
        for index, value in enumerate(expected_totals):
            self.assertEqual(
                cache_info[index].get("total_cache_size"), {"value": value, "unit": "KB"}
            )

        # Per-instance size in KB, line size in bytes, N/A where the driver reports none.
        self.assertEqual(cache_info[0]["cache_size"], {"value": 16, "unit": "KB"})
        self.assertEqual(cache_info[0]["cache_line_size"], {"value": 128, "unit": "B"})
        self.assertEqual(cache_info[2]["cache_line_size"], {"value": 64, "unit": "B"})
        self.assertEqual(cache_info[3]["cache_line_size"], "N/A")
        self.assertEqual(cache_info[3]["min_num_cu_shared"], 8)

    def test_level1_labels_follow_kfd_order_when_sharing_matches(self):
        self._use_fixture(_GL1_ONE_CU_SQC_CACHE)
        cache_info = self._run_cache("json")

        acronyms = [entry.get("cache_acronym") for entry in cache_info]
        self.assertEqual(acronyms, ["vL1D", "L1I", "sL1D", "GL1", "L2"])

    def test_level1_data_cache_without_instruction_cache(self):
        self._use_fixture([_GL1_CACHE[0], _GL1_CACHE[4]])
        cache_info = self._run_cache("json")

        acronyms = [entry.get("cache_acronym") for entry in cache_info]
        self.assertEqual(acronyms, ["L1D", "L2"])

    def test_level1_labels_on_layouts_without_gl1(self):
        for cache, expected in (
            (_GFX90A_CACHE, ["vL1D", "L1I", "sL1D", "L2"]),
            (_GFX950_CACHE, ["vL1D", "L1I", "sL1D", "L2", "L3"]),
        ):
            with self.subTest(levels=[entry["cache_level"] for entry in cache]):
                self._use_fixture(cache)
                acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
                self.assertEqual(acronyms, expected)

    def test_level1_gl1_without_scalar_data_cache(self):
        # Firmware that reports no scalar L1D size makes KFD drop that cache; the
        # GL1 then follows the L1I directly but shares far more CUs than it.
        self._use_fixture([_GL1_CACHE[i] for i in (0, 1, 3, 4)])
        acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
        self.assertEqual(acronyms, ["vL1D", "L1I", "GL1", "L2"])

    def test_level1_gl1_sharing_like_the_sqc(self):
        # gfx1036/gfx1037 (Raphael/Mendocino): the GL1 serves as many CUs as an SQC.
        cache = [
            _cache("DATA_CACHE", 16, 1, 1, 2),
            _cache("INST_CACHE", 32, 1, 2, 1),
            _cache("DATA_CACHE", 16, 1, 2, 1),
            _cache("DATA_CACHE", 128, 1, 2, 1),
            _cache("DATA_CACHE", 256, 2, 2, 1),
        ]
        for entry in cache:
            entry["min_num_cu_shared"] = entry["max_num_cu_shared"]
        self._use_fixture(cache)
        acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
        self.assertEqual(acronyms, ["vL1D", "L1I", "sL1D", "GL1", "L2"])

        self._use_fixture([cache[i] for i in (0, 1, 3, 4)])
        acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
        self.assertEqual(acronyms, ["vL1D", "L1I", "GL1", "L2"])

    def test_level1_scalar_data_cache_listed_before_instruction_cache(self):
        self._use_fixture([_GFX90A_CACHE[i] for i in (0, 2, 1, 3)])
        acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
        self.assertEqual(acronyms, ["vL1D", "sL1D", "L1I", "L2"])

    def test_level1_vector_cache_sharing_like_the_instruction_cache(self):
        # Only the L1I moves to one CU per instance, so it shares like the vector
        # L1D in front of it, not like the scalar L1D after it.
        inst_one_cu = copy.deepcopy(_GL1_CACHE)
        inst_one_cu[1].update(max_num_cu_shared=1, num_cache_instance=76)
        one_cu_no_scalar = [_GL1_ONE_CU_SQC_CACHE[i] for i in (0, 1, 3, 4)]
        for cache, expected in (
            (inst_one_cu, ["vL1D", "L1I", "sL1D", "GL1", "L2"]),
            (one_cu_no_scalar, ["vL1D", "L1I", "GL1", "L2"]),
        ):
            with self.subTest(entries=len(cache)):
                self._use_fixture(cache)
                acronyms = [entry.get("cache_acronym") for entry in self._run_cache("json")]
                self.assertEqual(acronyms, expected)
