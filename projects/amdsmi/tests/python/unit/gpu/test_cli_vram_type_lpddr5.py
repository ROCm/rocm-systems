#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit test for ``amd-smi static --vram`` memory-type labelling.

Covers the LPDDR5 memory-type label. ``AMDSMI_VRAM_TYPE__MAX`` aliases
the highest real enum value (``LPDDR5`` == 31), and the auto-generated
``amdsmi_vram_type_t__enumvalues`` map resolves that shared key to the ``__MAX``
label. ``static.py`` special-cases the ``__MAX`` value and must translate it to
``LPDDR5``; it was previously mislabelled ``GDDR7``, which surfaced on gfx117x
APUs whose unified memory reports as LPDDR5.

``static.py`` is loaded from the source tree so the test exercises the code
under development rather than a possibly-stale installed copy.
"""

import argparse
import copy
import importlib.util
import os
import unittest

from common.common import amdsmi_path, fake_module, find_cli_dir, stub_modules

# Locate the CLI dir (amdsmi_path first so an AMDSMI_PATH override selects the
# matching install; see common.find_cli_dir). None -> setUpClass skips.
_CLI_DIR = find_cli_dir(amdsmi_path, os.path.dirname(os.path.abspath(__file__)))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None

# Mirrors amdsmi_wrapper.amdsmi_vram_type_t__enumvalues: LPDDR5 and __MAX share
# value 31, and the later duplicate key (__MAX) wins in the dict literal, so a
# plain lookup of 31 yields "__MAX" rather than "LPDDR5".
_ENUMVALUES = {
    0: "AMDSMI_VRAM_TYPE_UNKNOWN",
    1: "AMDSMI_VRAM_TYPE_HBM",
    22: "AMDSMI_VRAM_TYPE_GDDR6",
    23: "AMDSMI_VRAM_TYPE_GDDR7",
    30: "AMDSMI_VRAM_TYPE_LPDDR4",
    31: "AMDSMI_VRAM_TYPE__MAX",
}
_VRAM_TYPE__MAX = 31

_BASE_VRAM_INFO = {
    "vram_type": _VRAM_TYPE__MAX,
    "vram_vendor": "UNKNOWN",
    "vram_size": 512,
    "vram_bit_width": 128,
    "vram_max_bandwidth": "N/A",
}


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)


def _fake_modules(holder):
    """Build a stub ``amdsmi`` package plus the sibling CLI modules.

    Returns the name -> module mapping for ``common.stub_modules``.
    ``holder["info"]`` is the vram payload returned to ``static.py`` so each
    test can swap the reported ``vram_type`` without reloading the module.
    """

    def _get_vram_info(_handle):
        return copy.deepcopy(holder["info"])

    wrapper = fake_module(
        "amdsmi.amdsmi_interface.amdsmi_wrapper",
        AMDSMI_VRAM_TYPE__MAX=_VRAM_TYPE__MAX,
        amdsmi_vram_type_t__enumvalues=dict(_ENUMVALUES),
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface", amdsmi_get_gpu_vram_info=_get_vram_info, amdsmi_wrapper=wrapper
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)

    # ``static.py`` imports these sibling names at load time; the vram path
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
    spec = importlib.util.spec_from_file_location("static_under_test", STATIC_PATH)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"amd-smi CLI static.py is not loadable ({STATIC_PATH})")
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
    """Minimal helpers stub for the single-GPU, baremetal-off vram path."""

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
    """Namespace with vram on and every other static section off."""
    return argparse.Namespace(
        gpu=object(),
        asic=False,
        bus=False,
        vbios=False,
        driver=False,
        ras=False,
        vram=True,
        cache=False,
        board=False,
        process_isolation=False,
        clock=False,
        mem_carveout=False,
        partition=False,
    )


class TestCliVramTypeLpddr5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        cls.holder = {"info": copy.deepcopy(_BASE_VRAM_INFO)}
        stub_modules(cls, _fake_modules(cls.holder))
        cls.static_module = _load_static_module()

    def _run_vram(self, vram_type_value, fmt="human"):
        info = copy.deepcopy(_BASE_VRAM_INFO)
        info["vram_type"] = vram_type_value
        self.holder["info"] = info

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
        self.assertIn("vram", static_dict)
        return static_dict["vram"]["type"]

    def test_max_value_maps_to_lpddr5_human(self):
        # AMDSMI_VRAM_TYPE__MAX == LPDDR5 == 31; must not be mislabelled GDDR7.
        self.assertEqual(self._run_vram(_VRAM_TYPE__MAX, "human"), "LPDDR5")

    def test_max_value_maps_to_lpddr5_json(self):
        self.assertEqual(self._run_vram(_VRAM_TYPE__MAX, "json"), "LPDDR5")

    def test_lpddr4_still_labelled(self):
        self.assertEqual(self._run_vram(30, "human"), "LPDDR4")

    def test_non_max_type_unaffected(self):
        self.assertEqual(self._run_vram(22, "human"), "GDDR6")
