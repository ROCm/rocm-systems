#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Mock-based unit tests for ``amd-smi static --ampp`` display logic.

Covers ``StaticCommands.static_gpu``'s ``--ampp`` block: JSON output shape,
human-readable formatting (including a per-profile fields-fetch failure), and
the NOT_SUPPORTED fallback. Follows the same load-from-source-tree / stub the
``amdsmi`` package pattern used by the other source-loaded CLI tests.
"""

import copy
import importlib.util
import os
import sys
import types
import unittest
from argparse import Namespace

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
STATIC_PATH = os.path.join(_REPO_ROOT, "amdsmi_cli", "subcommands", "static.py")

_STATUS_NOT_SUPPORTED = 2
_STATUS_NO_DATA = 40

_PROFILES = [
    {
        "name": "profile_0",
        "index": 0,
        "is_active": True,
        "is_writable": False,
        "is_configured": True,
    },
    {
        "name": "profile_2",
        "index": 2,
        "is_active": False,
        "is_writable": True,
        "is_configured": True,
    },
]

_FIELDS_BY_PROFILE = {
    "profile_0": [
        {
            "name": "PPT0_Limit",
            "unit": "W",
            "value": 300,
            "limit_min": 0,
            "limit_max": 0,
            "has_limits": False,
        }
    ],
    "profile_2": [
        {
            "name": "PPT0_Limit",
            "unit": "W",
            "value": 250,
            "limit_min": 100,
            "limit_max": 400,
            "has_limits": True,
        }
    ],
}


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self, detailed=True):
        return self._message


def _install_fake_modules(holder):
    """Register a stub ``amdsmi`` package plus the sibling CLI modules.

    ``holder`` supplies the per-test behavior for
    ``amdsmi_get_ampp_profiles``/``amdsmi_get_ampp_fields`` so each test can
    swap it without reloading ``static.py``.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    def _get_ampp_profiles(_handle):
        return holder["get_profiles"]()

    def _get_ampp_fields(_handle, profile_name):
        return holder["get_fields"](profile_name)

    interface.amdsmi_get_ampp_profiles = _get_ampp_profiles
    interface.amdsmi_get_ampp_fields = _get_ampp_fields

    wrapper = types.ModuleType("amdsmi.amdsmi_interface.amdsmi_wrapper")
    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    wrapper.AMDSMI_STATUS_NO_DATA = _STATUS_NO_DATA
    interface.amdsmi_wrapper = wrapper

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception

    # ``static.py`` imports these sibling names at load time; the ampp path
    # never instantiates them (the test injects a fake helpers object).
    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = object
    sys.modules["amdsmi_helpers"] = helpers_mod

    exceptions_mod = types.ModuleType("amdsmi_cli_exceptions")
    exceptions_mod.AmdSmiInvalidParameterException = type(
        "AmdSmiInvalidParameterException", (Exception,), {}
    )
    sys.modules["amdsmi_cli_exceptions"] = exceptions_mod


def _load_static_module():
    spec = importlib.util.spec_from_file_location("static_ampp_under_test", STATIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values``/JSON payload ``static_gpu`` stores per GPU."""

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
    """Minimal helpers stub for the single-GPU ampp path."""

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
    """Namespace with ampp on and every other static section off."""
    return Namespace(
        gpu=object(),
        asic=False,
        bus=False,
        vbios=False,
        driver=False,
        ras=False,
        vram=False,
        cache=False,
        board=False,
        process_isolation=False,
        clock=False,
        mem_carveout=False,
        partition=False,
        ampp=True,
    )


class TestCliStaticAmpp(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi_helpers",
        "amdsmi_cli_exceptions",
    )

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI static.py not found at {STATIC_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.holder = {
            "get_profiles": lambda: ("1.0", copy.deepcopy(_PROFILES)),
            "get_fields": lambda name: copy.deepcopy(_FIELDS_BY_PROFILE[name]),
        }
        _install_fake_modules(cls.holder)
        cls.static_module = _load_static_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def setUp(self):
        # Tests mutate the shared class-level holder; reset it so one test's
        # override can't leak into another regardless of run order.
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))
        self.holder["get_fields"] = lambda name: copy.deepcopy(_FIELDS_BY_PROFILE[name])

    def _run_ampp(self, fmt="human"):
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
        return static_dict

    def test_json_output_includes_version_and_profiles(self):
        static_dict = self._run_ampp("json")
        self.assertIn("ampp", static_dict)
        self.assertEqual(static_dict["ampp"]["version"], "1.0")
        names = [p["name"] for p in static_dict["ampp"]["profiles"]]
        self.assertEqual(names, ["profile_0", "profile_2"])
        # Configured profiles get their fields populated.
        self.assertEqual(
            static_dict["ampp"]["profiles"][0]["fields"], _FIELDS_BY_PROFILE["profile_0"]
        )

    def test_human_readable_output_lists_profiles_and_fields(self):
        static_dict = self._run_ampp("human")
        ampp_text = static_dict["ampp"]
        self.assertIn("ABI_VERSION: 1.0", ampp_text)
        self.assertIn("*[0] profile_0 (configured)", ampp_text)
        self.assertIn(" [2] profile_2 (writable, configured)", ampp_text)
        self.assertIn("PPT0_Limit: 300 W", ampp_text)
        self.assertIn("PPT0_Limit: 250 W (min=100, max=400)", ampp_text)

    def test_not_supported_reports_descriptive_reason_human(self):
        self.holder["get_profiles"] = lambda: (_ for _ in ()).throw(
            _FakeLibraryException(_STATUS_NOT_SUPPORTED, "not supported")
        )
        static_dict = self._run_ampp("human")
        self.assertEqual(static_dict["ampp"], "N/A (AMPP is not supported on this ASIC/VBIOS)")

    def test_not_supported_reports_bare_na_json(self):
        self.holder["get_profiles"] = lambda: (_ for _ in ()).throw(
            _FakeLibraryException(_STATUS_NOT_SUPPORTED, "not supported")
        )
        static_dict = self._run_ampp("json")
        self.assertEqual(static_dict["ampp"], "N/A")

    def test_per_profile_fields_fetch_failure_no_data_yields_empty_fields(self):
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))

        def _raise_no_data(_name):
            raise _FakeLibraryException(_STATUS_NO_DATA, "no data")

        self.holder["get_fields"] = _raise_no_data
        static_dict = self._run_ampp("json")
        for profile in static_dict["ampp"]["profiles"]:
            self.assertEqual(profile["fields"], [])

    def test_per_profile_fields_fetch_failure_other_error_yields_na(self):
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))

        def _raise_other(_name):
            raise _FakeLibraryException(9999, "some other error")

        self.holder["get_fields"] = _raise_other
        static_dict = self._run_ampp("json")
        for profile in static_dict["ampp"]["profiles"]:
            self.assertEqual(profile["fields"], "N/A")

    def test_per_profile_fields_fetch_failure_other_error_yields_na_human(self):
        # The "N/A" sentinel is a str, not a list -- the human-readable
        # branch must not iterate it character by character.
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))

        def _raise_other(_name):
            raise _FakeLibraryException(9999, "some other error")

        self.holder["get_fields"] = _raise_other
        static_dict = self._run_ampp("human")
        ampp_text = static_dict["ampp"]
        self.assertIn("ABI_VERSION: 1.0", ampp_text)
        self.assertIn("FIELDS: N/A", ampp_text)

    def test_human_readable_output_zero_profiles_reports_na(self):
        self.holder["get_profiles"] = lambda: ("1.0", [])
        static_dict = self._run_ampp("human")
        self.assertEqual(static_dict["ampp"], "N/A")

    def test_csv_output_zero_profiles_reports_na(self):
        self.holder["get_profiles"] = lambda: ("1.0", [])
        static_dict = self._run_ampp("csv")
        self.assertEqual(static_dict["ampp_version"], "1.0")
        self.assertEqual(static_dict["ampp"], "N/A")

    def test_csv_output_includes_flattened_profile_state(self):
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))
        self.holder["get_fields"] = lambda name: copy.deepcopy(_FIELDS_BY_PROFILE[name])
        static_dict = self._run_ampp("csv")
        self.assertEqual(static_dict["ampp_version"], "1.0")
        self.assertIn("profile_0(active=True,configured=True,writable=False)", static_dict["ampp"])
        self.assertIn("profile_2(active=False,configured=True,writable=True)", static_dict["ampp"])

    def test_unconfigured_writable_profile_reports_no_fields(self):
        unconfigured = copy.deepcopy(_PROFILES) + [
            {
                "name": "profile_3",
                "index": 3,
                "is_active": False,
                "is_writable": True,
                "is_configured": False,
            }
        ]
        self.holder["get_profiles"] = lambda: ("1.0", unconfigured)
        static_dict = self._run_ampp("human")
        ampp_text = static_dict["ampp"]
        self.assertIn(" [3] profile_3 (writable, unconfigured)", ampp_text)

    def test_field_with_empty_unit_displayed_without_extra_formatting(self):
        self.holder["get_profiles"] = lambda: ("1.0", copy.deepcopy(_PROFILES))
        self.holder["get_fields"] = lambda _name: [
            {
                "name": "SomeCount",
                "unit": "",
                "value": 5,
                "limit_min": 0,
                "limit_max": 0,
                "has_limits": False,
            }
        ]
        static_dict = self._run_ampp("human")
        self.assertIn("SomeCount: 5 ", static_dict["ampp"])


if __name__ == "__main__":
    unittest.main()
