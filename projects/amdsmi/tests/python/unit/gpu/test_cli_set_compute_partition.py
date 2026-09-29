#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for how ``amd-smi set --compute-partition`` validates its value.

Builds the real ``set`` subparser and runs the real accelerator partition helpers
against a stubbed library call and effective UID, so no GPU hardware is needed.
Covers GPUs without accelerator partition profiles (for example MI210): run as
root, the option must be reported as not supported rather than as needing sudo,
while any other failed profile query keeps its library error.
"""

import contextlib
import enum
import importlib.util
import inspect
import io
import os
import sys
import types
import unittest
from unittest import mock

# common.common bootstraps the installed amdsmi package; it is only needed to
# locate the installed CLI when this file does not run from a source checkout.
try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)
_CLI_DIR = next(
    (
        cli_dir
        for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR)
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "amdsmi_parser.py"))
    ),
    None,
)
_SWAPPED_MODULES = (
    "amdsmi",
    "amdsmi_init",
    "amdsmi_helpers",
    "amdsmi_cli_exceptions",
    "BDF",
    "_version",
)

_STATUS_NOT_SUPPORTED = 2  # AMDSMI_STATUS_NOT_SUPPORTED
_STATUS_BUSY = 30  # AMDSMI_STATUS_BUSY
_ROOT = 0
_USER = 1000
_CP = "--compute-partition"


class _FakeLibraryException(Exception):
    """Stand-in for ``AmdSmiLibraryException``; the helper reads ``err_code``."""

    def __init__(self, err_code):
        super().__init__(f"AMDSMI status {err_code}")
        self.err_code = err_code


def _profile_config(profile_types):
    """Library result for a GPU whose profiles 0..N-1 are ``profile_types``."""
    return {
        "num_profiles": len(profile_types),
        "profiles": [
            {"profile_index": i, "profile_type": t, "memory_caps": {}}
            for i, t in enumerate(profile_types)
        ],
    }


_PARTITIONED = _profile_config(["SPX", "DPX", "CPX"])  # MI300-class GPU


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_cli_modules():
    """Import the CLI helpers and parser; returns them and the stub library interface.

    ``amdsmi_init`` initializes the library on import, so it is replaced with the
    names the helpers bind. ``sys.modules`` and ``sys.path`` are changed only for
    the imports so stubs do not leak into sibling suites.
    """
    saved = {name: sys.modules.pop(name, None) for name in _SWAPPED_MODULES}
    added_path = _CLI_DIR not in sys.path
    try:
        interface = types.SimpleNamespace(
            AmdSmiPtlData=enum.Enum("AmdSmiPtlData", "INVALID I8 F16 BF16 F32 F64"),
            AmdSmiLibraryException=_FakeLibraryException,
            amdsmi_wrapper=types.SimpleNamespace(AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED),
            amdsmi_get_processor_handles=lambda: ["gpu0"],
            amdsmi_get_gpu_accelerator_partition_profile_config=None,  # patched per test
        )
        amdsmi_stub = types.ModuleType("amdsmi")
        amdsmi_stub.amdsmi_interface = interface
        sys.modules["amdsmi"] = amdsmi_stub
        init_stub = types.ModuleType("amdsmi_init")
        init_stub.AMD_VENDOR_ID = 0x1002
        init_stub.amdsmi_interface = interface
        init_stub.amdsmi_exception = types.SimpleNamespace()
        sys.modules["amdsmi_init"] = init_stub
        if not os.path.isfile(os.path.join(_CLI_DIR, "_version.py")):  # generated at build
            version_stub = types.ModuleType("_version")
            version_stub.__version__ = "0.0.0-test"
            sys.modules["_version"] = version_stub
        if added_path:
            sys.path.insert(0, _CLI_DIR)
        helpers_mod = _load_module("amdsmi_helpers", os.path.join(_CLI_DIR, "amdsmi_helpers.py"))
        sys.modules["amdsmi_helpers"] = helpers_mod
        parser_mod = _load_module(
            "amdsmi_parser_under_test", os.path.join(_CLI_DIR, "amdsmi_parser.py")
        )
    finally:
        if added_path:
            sys.path.remove(_CLI_DIR)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return helpers_mod, parser_mod, interface


class _FakeHelpers:
    """Helper surface the parser reads to build ``set`` on a Linux baremetal GPU.

    The two accelerator partition helpers are the real ones, bound in setUpClass.
    """

    def is_amdgpu_initialized(self):
        return True

    def is_ainic_initialized(self):
        return False

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def is_amd_hsmp_initialized(self):
        return False

    def get_gpu_choices(self):
        return {"0": "gpu0"}, "0"

    def is_linux(self):
        return True

    def is_baremetal(self):
        return True

    def is_hypervisor(self):
        return False

    def os_info(self):
        return "Linux Baremetal"

    def get_rocm_version(self):
        return "N/A"

    def get_output_format(self):
        return "human"

    def get_fan_support(self):
        return "0-255 or 0-100%"

    def get_perf_levels(self):
        return ["AUTO", "LOW", "HIGH", "UNKNOWN"], [0, 1, 2, 3]

    def get_power_profiles(self):
        return ["COMPUTE_MASK"]

    def get_memory_partition_types(self):
        return ["NPS1", "NPS2", "NPS4", "NPS8"]

    def get_soc_pstates(self):
        return ["N/A"]

    def get_xgmi_plpd_policies(self):
        return ["N/A"]

    def get_ptl_values(self):
        return ["I8", "F16", "INVALID"], ["I8", "F16", "INVALID"]

    def get_power_caps(self):
        return 0, 0, 0, 0


@unittest.skipIf(_CLI_DIR is None, "amd-smi CLI not found (source or installed)")
class TestSetComputePartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers_mod, cls.parser_mod, cls.interface = _import_cli_modules()
        cls.exceptions = cls.parser_mod.amdsmi_cli_exceptions
        for name in (
            "get_accelerator_partition_profile_config",
            "get_accelerator_choices_types_indices",
        ):
            setattr(_FakeHelpers, name, getattr(helpers_mod.AMDSMIHelpers, name))

    def _make_parser(self, euid, library, args):
        """Build the parser as amdsmi_cli.py does for ``amd-smi set <args>``.

        ``library`` is the profile-config result or the exception the library
        raises; it is read at build time. Returns (parser, library call mock).
        """
        parser_cls = self.parser_mod.AMDSMIParser
        callbacks = {
            name: (lambda _args: None)
            for name in inspect.signature(parser_cls.__init__).parameters
            if name not in ("self", "sys_argv", "helpers")
        }
        outcome = (
            {"side_effect": library}
            if isinstance(library, Exception)
            else {"return_value": library}
        )
        query = mock.patch.object(
            self.interface, "amdsmi_get_gpu_accelerator_partition_profile_config", **outcome
        )
        with mock.patch("os.geteuid", return_value=euid), query as query_mock:
            parser = parser_cls(
                **callbacks, sys_argv=["amd-smi", "set", *args], helpers=_FakeHelpers()
            )
        return parser, query_mock

    @staticmethod
    def _run(parser, args):
        argv = ["amd-smi", "set", *args]  # the CLI lower-cases values before parsing
        # Invalid values print the valid list and name sys.argv[1] in the error.
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return parser.parse_args(argv[1:])

    def _parse(self, euid, library, *args):
        parser, _ = self._make_parser(euid, library, args)
        return self._run(parser, args)

    def _parse_error(self, euid, library, *args):
        with self.assertRaises(self.exceptions.AmdSmiException) as ctx:
            self._parse(euid, library, *args)
        return ctx.exception

    def _compute_partition_help(self, euid, library):
        parser, _ = self._make_parser(euid, library, (_CP, "spx"))
        for action in parser.subparsers.choices["set"]._actions:
            if _CP in action.option_strings:
                return action.help
        self.fail(f"{_CP} is not registered on the set subcommand")

    def test_root_without_profiles_reports_not_supported(self):
        # A GPU without accelerator partition profiles (e.g. MI210), run as root:
        # the option is unsupported (-8); asking for sudo (-11) would mislead.
        for library in (_FakeLibraryException(_STATUS_NOT_SUPPORTED), _profile_config([])):
            for value in ("spx", "0"):  # a type or a profile index
                with self.subTest(library=library, value=value):
                    exc = self._parse_error(_ROOT, library, _CP, value)
                    self.assertIsInstance(exc, self.exceptions.AmdSmiParameterNotSupportedException)
                    self.assertEqual(exc.value, -8)
                    self.assertIn(f"'{_CP}' is not supported", str(exc))

    def test_root_query_failure_keeps_library_error(self):
        # Only NOT_SUPPORTED means no partitioning. Another failed profile query
        # surfaces its own library error, and only when -C is used: building
        # `set` for any other option still works.
        with self.assertRaises(_FakeLibraryException) as ctx:
            self._parse(_ROOT, _FakeLibraryException(_STATUS_BUSY), _CP, "spx")
        self.assertEqual(ctx.exception.err_code, _STATUS_BUSY)
        args = self._parse(_ROOT, _FakeLibraryException(_STATUS_BUSY), "--memory-partition", "nps1")
        self.assertEqual(args.memory_partition, "NPS1")

    def test_non_root_reports_elevation_required(self):
        # Profiles can only be read as root, so non-root keeps the sudo hint on
        # every GPU, and the library is never asked for profiles.
        for library in (_PARTITIONED, _FakeLibraryException(_STATUS_NOT_SUPPORTED)):
            with self.subTest(library=library):
                parser, query = self._make_parser(_USER, library, (_CP, "spx"))
                with self.assertRaises(self.exceptions.AmdSmiPermissionDeniedException) as ctx:
                    self._run(parser, (_CP, "spx"))
                self.assertEqual(ctx.exception.value, -11)
                query.assert_not_called()

    def test_root_with_profiles_accepts_type_or_index(self):
        for value, expected in (("spx", "SPX"), ("cpx", "CPX"), ("1", "1")):
            with self.subTest(value=value):
                args = self._parse(_ROOT, _PARTITIONED, _CP, value)
                self.assertEqual(args.compute_partition, expected)

    def test_root_with_profiles_rejects_unlisted_value(self):
        # A type the GPU does not offer stays an invalid value (-5), not -8.
        exc = self._parse_error(_ROOT, _PARTITIONED, _CP, "qpx")
        self.assertIsInstance(exc, self.exceptions.AmdSmiInvalidParameterValueException)
        self.assertEqual(exc.value, -5)

    def test_help_lists_choices_or_na(self):
        # Without choices the help must read "N/A", not the characters of "N/A".
        for euid, library in (
            (_USER, _PARTITIONED),
            (_ROOT, _FakeLibraryException(_STATUS_NOT_SUPPORTED)),
            (_ROOT, _FakeLibraryException(_STATUS_BUSY)),
        ):
            with self.subTest(euid=euid, library=library):
                help_text = self._compute_partition_help(euid, library)
                self.assertIn("\n\tN/A.\n", help_text)
                self.assertNotIn("N, /, A", help_text)
        help_text = self._compute_partition_help(_ROOT, _PARTITIONED)
        self.assertIn("\n\tSPX, DPX, CPX, 0, 1, 2.\n", help_text)


if __name__ == "__main__":
    unittest.main()
