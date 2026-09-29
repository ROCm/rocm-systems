#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for how ``amd-smi set --compute-partition`` validates its value.

Builds the real ``set`` subparser and drives the real accelerator-choice helper
with the library query and effective UID faked, so no GPU hardware is needed.
Covers GPUs without accelerator partition profiles (for example MI210): run as
root, the option must be reported as not supported rather than as needing sudo.
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

# Profile config as returned for an MI300-class GPU and for a GPU without
# accelerator partitioning (the library reports AMDSMI_STATUS_NOT_SUPPORTED).
_PARTITION_PROFILES = {
    "profile_indices": ["0", "1", "2"],
    "profile_types": ["SPX", "DPX", "CPX"],
    "memory_caps": [{}, {}, {}],
}
_NO_PROFILES = {"profile_indices": [], "profile_types": [], "memory_caps": []}
_ROOT = 0
_USER = 1000


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_cli_modules():
    """Import the CLI helpers and parser with their library imports stubbed.

    ``amdsmi_init`` initializes the library on import, so it is replaced with the
    names the helpers bind. ``sys.modules`` and ``sys.path`` are changed only for
    the imports so stubs do not leak into sibling suites.
    """
    saved = {name: sys.modules.pop(name, None) for name in _SWAPPED_MODULES}
    added_path = _CLI_DIR not in sys.path
    try:
        interface = types.SimpleNamespace(
            AmdSmiPtlData=enum.Enum("AmdSmiPtlData", "INVALID I8 F16 BF16 F32 F64")
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
    return helpers_mod, parser_mod


class _FakeHelpers:
    """Helper surface the parser reads to build ``set`` on a Linux baremetal GPU.

    ``get_accelerator_choices_types_indices`` is the real helper (bound in
    setUpClass); only the library profile query below is faked.
    """

    def __init__(self, profiles):
        self._profiles = profiles

    def get_accelerator_partition_profile_config(self):
        return self._profiles

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
        helpers_mod, cls.parser_mod = _import_cli_modules()
        cls.exceptions = cls.parser_mod.amdsmi_cli_exceptions
        _FakeHelpers.get_accelerator_choices_types_indices = (
            helpers_mod.AMDSMIHelpers.get_accelerator_choices_types_indices
        )

    def _make_parser(self, euid, profiles, value="spx"):
        """Build the parser as amdsmi_cli.py does for ``amd-smi set -C <value>``.

        The CLI lower-cases the value before parsing; choices are read at build.
        """
        argv = ["amd-smi", "set", "--compute-partition", value]
        parser_cls = self.parser_mod.AMDSMIParser
        callbacks = {
            name: (lambda _args: None)
            for name in inspect.signature(parser_cls.__init__).parameters
            if name not in ("self", "sys_argv", "helpers")
        }
        with mock.patch("os.geteuid", return_value=euid):
            parser = parser_cls(**callbacks, sys_argv=argv, helpers=_FakeHelpers(profiles))
        return parser, argv

    def _parse(self, euid, profiles, value="spx"):
        parser, argv = self._make_parser(euid, profiles, value)
        # Invalid values print the valid list and name sys.argv[1] in the error.
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return parser.parse_args(argv[1:])

    def _parse_error(self, euid, profiles, value="spx"):
        with self.assertRaises(self.exceptions.AmdSmiException) as ctx:
            self._parse(euid, profiles, value)
        return ctx.exception

    def _compute_partition_help(self, euid, profiles):
        parser, _ = self._make_parser(euid, profiles)
        for action in parser.subparsers.choices["set"]._actions:
            if "--compute-partition" in action.option_strings:
                return action.help
        self.fail("--compute-partition is not registered on the set subcommand")

    def test_root_without_profiles_reports_not_supported(self):
        # A GPU without accelerator partition profiles (e.g. MI210), run as root:
        # the option is unsupported (-8); asking for sudo (-11) would mislead.
        for value in ("spx", "0"):  # a type or a profile index
            with self.subTest(value=value):
                exc = self._parse_error(_ROOT, _NO_PROFILES, value)
                self.assertIsInstance(exc, self.exceptions.AmdSmiParameterNotSupportedException)
                self.assertEqual(exc.value, -8)
                self.assertIn("'--compute-partition' is not supported", str(exc))

    def test_non_root_reports_elevation_required(self):
        # Profiles can only be read as root, so non-root keeps the sudo hint on
        # every GPU, and the library is never asked for profiles.
        for profiles in (_PARTITION_PROFILES, _NO_PROFILES):
            with self.subTest(profiles=profiles["profile_types"]):
                with mock.patch.object(
                    _FakeHelpers, "get_accelerator_partition_profile_config"
                ) as q:
                    exc = self._parse_error(_USER, profiles)
                self.assertIsInstance(exc, self.exceptions.AmdSmiPermissionDeniedException)
                self.assertEqual(exc.value, -11)
                q.assert_not_called()

    def test_root_with_profiles_accepts_type_or_index(self):
        for value, expected in (("spx", "SPX"), ("cpx", "CPX"), ("1", "1")):
            with self.subTest(value=value):
                args = self._parse(_ROOT, _PARTITION_PROFILES, value)
                self.assertEqual(args.compute_partition, expected)

    def test_root_with_profiles_rejects_unlisted_value(self):
        # A type the GPU does not offer stays an invalid value (-5), not -8.
        exc = self._parse_error(_ROOT, _PARTITION_PROFILES, "qpx")
        self.assertIsInstance(exc, self.exceptions.AmdSmiInvalidParameterValueException)
        self.assertEqual(exc.value, -5)

    def test_help_lists_choices_or_na(self):
        # Without choices the help must read "N/A", not the characters of "N/A".
        for euid, profiles in ((_USER, _PARTITION_PROFILES), (_ROOT, _NO_PROFILES)):
            with self.subTest(euid=euid, profiles=profiles["profile_types"]):
                help_text = self._compute_partition_help(euid, profiles)
                self.assertIn("\n\tN/A.\n", help_text)
                self.assertNotIn("N, /, A", help_text)
        help_text = self._compute_partition_help(_ROOT, _PARTITION_PROFILES)
        self.assertIn("\n\tSPX, DPX, CPX, 0, 1, 2.\n", help_text)


if __name__ == "__main__":
    unittest.main()
