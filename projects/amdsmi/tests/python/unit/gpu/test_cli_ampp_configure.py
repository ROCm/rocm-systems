#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Mock-based unit tests for the ``amd-smi set --ampp-configure`` CLI flag.

These stub the ``amdsmi`` package (and the small set of sibling CLI modules
``amdsmi_parser.py``/``set_value.py`` import at module scope) so the tests run
without GPU hardware or the compiled ``amdsmi`` package. They lock in two
layers of the CONFIGURE CLI surface added alongside the AMPP feature:

* ``AMDSMIParser._ampp_configure_options`` (the ``argparse.Action`` behind
  ``--ampp-configure PROFILE_NAME KEY=VALUE ...``): valid input parses into
  the ``(profile_name, fields)`` namedtuple the dispatch layer expects, and
  malformed tokens (missing PROFILE_NAME, missing "=", empty key, non-integer
  value) raise ``AmdSmiInvalidParameter(Value)?Exception`` instead of
  silently accepting garbage.
* ``SetValueCommands.set_gpu``'s CONFIGURE dispatch block: a parsed
  ``ampp_configure_args`` reaches ``amdsmi_configure_ampp_profile(...)`` with
  the right arguments, DRY_RUN-style (the C library call itself is stubbed,
  so no real sysfs write occurs), and a driver-side INVAL is surfaced as a
  readable per-GPU error listing the writable profiles.
"""

import collections
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py")
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py")
HELPERS_PATH = os.path.join(_CLI_DIR, "amdsmi_helpers.py")

# AMDSMI_STATUS_* sentinels used by the stubbed library-error paths.
_STATUS_INVAL = 1
_STATUS_NO_PERM = 10


class _FakeInitFlagsForHelpersImport:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeClkTypeForHelpersImport:
    """Enough ``AmdSmiClkType`` members for ``amdsmi_helpers`` to import."""

    SYS = "SYS"
    MEM = "MEM"
    DF = "DF"
    SOC = "SOC"
    DCEF = "DCEF"
    VCLK0 = "VCLK0"
    VCLK1 = "VCLK1"
    DCLK0 = "DCLK0"
    DCLK1 = "DCLK1"


def _import_real_amdsmi_helpers():
    """Load the REAL ``AMDSMIHelpers`` class (same file
    ``test_amdsmi_helpers_ampp.py`` pins down directly) so
    ``_FakeAMDSMIHelpers`` below can delegate ``is_valid_ampp_field_name``/
    ``parse_ampp_field_value`` to it instead of hand-reimplementing them --
    otherwise this file's copy could silently drift from the shipped
    validators. Uses a throwaway module name so it never collides with the
    per-test fake ``amdsmi_helpers`` module installed by
    ``_install_fake_modules()``.
    """
    saved = {
        name: sys.modules.pop(name, None)
        for name in ("amdsmi", "amdsmi.amdsmi_interface", "amdsmi.amdsmi_exception", "amdsmi_init")
    }
    try:
        amdsmi_pkg = types.ModuleType("amdsmi")
        interface = types.ModuleType("amdsmi.amdsmi_interface")
        exception = types.ModuleType("amdsmi.amdsmi_exception")

        interface.amdsmi_wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")
        interface.AmdSmiInitFlags = _FakeInitFlagsForHelpersImport
        interface.AmdSmiClkType = _FakeClkTypeForHelpersImport
        interface.AmdSmiLibraryException = _FakeLibraryException
        interface.AmdSmiParameterException = _FakeLibraryException
        interface.AMDSMI_MAX_STRING_LENGTH = 256
        interface.amdsmi_init = lambda _flag: None
        interface.amdsmi_shut_down = lambda: None
        interface.amdsmi_get_processor_handles = lambda: []

        exception.AmdSmiLibraryException = _FakeLibraryException
        exception.AmdSmiParameterException = _FakeLibraryException

        amdsmi_pkg.amdsmi_interface = interface
        amdsmi_pkg.amdsmi_exception = exception

        sys.modules["amdsmi"] = amdsmi_pkg
        sys.modules["amdsmi.amdsmi_interface"] = interface
        sys.modules["amdsmi.amdsmi_exception"] = exception

        spec = importlib.util.spec_from_file_location("_real_amdsmi_helpers", HELPERS_PATH)
        module = importlib.util.module_from_spec(spec)
        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)
        spec.loader.exec_module(module)
        return module.AMDSMIHelpers
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class _FakeLibraryException(Exception):
    """Stand-in for ``amdsmi_exception.AmdSmiLibraryException``."""

    def __init__(self, err_code=_STATUS_INVAL, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self, detailed=True):
        return self._message


_RealAMDSMIHelpers = _import_real_amdsmi_helpers()


def _install_fake_modules():
    """Register a stub ``amdsmi`` package plus the sibling CLI modules that
    ``amdsmi_parser.py`` and ``set_value.py`` import at module scope.

    Returns the fake ``amdsmi_interface`` module so individual tests can swap
    in per-case ``amdsmi_configure_ampp_profile``/``amdsmi_get_ampp_profiles``
    behavior.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    wrapper.AMDSMI_STATUS_NO_PERM = _STATUS_NO_PERM
    interface.amdsmi_wrapper = wrapper
    # Constants set_value.py binds at import time; the values are irrelevant
    # to the ampp-configure paths exercised here.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    interface.AMDSMI_MAX_STRING_LENGTH = 256
    # Overwritten per-test; the default keeps the CONFIGURE path a no-op.
    interface.amdsmi_configure_ampp_profile = lambda *a, **k: None
    interface.amdsmi_get_ampp_profiles = lambda _h: ("1.0", [])

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper

    # amdsmi_parser.py and set_value.py additionally import this sibling
    # module by bare name at module scope; the tests here never construct a
    # real AMDSMIParser/heavyweight AMDSMIHelpers (only the unbound
    # ``_ampp_configure_options`` method and the two static AMPP validators
    # are ever touched), so a minimal stand-in is enough -- but the two
    # validators themselves delegate to the real implementation (pinned
    # directly in test_amdsmi_helpers_ampp.py) so this file can't drift out
    # of sync with it.
    class _FakeAMDSMIHelpers:
        is_valid_ampp_field_name = staticmethod(_RealAMDSMIHelpers.is_valid_ampp_field_name)
        parse_ampp_field_value = staticmethod(_RealAMDSMIHelpers.parse_ampp_field_value)

    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = _FakeAMDSMIHelpers
    sys.modules["amdsmi_helpers"] = helpers_mod

    version_mod = types.ModuleType("_version")
    version_mod.__version__ = "0.0.0-test"
    sys.modules["_version"] = version_mod

    return interface


def _load_module(name, path):
    if _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SAVED_MODULE_NAMES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi.amdsmi_wrapper",
    "amdsmi_helpers",
    "_version",
)


class _StubHelpersForParser:
    """Minimal ``self.helpers`` stand-in used only for
    ``AMDSMIParser._ampp_configure_options``, which reads
    ``self.helpers.get_output_format()``."""

    def get_output_format(self):
        return "human"


class _StubParserSelf:
    """Minimal stand-in for the ``AMDSMIParser`` instance that
    ``_ampp_configure_options`` is normally called on (an unbound method
    that only reads ``self.helpers``)."""

    def __init__(self):
        self.helpers = _StubHelpersForParser()


class TestAmppConfigureParserAction(unittest.TestCase):
    """Unit tests for the ``--ampp-configure`` custom argparse Action."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest(f"amd-smi CLI amdsmi_parser.py not found at {PARSER_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in _SAVED_MODULE_NAMES}
        _install_fake_modules()
        cls.parser_module = _load_module("amdsmi_parser_under_test", PARSER_PATH)
        # amdsmi_parser.py resolves this via a plain ``import
        # amdsmi_cli_exceptions`` (not spec_from_file_location), so the
        # exceptions it actually raises are instances of the classes in
        # *this* sys.modules entry -- reuse it rather than loading a second,
        # class-identity-incompatible copy.
        cls.exceptions_module = sys.modules["amdsmi_cli_exceptions"]

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _make_action(self):
        # ``_ampp_configure_options`` is an ordinary (unbound) method on
        # AMDSMIParser: it only touches ``self.helpers``, so a minimal stub
        # avoids constructing a full parser (which needs a live amdgpu/HSMP
        # environment).
        action_cls = self.parser_module.AMDSMIParser._ampp_configure_options(_StubParserSelf())
        return action_cls(option_strings=["--ampp-configure"], dest="ampp_configure", nargs="+")

    def _invoke(self, values):
        action = self._make_action()
        namespace = types.SimpleNamespace(ampp_configure=None)
        # sys.argv[1] is read for the exception's "command" field; keep argv
        # stable across the whole suite regardless of the runner invocation.
        saved_argv = sys.argv[:]
        sys.argv = ["amd-smi", "set"]
        try:
            action(parser=None, namespace=namespace, values=values)
        finally:
            sys.argv = saved_argv
        return namespace.ampp_configure

    def test_profile_name_only_raises(self):
        # The driver's config/commit rejects -EINVAL if no field was ever
        # staged for this attempt; the CLI now requires at least one
        # KEY=VALUE field up front instead of accepting a bare PROFILE_NAME
        # and later surfacing an uninterpreted library error.
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterException):
            self._invoke(["profile_2"])

    def test_profile_name_and_single_field_parses(self):
        result = self._invoke(["profile_2", "PPT0_Limit=300"])
        self.assertEqual(result.profile_name, "profile_2")
        self.assertEqual(result.fields, [{"name": "PPT0_Limit", "value": 300}])

    def test_multiple_fields_parse_in_order(self):
        result = self._invoke(["profile_3", "PPT0_Limit=300", "PPT1_Limit=150"])
        self.assertEqual(result.profile_name, "profile_3")
        self.assertEqual(
            result.fields,
            [{"name": "PPT0_Limit", "value": 300}, {"name": "PPT1_Limit", "value": 150}],
        )

    def test_missing_profile_name_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterException):
            self._invoke([])

    def test_field_missing_equals_sign_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["profile_2", "PPT0_Limit"])

    def test_field_empty_key_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["profile_2", "=300"])

    def test_field_non_integer_value_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["profile_2", "PPT0_Limit=not_a_number"])

    def test_field_oversized_name_raises(self):
        oversized_key = "K" * 256
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["profile_2", f"{oversized_key}=300"])

    def test_field_out_of_int64_range_value_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["profile_2", f"PPT0_Limit={2**63}"])

    def test_file_path_parses(self):
        result = self._invoke(["@profiles.json"])
        self.assertIsNone(result.profile_name)
        self.assertIsNone(result.fields)
        self.assertEqual(result.file_path, "profiles.json")

    def test_file_path_with_extra_token_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["@profiles.json", "PPT0_Limit=300"])

    def test_file_path_empty_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["@"])

    def test_profile_name_and_field_still_has_null_file_path(self):
        result = self._invoke(["profile_2", "PPT0_Limit=300"])
        self.assertIsNone(result.file_path)

    def test_profile_name_with_lone_surrogate_raises(self):
        # A lone UTF-16 surrogate reaches argv via surrogateescape and is not
        # UTF-8-encodable, so amdsmi_configure_ampp_profile()'s name.encode()
        # would raise an unhandled UnicodeEncodeError without this check.
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["\ud800", "PPT0_Limit=300"])

    def test_profile_name_oversized_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._invoke(["P" * 256, "PPT0_Limit=300"])

    def test_activate_profile_name_validator_accepts_valid_name(self):
        parser_self = _StubParserSelf()
        validator = self.parser_module.AMDSMIParser._valid_ampp_profile_name
        self.assertEqual(validator(parser_self, "profile_5"), "profile_5")

    def test_activate_profile_name_validator_rejects_invalid_names(self):
        parser_self = _StubParserSelf()
        validator = self.parser_module.AMDSMIParser._valid_ampp_profile_name
        saved_argv = sys.argv[:]
        sys.argv = ["amd-smi", "set"]
        try:
            for bad_name in ("", "\ud800", "P" * 256):
                with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
                    validator(parser_self, bad_name)
        finally:
            sys.argv = saved_argv


class _RecordingLogger:
    """Minimal ``self.logger`` stub capturing ``store_output`` payloads."""

    def __init__(self):
        self.format = "human"
        self.outputs = []

    def store_output(self, device, key, value):
        self.outputs.append((device, key, value))

    def print_output(self, *args, **kwargs):
        pass

    def clear_multiple_devices_output(self):
        pass

    def last_output(self, key):
        for _device, stored_key, value in reversed(self.outputs):
            if stored_key == key:
                return value
        return None


class _StubHelpersForSetGpu:
    """Minimal ``self.helpers`` stub for the GPU ``set`` path."""

    def check_required_groups(self):
        pass

    def handle_gpus(self, args, logger, func):
        # Single device: never recurse, echo the resolved handle back.
        return (False, args.gpu)

    def is_baremetal(self):
        return True

    def get_gpu_id_from_device_handle(self, handle):
        return 0


_AmppConfigureArgs = collections.namedtuple(
    "_AmppConfigureArgs", ["profile_name", "fields", "file_path"], defaults=[None]
)


class TestSetGpuAmppConfigureCallSite(unittest.TestCase):
    """Call-site tests for the ``set_gpu`` CONFIGURE dispatch branch.

    Complements ``TestAmppConfigureParserAction`` (parser layer in isolation)
    by driving the real ``set_gpu`` dispatch with the C library stubbed,
    proving the parsed ``ampp_configure_args`` namedtuple reaches
    ``amdsmi_configure_ampp_profile`` with the staged fields, and that a
    driver-side INVAL surfaces a readable per-GPU error instead of
    propagating raw.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI set_value.py not found at {SET_VALUE_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in _SAVED_MODULE_NAMES}
        cls.interface = _install_fake_modules()
        cls.interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"
        cls.module = _load_module("set_value_under_test_ampp", SET_VALUE_PATH)

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _run_set_gpu(self, ampp_configure):
        logger = _RecordingLogger()
        cmd = self.module.SetValueCommands()
        cmd.logger = logger
        cmd.helpers = _StubHelpersForSetGpu()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]

        args = types.SimpleNamespace(
            gpu="gpu0",
            fan=None,
            perf_level=None,
            profile=None,
            perf_determinism=None,
            compute_partition=None,
            memory_partition=None,
            power_cap=None,
            soc_pstate=None,
            xgmi_plpd=None,
            process_isolation=None,
            clk_limit=None,
            clk_level=None,
            ptl_status=None,
            ptl_format=None,
            mem_carveout=None,
            compute_partition_mem_alloc_mode=None,
            ampp_activate=None,
            ampp_configure=ampp_configure,
        )
        cmd.set_gpu(args)
        return logger

    def test_configure_success_calls_library_with_configure_op_and_fields(self):
        set_calls = []
        self.interface.amdsmi_configure_ampp_profile = lambda gpu, name, fields=None: (
            set_calls.append((gpu, name, fields))
        )

        ampp_configure = _AmppConfigureArgs("profile_2", [{"name": "PPT0_Limit", "value": 300}])
        logger = self._run_set_gpu(ampp_configure)

        self.assertEqual(len(set_calls), 1)
        gpu, name, fields = set_calls[0]
        self.assertEqual(gpu, "gpu0")
        self.assertEqual(name, "profile_2")
        self.assertEqual(fields, [{"name": "PPT0_Limit", "value": 300}])

        message = logger.last_output("ampp_configure")
        self.assertIn("Successfully configured AMPP profile profile_2", message)
        self.assertIn("PPT0_Limit=300", message)

    def test_configure_with_no_fields_reports_no_fields_staged(self):
        self.interface.amdsmi_configure_ampp_profile = lambda *a, **k: None
        ampp_configure = _AmppConfigureArgs("profile_2", [])
        logger = self._run_set_gpu(ampp_configure)
        message = logger.last_output("ampp_configure")
        self.assertIn("no fields staged", message)

    def test_configure_inval_reports_writable_profiles(self):
        def _raise(*_a, **_k):
            raise _FakeLibraryException(_STATUS_INVAL, "Invalid parameters")

        self.interface.amdsmi_configure_ampp_profile = _raise
        self.interface.amdsmi_get_ampp_profiles = lambda _h: (
            "1.0",
            [
                {"name": "profile_0", "is_writable": False},
                {"name": "profile_2", "is_writable": True},
            ],
        )

        ampp_configure = _AmppConfigureArgs("profile_2", [{"name": "not_a_real_field", "value": 1}])
        logger = self._run_set_gpu(ampp_configure)
        message = logger.last_output("ampp_configure")
        self.assertIn("Unable to configure AMPP profile profile_2", message)

    def test_configure_no_perm_raises_permission_error(self):
        def _raise(*_a, **_k):
            raise _FakeLibraryException(_STATUS_NO_PERM, "Permission denied")

        self.interface.amdsmi_configure_ampp_profile = _raise
        ampp_configure = _AmppConfigureArgs("profile_2", [{"name": "PPT0_Limit", "value": 300}])
        with self.assertRaises(PermissionError):
            self._run_set_gpu(ampp_configure)


class TestSetGpuAmppActivateCallSite(unittest.TestCase):
    """Call-site tests for the ``set_gpu`` ACTIVATE dispatch branch.

    Mirrors ``TestSetGpuAmppConfigureCallSite`` for the sibling
    ``--ampp-activate`` dispatch: proves the profile name reaches
    ``amdsmi_activate_ampp_profile`` unchanged, that a driver-side error
    surfaces the writable-profile listing, and that NO_PERM raises
    ``PermissionError`` instead of a stored per-GPU message.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI set_value.py not found at {SET_VALUE_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in _SAVED_MODULE_NAMES}
        cls.interface = _install_fake_modules()
        cls.interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"
        cls.module = _load_module("set_value_under_test_ampp_activate", SET_VALUE_PATH)

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _run_set_gpu(self, ampp_activate):
        logger = _RecordingLogger()
        cmd = self.module.SetValueCommands()
        cmd.logger = logger
        cmd.helpers = _StubHelpersForSetGpu()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]

        args = types.SimpleNamespace(
            gpu="gpu0",
            fan=None,
            perf_level=None,
            profile=None,
            perf_determinism=None,
            compute_partition=None,
            memory_partition=None,
            power_cap=None,
            soc_pstate=None,
            xgmi_plpd=None,
            process_isolation=None,
            clk_limit=None,
            clk_level=None,
            ptl_status=None,
            ptl_format=None,
            mem_carveout=None,
            compute_partition_mem_alloc_mode=None,
            ampp_activate=ampp_activate,
            ampp_configure=None,
        )
        cmd.set_gpu(args)
        return logger

    def test_activate_success_calls_library_with_profile_name(self):
        activate_calls = []
        self.interface.amdsmi_activate_ampp_profile = lambda gpu, name: activate_calls.append(
            (gpu, name)
        )

        logger = self._run_set_gpu("profile_2")

        self.assertEqual(activate_calls, [("gpu0", "profile_2")])
        message = logger.last_output("ampp_activate")
        self.assertIn("Successfully activated AMPP profile profile_2", message)

    def test_activate_inval_reports_writable_profiles(self):
        def _raise(*_a, **_k):
            raise _FakeLibraryException(_STATUS_INVAL, "Invalid parameters")

        self.interface.amdsmi_activate_ampp_profile = _raise
        self.interface.amdsmi_get_ampp_profiles = lambda _h: (
            "1.0",
            [
                {"name": "profile_0", "is_writable": False},
                {"name": "profile_2", "is_writable": True},
            ],
        )

        logger = self._run_set_gpu("profile_2")
        message = logger.last_output("ampp_activate")
        self.assertIn("Unable to activate AMPP profile profile_2", message)

    def test_activate_no_perm_raises_permission_error(self):
        def _raise(*_a, **_k):
            raise _FakeLibraryException(_STATUS_NO_PERM, "Permission denied")

        self.interface.amdsmi_activate_ampp_profile = _raise
        with self.assertRaises(PermissionError):
            self._run_set_gpu("profile_2")


class TestSetGpuAmppConfigureFromFile(unittest.TestCase):
    """Tests for the ``--ampp-configure @<path>`` file-restore branch.

    Fixtures mirror the real ``amd-smi static --ampp --json`` output shape
    (``AMDSMILogger.combine_arrays_to_json``): a top-level ``"gpu_data"``
    list of per-GPU dicts, matched by the integer ``"gpu"`` key. The stubbed
    ``_StubHelpersForSetGpu.get_gpu_id_from_device_handle`` (and
    ``amdsmi_get_gpu_device_bdf``, set up in ``setUpClass``) resolve
    ``gpu0`` to gpu id ``0`` / BDF ``"0000:00:00.0"``.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI set_value.py not found at {SET_VALUE_PATH}")
        cls._saved_modules = {name: sys.modules.get(name) for name in _SAVED_MODULE_NAMES}
        cls.interface = _install_fake_modules()
        cls.interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"
        cls.module = _load_module("set_value_under_test_ampp_file", SET_VALUE_PATH)
        cls.exceptions_module = sys.modules["amdsmi_cli_exceptions"]

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self.addCleanup(lambda: os.unlink(self._tmp.name))

    def _write_json(self, data):
        json.dump(data, self._tmp)
        self._tmp.close()
        return self._tmp.name

    def _run_set_gpu(self, file_path):
        logger = _RecordingLogger()
        cmd = self.module.SetValueCommands()
        cmd.logger = logger
        cmd.helpers = _StubHelpersForSetGpu()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]

        args = types.SimpleNamespace(
            gpu="gpu0",
            fan=None,
            perf_level=None,
            profile=None,
            perf_determinism=None,
            compute_partition=None,
            memory_partition=None,
            power_cap=None,
            soc_pstate=None,
            xgmi_plpd=None,
            process_isolation=None,
            clk_limit=None,
            clk_level=None,
            ptl_status=None,
            ptl_format=None,
            mem_carveout=None,
            compute_partition_mem_alloc_mode=None,
            ampp_activate=None,
            ampp_configure=_AmppConfigureArgs(None, None, file_path),
        )
        cmd.set_gpu(args)
        return logger

    def test_restores_every_writable_configured_profile(self):
        set_calls = []
        self.interface.amdsmi_configure_ampp_profile = lambda gpu, name, fields=None: (
            set_calls.append((gpu, name, fields))
        )
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [
                                {"name": "profile_0", "is_writable": False, "fields": []},
                                {
                                    "name": "profile_5",
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit", "value": 550}],
                                },
                                {
                                    "name": "profile_6",
                                    "is_writable": True,
                                    "fields": [{"name": "MaxGfxclkFreq", "value": 1900}],
                                },
                            ],
                        },
                    }
                ]
            }
        )

        logger = self._run_set_gpu(file_path)

        self.assertEqual(len(set_calls), 2)
        self.assertEqual(
            set_calls[0], ("gpu0", "profile_5", [{"name": "PPT0_Limit", "value": 550}])
        )
        self.assertEqual(
            set_calls[1], ("gpu0", "profile_6", [{"name": "MaxGfxclkFreq", "value": 1900}])
        )

        results = logger.last_output("ampp_configure")
        self.assertEqual(len(results), 2)
        self.assertIn("profile_5: Successfully configured (PPT0_Limit=550)", results[0])
        self.assertIn("profile_6: Successfully configured (MaxGfxclkFreq=1900)", results[1])

    def test_no_writable_profiles_reports_message(self):
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [{"name": "profile_0", "is_writable": False, "fields": []}],
                        },
                    }
                ]
            }
        )
        logger = self._run_set_gpu(file_path)
        message = logger.last_output("ampp_configure")
        self.assertIn("No writable AMPP profiles", message)

    def test_missing_gpu_entry_raises(self):
        file_path = self._write_json({"gpu_data": [{"gpu": 1, "ampp": {"profiles": []}}]})
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu(file_path)

    def test_missing_file_raises(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu("/nonexistent/path/profiles.json")

    def test_invalid_json_raises(self):
        with open(self._tmp.name, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self._tmp.close()
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu(self._tmp.name)

    def test_top_level_not_gpu_data_shape_raises(self):
        # Old (never-correct) BDF-keyed top level, and a bare list, must both
        # be rejected with a clean error rather than an AttributeError/KeyError.
        file_path = self._write_json({"0000:00:00.0": {"ampp": {"profiles": []}}})
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu(file_path)

    def test_top_level_list_raises(self):
        file_path = self._write_json([{"gpu": 0, "ampp": {"profiles": []}}])
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu(file_path)

    def test_ampp_not_supported_string_raises_clean_error(self):
        # static.py legitimately emits "ampp": "N/A" (a string, not a dict)
        # when AMPP isn't supported/fetchable for a GPU.
        file_path = self._write_json({"gpu_data": [{"gpu": 0, "ampp": "N/A"}]})
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidFilePathException):
            self._run_set_gpu(file_path)

    def test_profiles_not_a_list_treated_as_no_writable_profiles(self):
        file_path = self._write_json(
            {"gpu_data": [{"gpu": 0, "ampp": {"version": "1.0", "profiles": "oops"}}]}
        )
        logger = self._run_set_gpu(file_path)
        message = logger.last_output("ampp_configure")
        self.assertIn("No writable AMPP profiles", message)

    def test_profile_missing_name_is_skipped(self):
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [
                                {
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit", "value": 550}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
        logger = self._run_set_gpu(file_path)
        message = logger.last_output("ampp_configure")
        self.assertIn("No writable AMPP profiles", message)

    def test_fields_not_a_list_is_skipped(self):
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [
                                {"name": "profile_5", "is_writable": True, "fields": "oops"}
                            ],
                        },
                    }
                ]
            }
        )
        logger = self._run_set_gpu(file_path)
        message = logger.last_output("ampp_configure")
        self.assertIn("No writable AMPP profiles", message)

    def test_field_missing_name_or_value_reports_malformed(self):
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [
                                {
                                    "name": "profile_5",
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit"}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
        logger = self._run_set_gpu(file_path)
        results = logger.last_output("ampp_configure")
        self.assertIn("Malformed 'fields' entry", results[0])

    def test_field_non_numeric_value_reports_malformed(self):
        # amdsmi_configure_ampp_profile() does int(field["value"]) internally
        # and raises a bare ValueError for a non-numeric value -- this must
        # be rejected before the library call, not left to crash the loop.
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "version": "1.0",
                            "profiles": [
                                {
                                    "name": "profile_5",
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit", "value": "not_a_number"}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
        logger = self._run_set_gpu(file_path)
        results = logger.last_output("ampp_configure")
        self.assertIn("Malformed 'fields' entry", results[0])

    def test_profile_name_with_lone_surrogate_reports_malformed_name(self):
        # A crafted JSON restore file can carry a lone UTF-16 surrogate;
        # amdsmi_configure_ampp_profile()'s name.encode() would raise an
        # unhandled UnicodeEncodeError mid-restore without this check.
        set_calls = []
        self.interface.amdsmi_configure_ampp_profile = lambda gpu, name, fields=None: (
            set_calls.append((gpu, name, fields))
        )
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "profiles": [
                                {
                                    "name": "\ud800",
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit", "value": 550}],
                                },
                                {
                                    "name": "profile_6",
                                    "is_writable": True,
                                    "fields": [{"name": "MaxGfxclkFreq", "value": 1900}],
                                },
                            ]
                        },
                    }
                ]
            }
        )

        logger = self._run_set_gpu(file_path)
        results = logger.last_output("ampp_configure")
        self.assertIn("Malformed profile 'name'", results[0])
        # The bad entry doesn't abort the restore of the remaining profiles.
        self.assertIn("profile_6: Successfully configured", results[1])
        self.assertEqual(len(set_calls), 1)
        self.assertEqual(set_calls[0][1], "profile_6")

    def test_partial_failure_reports_per_profile_result(self):
        def _configure(gpu, name, fields=None):
            if name == "profile_6":
                raise _FakeLibraryException(_STATUS_INVAL, "Invalid parameters")

        self.interface.amdsmi_configure_ampp_profile = _configure
        file_path = self._write_json(
            {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "ampp": {
                            "profiles": [
                                {
                                    "name": "profile_5",
                                    "is_writable": True,
                                    "fields": [{"name": "PPT0_Limit", "value": 550}],
                                },
                                {
                                    "name": "profile_6",
                                    "is_writable": True,
                                    "fields": [{"name": "MaxGfxclkFreq", "value": 1900}],
                                },
                            ]
                        },
                    }
                ]
            }
        )

        logger = self._run_set_gpu(file_path)
        results = logger.last_output("ampp_configure")
        self.assertIn("profile_5: Successfully configured", results[0])
        self.assertIn("profile_6:", results[1])
        self.assertIn("Unable to configure", results[1])

    def test_round_trips_real_static_json_output(self):
        # Build the file the same way static.py's static_gpu() + static()
        # actually do (static_dict shape + combine_arrays_to_json wrapping),
        # rather than a hand-fabricated fixture -- this is what would have
        # caught the original BDF-keyed-shape mismatch.
        static_dict = {"gpu": 0}
        static_dict["ampp"] = {
            "version": "1.0",
            "profiles": [
                {
                    "name": "profile_5",
                    "is_writable": True,
                    "fields": [{"name": "PPT0_Limit", "value": 550}],
                }
            ],
        }
        combined_json = {"gpu_data": [static_dict]}
        file_path = self._write_json(combined_json)

        set_calls = []
        self.interface.amdsmi_configure_ampp_profile = lambda gpu, name, fields=None: (
            set_calls.append((gpu, name, fields))
        )
        logger = self._run_set_gpu(file_path)
        self.assertEqual(len(set_calls), 1)
        self.assertEqual(
            set_calls[0], ("gpu0", "profile_5", [{"name": "PPT0_Limit", "value": 550}])
        )
        results = logger.last_output("ampp_configure")
        self.assertIn("profile_5: Successfully configured (PPT0_Limit=550)", results[0])


if __name__ == "__main__":
    unittest.main()
