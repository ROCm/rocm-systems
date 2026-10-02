#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi set --node-power-limit``.

* ``TestCliSetNodePowerLimit``: the node-wide dispatch block in ``set_value.py``,
  with ``self.helpers`` stubbed.
* ``TestValidateAndSetNodePowerLimit``: the real
  ``AMDSMIHelpers.validate_and_set_node_power_limit()`` with only the C library faked.
* ``TestSetValueParser``: ``-n`` registers on a 1VF guest, and the ``--gtt``/``--gpu``
  guard rewrites only argparse's missing ``--gpu`` value error.
"""

import argparse
import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""

_STATUS_NOT_SUPPORTED = 2
_STATUS_NO_PERM = 10
_STATUS_INVAL = 5
_STATUS_UNEXPECTED_SIZE = 42
_STATUS_UNEXPECTED_DATA = 43
_NPM_STATUS_DISABLED = 0
_NPM_STATUS_ENABLED = 1


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self, detailed=True):
        return self._message if detailed else self._message.split(" - ")[0]


def _install_fake_amdsmi():
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper",
        AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED,
        AMDSMI_STATUS_NO_PERM=_STATUS_NO_PERM,
        AMDSMI_STATUS_INVAL=_STATUS_INVAL,
        # Unused by this file's own tests, but amdsmi_helpers.AMDSMIHelpers's
        # CPER_DECODE_MESSAGES class body reads these off amdsmi_wrapper at
        # import time, so `import amdsmi_helpers` under this stub needs them
        # present or it dies with an AttributeError before any test runs.
        AMDSMI_STATUS_UNEXPECTED_SIZE=_STATUS_UNEXPECTED_SIZE,
        AMDSMI_STATUS_UNEXPECTED_DATA=_STATUS_UNEXPECTED_DATA,
        AMDSMI_NPM_STATUS_DISABLED=_NPM_STATUS_DISABLED,
        AMDSMI_NPM_STATUS_ENABLED=_NPM_STATUS_ENABLED,
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        AMDSMI_MAX_PPT_LIMIT=0,
        AMDSMI_MAX_UTIL=100,
        amdsmi_wrapper=wrapper,
        # Overwritten per-test.
        amdsmi_set_npm_limit=lambda _handle, _limit: None,
        amdsmi_get_npm_info=lambda _handle: {
            "max_node_power_limit": "N/A",
            "status": _NPM_STATUS_ENABLED,
        },
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


def _load_set_value_module():
    if _CLI_DIR and _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    # Some sibling test files (e.g. test_cli_cache_labels.py,
    # test_cli_vram_type_lpddr5.py) install a stub top-level
    # "amdsmi_cli_exceptions" module into sys.modules to satisfy a different
    # CLI submodule's import needs, and don't always restore the original
    # entry afterwards. set_value.py does `from amdsmi_cli_exceptions import
    # AmdSmiRequiredCommandException` at module scope; if a stale stub
    # lacking that class is already cached, the import fails even though the
    # real module (importable via _CLI_DIR on sys.path) has it. Drop any
    # cached entry so the real module is freely re-resolved here, regardless
    # of what ran earlier in the same pytest session.
    sys.modules.pop("amdsmi_cli_exceptions", None)
    spec = importlib.util.spec_from_file_location("set_value_under_test_npm", SET_VALUE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def __init__(self):
        self.output = {}

    def print_output(self, *args, **kwargs):
        pass

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False


class _RecordingErrorCollector:
    def __init__(self):
        self.codes = []
        self.library_codes = []

    def record(self, exit_code):
        self.codes.append(exit_code)

    def record_library_error(self, error_code):
        self.library_codes.append(error_code)


class _StubHelpers:
    """``self.helpers`` stub reproducing validate_and_set_node_power_limit()'s
    status mapping; the real validation is covered by TestValidateAndSetNodePowerLimit.
    """

    def __init__(self):
        self.error_collector = _RecordingErrorCollector()

    def get_output_format(self):
        return "human"

    def validate_and_set_node_power_limit(self, node_handle, requested_limit, logger):
        # Resolved lazily (not at module scope) because the fake "amdsmi"
        # package is only installed into sys.modules once a test class's
        # setUpClass() runs, which is after this class body executes.
        amdsmi_pkg = sys.modules["amdsmi"]
        interface = amdsmi_pkg.amdsmi_interface
        exception = amdsmi_pkg.amdsmi_exception
        try:
            interface.amdsmi_set_npm_limit(node_handle, requested_limit)
            return f"Successfully set node power limit to {requested_limit} W."
        except exception.AmdSmiLibraryException as e:
            if e.get_error_code() == interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                raise PermissionError("Command requires elevation") from e
            self.error_collector.record_library_error(e.get_error_code())
            return (
                f"[{e.get_error_info(detailed=False)}] "
                f"Unable to set node power limit to {requested_limit} W"
            )


class TestCliSetNodePowerLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        modules = _install_fake_amdsmi()
        modules["amdsmi_cli_exceptions"] = None
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = _load_set_value_module()

    def _make_command(self, node_handle):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.node_handle = node_handle
        cmd.helpers = _StubHelpers()
        return cmd

    def _make_args(self, node_power_limit, gpu=None):
        return types.SimpleNamespace(gpu=gpu, node_power_limit=node_power_limit)

    def test_gpu_conflict_raises_invalid_parameter(self):
        calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, limit: calls.append((h, limit))
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_power_limit=250, gpu="gpu0")

        with self.assertRaises(self.module.AmdSmiInvalidParameterException) as ctx:
            cmd.set_value(args)

        self.assertIn(
            "--node-power-limit/-n: not allowed with argument --gpu/-g", str(ctx.exception)
        )
        self.assertEqual(calls, [], "API must not be called when --gpu is also given")

    def test_no_node_handle_records_not_supported_without_calling_api(self):
        calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, limit: calls.append((h, limit))
        cmd = self._make_command(node_handle=None)
        args = self._make_args(node_power_limit=250)

        cmd.set_value(args)

        self.assertEqual(calls, [], "API must not be called when no node handle is available")
        message = cmd.logger.output["set_node_power_limit"]
        self.assertIn("NOT_SUPPORTED", message)
        self.assertIn("no NPM-capable node found", message)
        self.assertEqual(cmd.helpers.error_collector.library_codes, [_STATUS_NOT_SUPPORTED])

    def test_success_reports_value_in_watts(self):
        calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, limit: calls.append((h, limit))
        node_handle = object()
        cmd = self._make_command(node_handle=node_handle)
        args = self._make_args(node_power_limit=6000)

        cmd.set_value(args)

        self.assertEqual(calls, [(node_handle, 6000)])
        message = cmd.logger.output["set_node_power_limit"]
        self.assertIn("Successfully set node power limit to 6000 W", message)

    def test_no_perm_raises_permission_error(self):
        def _raise(_handle, _limit):
            raise _FakeLibraryException(
                _STATUS_NO_PERM, "AMDSMI_STATUS_NO_PERM - Permission Denied"
            )

        self.interface.amdsmi_set_npm_limit = _raise
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_power_limit=250)

        with self.assertRaises(PermissionError):
            cmd.set_value(args)

    def test_not_supported_from_api_reports_error_without_raising(self):
        def _raise(_handle, _limit):
            raise _FakeLibraryException(
                _STATUS_NOT_SUPPORTED, "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported"
            )

        self.interface.amdsmi_set_npm_limit = _raise
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_power_limit=9999)

        cmd.set_value(args)  # must not raise

        message = cmd.logger.output["set_node_power_limit"]
        self.assertIn("AMDSMI_STATUS_NOT_SUPPORTED", message)
        self.assertIn("Unable to set node power limit to 9999 W", message)


# ---------------------------------------------------------------------------
# Direct tests for amdsmi_helpers.AMDSMIHelpers.validate_and_set_node_power_limit()
# ---------------------------------------------------------------------------


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeParameterException(Exception):
    pass


def _install_fake_amdsmi_for_helpers():
    """Like ``_install_fake_amdsmi()``, plus the extra surface
    ``amdsmi_helpers.py``'s module-level ``from amdsmi_init import *`` needs to
    import cleanly: ``amdsmi_init.py`` calls ``amdsmi_cli_init()`` at import
    time, which needs ``AmdSmiInitFlags``, a callable ``amdsmi_init()``, and
    (for its ``isinstance()`` check on the init result) ``AmdSmiLibraryException``
    / ``AmdSmiParameterException`` directly on the ``amdsmi_interface`` module
    (as opposed to the ``amdsmi_exception`` submodule).
    """
    modules = _install_fake_amdsmi()
    interface = modules["amdsmi.amdsmi_interface"]
    interface.AmdSmiInitFlags = _FakeInitFlags
    interface.amdsmi_init = lambda _flag: None
    interface.amdsmi_shut_down = lambda: None
    interface.AmdSmiLibraryException = _FakeLibraryException
    interface.AmdSmiParameterException = _FakeParameterException
    return modules


class TestValidateAndSetNodePowerLimit(unittest.TestCase):
    _EXTRA_STUBBED_NAMES = ("amdsmi_init", "amdsmi_helpers", "amdsmi_cli_exceptions", "BDF")

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = _install_fake_amdsmi_for_helpers()
        modules.update({name: None for name in cls._EXTRA_STUBBED_NAMES})
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]

        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_helpers as amdsmi_helpers_module

        # staticmethod() wrapping prevents `self.validate` from auto-binding
        # this TestCase instance as the method's `self` argument (functions
        # assigned as class attributes behave as descriptors; plain instance
        # attribute assignment doesn't trigger this, but a classmethod-set
        # attribute inherited by instances does).
        cls.validate = staticmethod(
            amdsmi_helpers_module.AMDSMIHelpers.validate_and_set_node_power_limit
        )
        cls.exceptions_module = sys.modules["amdsmi_cli_exceptions"]

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def setUp(self):
        self.calls = []
        self.error_collector = _RecordingErrorCollector()
        self.interface.amdsmi_set_npm_limit = lambda h, limit: self.calls.append((h, limit))

    def _validate(self, requested_limit, max_node_power_limit, npm_status=_NPM_STATUS_ENABLED):
        self.interface.amdsmi_get_npm_info = lambda _h: {
            "max_node_power_limit": max_node_power_limit,
            "status": npm_status,
        }
        # Duck-typed ``self`` skips AMDSMIHelpers.__init__'s platform probing.
        fake_self = types.SimpleNamespace(
            get_output_format=lambda: "human", error_collector=self.error_collector
        )
        logger = _FakeLogger()
        return self.validate(fake_self, object(), requested_limit, logger)

    def test_over_max_rejects(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(requested_limit=6401, max_node_power_limit=6400)
        message = str(ctx.exception)
        self.assertIn("--node-power-limit must be between 1W and 6400W", message)
        self.assertIn("amd-smi set -h", message)
        self.assertEqual(self.calls, [], "API must not be called on a rejected request")

    def test_zero_rejects(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(requested_limit=0, max_node_power_limit=6400)
        self.assertIn("must be between 1W and 6400W", str(ctx.exception))
        self.assertEqual(self.calls, [])

    def test_in_range_succeeds(self):
        result = self._validate(requested_limit=6000, max_node_power_limit=6400)

        self.assertIn("Successfully set node power limit to 6000 W", result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], 6000)

    def test_npm_disabled_rejects(self):
        # NPM disabled on this node: reject before even looking at the max
        # bound (writing cur_node_power_limit while NPM is disabled has no
        # defined effect).
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(
                requested_limit=250, max_node_power_limit=6400, npm_status=_NPM_STATUS_DISABLED
            )
        self.assertIn("NPM is disabled", str(ctx.exception))
        self.assertEqual(self.calls, [], "API must not be called when NPM is disabled")

    def test_na_max_rejects(self):
        # Fail closed: an unreadable platform max must not fall back to a bare > 0 check.
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(requested_limit=250, max_node_power_limit="N/A")
        self.assertIn("platform maximum is unavailable", str(ctx.exception))
        self.assertEqual(self.calls, [], "API must not be called when the platform max is unknown")

    def test_library_inval_raises_instead_of_returning(self):
        # The library can still reject a request that passed the CLI pre-check.
        def _raise(_handle, _limit):
            raise _FakeLibraryException(_STATUS_INVAL, "AMDSMI_STATUS_INVAL - Invalid parameters")

        self.interface.amdsmi_set_npm_limit = _raise
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._validate(requested_limit=250, max_node_power_limit=6400)

    def test_library_error_records_status(self):
        def _raise(_handle, _limit):
            raise _FakeLibraryException(
                _STATUS_NOT_SUPPORTED, "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported"
            )

        self.interface.amdsmi_set_npm_limit = _raise
        result = self._validate(requested_limit=250, max_node_power_limit=6400)

        self.assertIn("Unable to set node power limit to 250 W", result)
        self.assertEqual(self.error_collector.library_codes, [_STATUS_NOT_SUPPORTED])


# ---------------------------------------------------------------------------
# set parser: --node-power-limit on a 1VF guest, and the --gtt/--gpu guard
# ---------------------------------------------------------------------------

PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py") if _CLI_DIR else ""
_GTT_GPU_CONFLICT = "argument --gtt/-G: not allowed with argument --gpu/-g"


def _build_fake_modules_for_parser():
    """Minimal stub surface for importing amdsmi_parser.py.

    Mirrors test_output_file_stdin.py's _install_stubs(): the parser module
    only needs these names to bind at import time, not to be functional.
    """
    interface = fake_module("amdsmi.amdsmi_interface")
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface)
    version_mod = fake_module("_version", __version__="0.0.0-test")
    helpers_mod = fake_module("amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {}))

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "_version": version_mod,
        "amdsmi_helpers": helpers_mod,
    }


def _load_parser_module():
    if _CLI_DIR and _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    # See _load_set_value_module()'s comment: sibling test files can leave a
    # stale/incomplete "amdsmi_cli_exceptions" stub cached in sys.modules.
    # amdsmi_parser.py imports it at module scope, so drop any cached entry
    # first to force re-resolution of the real module.
    sys.modules.pop("amdsmi_cli_exceptions", None)
    spec = importlib.util.spec_from_file_location("amdsmi_parser_under_test_npm", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHelpersGuest:
    """Non-baremetal (1VF guest) helpers stub for ``_add_set_value_parser``.

    ``is_baremetal()`` is False, so the baremetal-only set arguments (fan,
    perf-level, ...) are skipped, but ``-o``/``--power-cap`` and
    ``-n``/``--node-power-limit`` are gated only by
    ``is_amdgpu_initialized()`` and must still register.
    """

    def is_linux(self):
        return True

    def is_amdgpu_initialized(self):
        return True

    def is_baremetal(self):
        return False

    def is_hypervisor(self):
        return False

    def is_amd_hsmp_initialized(self):
        return False

    def is_ainic_initialized(self):
        return False

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def get_power_caps(self):
        return ("0 W", "550 W", "0 W", "550 W")

    def get_output_format(self):
        return "human"

    def get_device_handles_from_gpu_selections(self, **_kwargs):
        return 1, True, [object()]


class TestSetValueParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARSER_PATH or not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest("amd-smi CLI amdsmi_parser.py not found (source or installed)")
        modules = _build_fake_modules_for_parser()
        modules["amdsmi_cli_exceptions"] = None
        stub_modules(cls, modules)
        cls.parser_mod = _load_parser_module()

    def _build_set_parser(self):
        # A minimal fake ``self`` -- exercises the real, unbound
        # _add_set_value_parser method body without paying for
        # AMDSMIParser.__init__'s full platform probing.
        fake_self = object.__new__(self.parser_mod.AMDSMIParser)
        fake_self.helpers = _FakeHelpersGuest()
        fake_self.description = "test"
        fake_self.gpu_choices = {}
        fake_self.gpu_choices_str = ""
        fake_self.cpu_choices_str = ""
        fake_self.nic_choices_str = ""
        fake_self.core_choices_str = ""
        fake_self.switch_choices_str = ""
        fake_self.vf_choices = []

        top_parser = argparse.ArgumentParser()
        subparsers = top_parser.add_subparsers()
        fake_self._add_set_value_parser(subparsers, func=lambda args: None)
        return subparsers.choices["set"]

    def test_node_power_limit_registered_on_guest(self):
        set_parser = self._build_set_parser()

        args = set_parser.parse_args(["--node-power-limit", "100"])

        self.assertEqual(args.node_power_limit, 100)

    def test_unrelated_gpu_error_keeps_argparse_error(self):
        parser = argparse.ArgumentParser()
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(parser, gtt_flags=("--gtt", "-G"))
        argv = ["amd-smi", "set", "--gtt", "8", "--gpu", "0"]

        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr, self.assertRaises(SystemExit) as raised:
            parser.error("argument --cpu: not allowed with argument --gpu/-g")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("argument --cpu", stderr.getvalue())
        self.assertNotIn(_GTT_GPU_CONFLICT, stderr.getvalue())

    def test_missing_gpu_value_gets_gtt_error(self):
        parser = argparse.ArgumentParser()
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(parser, gtt_flags=("--gtt", "-G"))
        argv = ["amd-smi", "set", "--gtt", "8", "--gpu"]

        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr, self.assertRaises(SystemExit) as raised:
            parser.error("argument --gpu/-g: expected at least one argument")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            _GTT_GPU_CONFLICT + " (--gtt is a system-wide setting, not per-GPU)", stderr.getvalue()
        )

    def test_missing_unrelated_value_keeps_argparse_error(self):
        parser = argparse.ArgumentParser()
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(parser, gtt_flags=("--gtt", "-G"))
        argv = ["amd-smi", "set", "--gtt", "8", "--gpu", "0", "--power-cap"]

        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr, self.assertRaises(SystemExit) as raised:
            parser.error("argument -o/--power-cap: expected at least one argument")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("argument -o/--power-cap", stderr.getvalue())
        self.assertNotIn(_GTT_GPU_CONFLICT, stderr.getvalue())
