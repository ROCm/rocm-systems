#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi set --node-power-limit``.

Two layers are covered here:

* ``TestCliSetNodePowerLimit`` drives ``SetValueCommands.set_value()``'s
  node-wide (not per-GPU) early-dispatch block, modeled on the existing
  ``--gtt`` block:

  * ``--node-power-limit`` combined with ``--gpu`` is rejected before any
    library call, exiting with status 2 and a stderr message (mirrors the
    ``--gtt``/``--gpu`` argparse-level guard's runtime-level counterpart).
  * ``self.node_handle is None`` (no NPM-capable node resolved at startup,
    e.g. the pre-existing ``oam_id != 0`` gate on current mock hardware)
    short-circuits to a NOT_SUPPORTED-style message without calling the API.
  * A successful set reports "Successfully set node power limit to <value> W."
  * ``AMDSMI_STATUS_NO_PERM`` from the library is translated into a raised
    ``PermissionError`` (so the CLI's elevation-prompt wrapper can catch it),
    not swallowed into an output message.
  * Any other library exception (e.g. NOT_SUPPORTED bubbling up from
    ``rsmi_dev_npm_limit_set()`` for a missing sysfs file) is caught and
    reported as an output message referencing the requested value, without
    raising.

  ``self.helpers`` is a minimal stub (mirroring ``test_cli_set_clk_limit.py``'s
  ``_StubHelpers`` pattern) that reproduces
  ``validate_and_set_node_power_limit()``'s call/status-mapping contract, so
  these tests can focus on ``set_value.py``'s own dispatch behavior without
  re-deriving the validation logic tested in depth below.

* ``TestValidateAndSetNodePowerLimit`` drives the real
  ``amdsmi_helpers.AMDSMIHelpers.validate_and_set_node_power_limit()``
  directly (added for the original F-1 fix, and updated for the F-2 fix
  below), against a real import of ``amdsmi_helpers`` with only the C library
  faked -- this is the validation logic itself, previously exercised nowhere
  in the tree:

  * a request above the platform max (``amdsmi_get_npm_info()``'s
    ``max_node_power_limit``) is rejected before any write.
  * a request of ``0`` is rejected.
  * an in-range request succeeds and calls ``amdsmi_set_npm_limit()`` with
    the requested value verbatim.
  * an unreadable platform max (``max_node_power_limit == "N/A"``) now fails
    *closed*: the request is rejected, not allowed through on a bare
    positivity check. (Previously this class of degraded-driver scenario
    was fail-open; see the F-2 handoff finding this test locks in.)

* ``TestNodePowerLimitGuestRegistration`` drives ``AMDSMIParser`` directly
  (loaded the same way as ``test_output_file_stdin.py``): commit 567bed2f89d
  moved ``-n``/``--node-power-limit`` argparse registration out of the
  ``is_baremetal()``-only block so it is gated only by
  ``is_amdgpu_initialized()`` (same pattern as ``-o``/``--power-cap``),
  since NPM is exposed to 1VF guests. Asserts the argument is still
  registered and parseable under a non-baremetal (guest) helpers stub.

* ``TestGetMaxNodePowerLimit`` drives the real
  ``amdsmi_helpers.AMDSMIHelpers.get_max_node_power_limit()`` (added by
  ece891b6787 to size --node-power-limit's help-text bound), against a real
  import of ``amdsmi_helpers`` with only the C library faked -- mirroring
  ``TestValidateAndSetNodePowerLimit``'s pattern:

  * a first-device success returns immediately without probing further
    devices.
  * an exception on every device falls back to "N/A".
  * a device that succeeds but reports "N/A" as the value itself (not an
    exception) is a distinct code path from the exception branch; the loop
    must still continue to the next device.
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


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "set_value.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""

_STATUS_NOT_SUPPORTED = 2
_STATUS_NO_PERM = 10
_STATUS_INVAL = 5


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
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    wrapper.AMDSMI_STATUS_NO_PERM = _STATUS_NO_PERM
    wrapper.AMDSMI_STATUS_INVAL = _STATUS_INVAL
    interface.amdsmi_wrapper = wrapper
    # Overwritten per-test.
    interface.amdsmi_set_npm_limit = lambda _handle, _limit: None
    interface.amdsmi_get_npm_info = lambda _handle: {"max_node_power_limit": "N/A"}

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


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


class _StubHelpers:
    """Minimal ``self.helpers`` stub for the node-power-limit dispatch block.

    Mirrors ``test_cli_set_clk_limit.py``'s ``_StubHelpers`` pattern: just
    enough of ``validate_and_set_node_power_limit()``'s call/status-mapping
    contract (call the C entry point, map ``AMDSMI_STATUS_NO_PERM`` to a
    raised ``PermissionError``, otherwise format success/error strings) for
    ``set_value.py``'s dispatch block to exercise, without pulling in the real
    max-bound validation -- that logic is exercised directly and in depth by
    ``TestValidateAndSetNodePowerLimit`` below, against the real
    ``amdsmi_helpers.AMDSMIHelpers`` implementation.
    """

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
            return (
                f"[{e.get_error_info(detailed=False)}] "
                f"Unable to set node power limit to {requested_limit} W"
            )


class TestCliSetNodePowerLimit(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
        "amdsmi_cli_exceptions",
    )

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()
        cls.module = _load_set_value_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _make_command(self, node_handle):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.node_handle = node_handle
        cmd.helpers = _StubHelpers()
        return cmd

    def _make_args(self, node_power_limit, gpu=None):
        return types.SimpleNamespace(gpu=gpu, node_power_limit=node_power_limit)

    def test_gpu_conflict_exits_with_status_2(self):
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_power_limit=250, gpu="gpu0")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cmd.set_value(args)

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--node-power-limit", stderr.getvalue())
        self.assertIn("--gpu", stderr.getvalue())

    def test_no_node_handle_reports_not_supported_without_calling_api(self):
        calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, l: calls.append((h, l))
        cmd = self._make_command(node_handle=None)
        args = self._make_args(node_power_limit=250)

        cmd.set_value(args)

        self.assertEqual(calls, [], "API must not be called when no node handle is available")
        message = cmd.logger.output["set_node_power_limit"]
        self.assertIn("NOT_SUPPORTED", message)
        self.assertIn("no NPM-capable node found", message)

    def test_success_reports_value_in_watts(self):
        calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, l: calls.append((h, l))
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
    interface = _install_fake_amdsmi()
    interface.AmdSmiInitFlags = _FakeInitFlags
    interface.amdsmi_init = lambda _flag: None
    interface.amdsmi_shut_down = lambda: None
    interface.AmdSmiLibraryException = _FakeLibraryException
    interface.AmdSmiParameterException = _FakeParameterException
    return interface


class TestValidateAndSetNodePowerLimit(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
        "amdsmi_init",
        "amdsmi_helpers",
        "amdsmi_cli_exceptions",
        "BDF",
    )

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        for name in cls._SAVED_MODULE_NAMES:
            sys.modules.pop(name, None)
        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        cls.interface = _install_fake_amdsmi_for_helpers()
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
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def setUp(self):
        self.calls = []
        self.interface.amdsmi_set_npm_limit = lambda h, l: self.calls.append((h, l))

    def _validate(self, requested_limit, max_node_power_limit):
        self.interface.amdsmi_get_npm_info = lambda _h: {
            "max_node_power_limit": max_node_power_limit
        }
        # A lightweight duck-typed ``self`` -- exercises the real, unbound
        # ``validate_and_set_node_power_limit`` method body without paying for
        # ``AMDSMIHelpers.__init__``'s platform/hypervisor probing, which is
        # irrelevant to this validation logic.
        fake_self = types.SimpleNamespace(get_output_format=lambda: "human")
        logger = _FakeLogger()
        return self.validate(fake_self, object(), requested_limit, logger)

    def test_over_max_rejects(self):
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(requested_limit=6401, max_node_power_limit=6400)
        self.assertIn("must be between 1W and 6400W", str(ctx.exception))
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

    def test_na_max_now_rejects(self):
        # F-2 fail-closed fix: an unreadable platform max ("N/A") must reject
        # the request outright, not fall back to a bare "> 0" check that would
        # let an unbounded value through in a degraded-driver scenario.
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException) as ctx:
            self._validate(requested_limit=250, max_node_power_limit="N/A")
        self.assertIn("platform maximum is unavailable", str(ctx.exception))
        self.assertEqual(self.calls, [], "API must not be called when the platform max is unknown")

    def test_library_inval_raises_instead_of_returning(self):
        # amdsmi_set_npm_limit() itself can still reject a request that passed
        # the CLI-side pre-check (e.g. a race against a shrinking platform
        # max). AMDSMI_STATUS_INVAL from the library must raise, exiting
        # non-zero, not be swallowed into a returned error string/dict.
        self.interface.amdsmi_set_npm_limit = lambda h, l: (_ for _ in ()).throw(
            _FakeLibraryException(_STATUS_INVAL, "AMDSMI_STATUS_INVAL - Invalid parameters")
        )
        with self.assertRaises(self.exceptions_module.AmdSmiInvalidParameterValueException):
            self._validate(requested_limit=250, max_node_power_limit=6400)


# ---------------------------------------------------------------------------
# Direct tests for amdsmi_helpers.AMDSMIHelpers.get_max_node_power_limit()
# ---------------------------------------------------------------------------


class TestGetMaxNodePowerLimit(unittest.TestCase):
    """Covers get_max_node_power_limit() (added by ece891b6787), which mirrors
    get_power_caps()'s per-device probe loop: it queries devices in order and
    returns the first successful reading, falling back to "N/A" only if every
    device fails or reports "N/A" itself.
    """

    _SAVED_MODULE_NAMES = TestValidateAndSetNodePowerLimit._SAVED_MODULE_NAMES

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        for name in cls._SAVED_MODULE_NAMES:
            sys.modules.pop(name, None)
        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        cls.interface = _install_fake_amdsmi_for_helpers()
        import amdsmi_helpers as amdsmi_helpers_module

        # staticmethod() wrapping avoids auto-binding this TestCase instance
        # as `self`; see TestValidateAndSetNodePowerLimit's setUpClass.
        cls.get_max = staticmethod(amdsmi_helpers_module.AMDSMIHelpers.get_max_node_power_limit)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _fake_self(self, gpu_handles):
        # get_max_node_power_limit() only calls self.get_gpu_handles(); a
        # duck-typed stub avoids paying for real device enumeration.
        return types.SimpleNamespace(get_gpu_handles=lambda: gpu_handles)

    def test_first_device_success_short_circuits(self):
        calls = []

        def node_handle(dev):
            calls.append(dev)
            return f"node_{dev}"

        self.interface.amdsmi_get_node_handle = node_handle
        self.interface.amdsmi_get_npm_info = lambda _h: {"max_node_power_limit": 6000}

        result = self.get_max(self._fake_self(["dev0", "dev1"]))

        self.assertEqual(result, "6000 W")
        self.assertEqual(calls, ["dev0"], "second device must not be probed after a success")

    def test_all_devices_exception_falls_back_to_na(self):
        def _raise(_dev):
            raise _FakeLibraryException("npm info unavailable")

        self.interface.amdsmi_get_node_handle = _raise

        result = self.get_max(self._fake_self(["dev0", "dev1"]))

        self.assertEqual(result, "N/A")

    def test_na_value_on_first_device_continues_to_next(self):
        # Distinct from the exception path: the device call succeeds but
        # reports "N/A" as the value itself, so the loop must fall through to
        # the next device rather than returning "N/A" early.
        def node_handle(dev):
            return f"node_{dev}"

        def npm_info(node_handle_):
            if node_handle_ == "node_dev0":
                return {"max_node_power_limit": "N/A"}
            return {"max_node_power_limit": 6400}

        self.interface.amdsmi_get_node_handle = node_handle
        self.interface.amdsmi_get_npm_info = npm_info

        result = self.get_max(self._fake_self(["dev0", "dev1"]))

        self.assertEqual(result, "6400 W")


# ---------------------------------------------------------------------------
# --node-power-limit registration on a non-baremetal (1VF guest) platform
# ---------------------------------------------------------------------------

PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py") if _CLI_DIR else ""


def _install_fake_amdsmi_for_parser():
    """Minimal stub surface for importing amdsmi_parser.py.

    Mirrors test_output_file_stdin.py's _install_stubs(): the parser module
    only needs these names to bind at import time, not to be functional.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    amdsmi_pkg.amdsmi_interface = interface
    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface

    version_mod = types.ModuleType("_version")
    version_mod.__version__ = "0.0.0-test"
    sys.modules["_version"] = version_mod

    helpers_mod = types.ModuleType("amdsmi_helpers")
    helpers_mod.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    sys.modules["amdsmi_helpers"] = helpers_mod


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

    def get_max_node_power_limit(self):
        return "6000 W"

    def get_output_format(self):
        return "human"

    def get_device_handles_from_gpu_selections(self, **_kwargs):
        return 1, True, [object()]


class TestNodePowerLimitGuestRegistration(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "_version",
        "amdsmi_helpers",
        "amdsmi_cli_exceptions",
    )

    @classmethod
    def setUpClass(cls):
        if not PARSER_PATH or not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest("amd-smi CLI amdsmi_parser.py not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        _install_fake_amdsmi_for_parser()
        cls.parser_mod = _load_parser_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

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
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(
            parser,
            gtt_flags=("--node-power-limit", "-n"),
            reason="--node-power-limit is a node-wide setting, not per-GPU",
        )
        argv = ["amd-smi", "set", "--node-power-limit", "100", "--gpu", "0"]

        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parser.error("argument --cpu: not allowed with argument --gpu/-g")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("argument --cpu", stderr.getvalue())
        self.assertNotIn(
            "--node-power-limit/-n: not allowed with argument --gpu/-g", stderr.getvalue()
        )

    def test_missing_gpu_value_gets_node_power_limit_error(self):
        parser = argparse.ArgumentParser()
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(
            parser,
            gtt_flags=("--node-power-limit", "-n"),
            reason="--node-power-limit is a node-wide setting, not per-GPU",
        )
        argv = ["amd-smi", "set", "--node-power-limit", "100", "--gpu"]

        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parser.error("argument --gpu/-g: expected at least one argument")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "--node-power-limit/-n: not allowed with argument --gpu/-g", stderr.getvalue()
        )

    def test_missing_unrelated_value_keeps_argparse_error(self):
        parser = argparse.ArgumentParser()
        self.parser_mod.AMDSMIParser._guard_gtt_gpu_conflict(
            parser,
            gtt_flags=("--node-power-limit", "-n"),
            reason="--node-power-limit is a node-wide setting, not per-GPU",
        )
        argv = ["amd-smi", "set", "--node-power-limit", "100", "--gpu", "0", "--power-cap"]

        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parser.error("argument -o/--power-cap: expected at least one argument")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("argument -o/--power-cap", stderr.getvalue())
        self.assertNotIn(
            "--node-power-limit/-n: not allowed with argument --gpu/-g", stderr.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
