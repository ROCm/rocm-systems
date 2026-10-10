#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for ``amd-smi set --node-balancing-mode``.

Two layers are covered here, modeled on ``test_cli_set_clk_limit.py`` /
``test_cli_ras_cper_json.py``:

* ``TestCliSetNodeBalancingMode`` drives ``SetValueCommands.set_value()``'s
  node-wide (not per-GPU) early-dispatch block for ``--node-balancing-mode``:

  * ``--node-balancing-mode`` combined with ``--gpu`` is rejected before any
    library call by raising ``AmdSmiInvalidParameterException``. ``amdsmi_parser.py``
    also registers ``--node-balancing-mode`` with ``_guard_gtt_gpu_conflict()``
    (alongside ``--gtt``/``--node-power-limit``), but that guard only improves
    the message for argparse's own "expected at least one argument" diagnostic
    (e.g. a bare trailing ``--gpu``) -- it does not fire for this
    fully-formed ``--gpu VALUE --node-balancing-mode VALUE`` combination, so
    the runtime check exercised here is still what rejects it, raising a plain
    exception, not a ``SystemExit``; only ``amdsmi_cli.py``'s top-level handler
    (bypassed by calling ``set_value()`` directly here) turns it into
    ``sys.exit()``.
  * A successful set reports "Successfully set NPM balancing mode to
    <POWER_BALANCING|FREQUENCY_BALANCING>".
  * ``AMDSMI_STATUS_NO_PERM`` from the library is translated into a raised
    ``PermissionError`` (so the CLI's elevation-prompt wrapper can catch it),
    not swallowed into an output message.
  * ``AMDSMI_STATUS_NOT_SUPPORTED`` (which covers both NPM being disabled and
    the board/mode sysfs file being missing/unreadable while NPM is enabled --
    indistinguishable from this status code alone) is caught and reported as
    "NPM balancing mode is not supported on this node; cannot set balancing
    mode", without raising.

  ``self.helpers`` is a minimal stub (mirroring ``test_cli_set_clk_limit.py``'s
  ``_StubHelpers`` pattern) that reproduces
  ``validate_and_set_node_balancing_mode()``'s call/status-mapping contract,
  so these tests can focus on ``set_value.py``'s own dispatch behavior
  without re-deriving the validation logic tested in depth below.

* ``TestValidateAndSetNodeBalancingMode`` drives the real
  ``amdsmi_helpers.AMDSMIHelpers.validate_and_set_node_balancing_mode()``
  directly, against a real import of ``amdsmi_helpers`` with only the C
  library faked:

  * ``node_handle is None`` is rejected with a "No node handle available"
    message, without calling the API.
  * A successful set translates the CLI's full-name mode
    (``POWER_BALANCING``/``FREQUENCY_BALANCING``) to the library's internal
    ``"PB"``/``"FB"`` strings via ``NPM_BALANCING_MODE_FROM_CLI`` before
    calling ``amdsmi_set_npm_balancing_mode()``, but echoes the full name back
    in its own message/return value -- ``amdsmi_interface.py``'s own
    ``"PB"``/``"FB"`` string contract is unchanged, translation happens only
    at this CLI-helper boundary.
  * ``AMDSMI_STATUS_NO_PERM`` raises ``PermissionError``.
  * ``AMDSMI_STATUS_NOT_SUPPORTED`` (the balancing-mode-specific divergence
    from ``amdsmi_set_npm_limit()``, which uses ``AMDSMI_STATUS_INVAL`` for
    its own disabled-check) returns a "not supported on this node" message
    without raising. This is reported either by a pre-check
    (``amdsmi_get_npm_info()`` status) when NPM itself is disabled, or by the
    set call itself when the board/mode sysfs file is missing/unreadable
    while NPM is enabled.
  * ``AMDSMI_STATUS_SETTING_UNAVAILABLE`` -- a distinct status from
    ``AMDSMI_STATUS_NOT_SUPPORTED`` -- means the requested mode is absent
    from this platform's ``supported_mode`` bitmask, and is reported as
    ``"BALANCING_MODE: [AMDSMI_STATUS_SETTING_UNAVAILABLE] <mode> is not
    supported on this platform"``.
  * Any other library exception (e.g. ``AMDSMI_STATUS_INVAL``) is reported as
    an error message referencing the requested mode, without raising.

Unlike ``--node-power-limit`` (moved out of the baremetal-only argparse gate
so it reaches 1VF guests), ``--node-balancing-mode`` stays inside
``amdsmi_parser.py``'s ``if self.helpers.is_baremetal():`` block (added right
alongside ``--gtt``), per this feature's "Lin BM only" scope -- so there is no
guest-registration test here.
"""

import os
import sys
import types
import unittest
from unittest import mock

from common.common import amdsmi_path, cli_search_order, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else None

_STATUS_NOT_SUPPORTED = 2
_STATUS_NO_PERM = 10
_STATUS_INVAL = 5
_STATUS_SETTING_UNAVAILABLE = 55

# Modules imported (directly or transitively) when the source CLI loads
# against the faked ``amdsmi`` package; cleared/restored around each suite so
# a real/installed copy loaded by a sibling test is never shadowed.
_CLI_MODULES = (
    "amdsmi",
    "amdsmi.amdsmi_interface",
    "amdsmi.amdsmi_exception",
    "amdsmi.amdsmi_wrapper",
    "amdsmi_init",
    "amdsmi_helpers",
    "amdsmi_cli_exceptions",
    "BDF",
)


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self, detailed=True):
        return self._message if detailed else self._message.split(" - ")[0]


def _build_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    wrapper.AMDSMI_STATUS_NO_PERM = _STATUS_NO_PERM
    wrapper.AMDSMI_STATUS_INVAL = _STATUS_INVAL
    wrapper.AMDSMI_STATUS_SETTING_UNAVAILABLE = _STATUS_SETTING_UNAVAILABLE
    # amdsmi_helpers.py's CPER_DECODE_MESSAGES class-body dict (unrelated to
    # this feature) is evaluated at import time and needs these to resolve.
    wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE = 100
    wrapper.AMDSMI_STATUS_UNEXPECTED_DATA = 101
    wrapper.AMDSMI_NPM_STATUS_DISABLED = 0
    wrapper.AMDSMI_NPM_STATUS_ENABLED = 1
    interface.amdsmi_wrapper = wrapper
    # set_value.py imports these two constants by name at module scope.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    # Overwritten per-test.
    interface.amdsmi_set_npm_balancing_mode = lambda _handle, _mode: None
    # Defaults to "NPM enabled" so existing tests reach the set call below the
    # enablement pre-check; overwritten per-test to exercise the disabled path.
    interface.amdsmi_get_npm_info = lambda _handle: {"status": wrapper.AMDSMI_NPM_STATUS_ENABLED}

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


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
    """Minimal ``self.helpers`` stub for the node-balancing-mode dispatch
    block. Mirrors ``validate_and_set_node_balancing_mode()``'s
    call/status-mapping contract for the non-None-node-handle path -- the
    None-node-handle path and the full validation logic are exercised
    directly and in depth by ``TestValidateAndSetNodeBalancingMode`` below,
    against the real ``amdsmi_helpers.AMDSMIHelpers`` implementation.
    """

    def __init__(self, interface):
        self._interface = interface

    def get_output_format(self):
        return "human"

    def validate_and_set_node_balancing_mode(self, node_handle, requested_mode, logger):
        interface = self._interface
        exception = sys.modules["amdsmi.amdsmi_exception"]
        try:
            interface.amdsmi_set_npm_balancing_mode(node_handle, requested_mode)
            return f"Successfully set NPM balancing mode to {requested_mode}"
        except exception.AmdSmiLibraryException as e:
            if e.get_error_code() == interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM:
                raise PermissionError("Command requires elevation") from e
            if e.get_error_code() == interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                return "NPM balancing mode is not supported on this node; cannot set balancing mode"
            return (
                f"[{e.get_error_info(detailed=False)}] "
                f"Unable to set NPM balancing mode to {requested_mode}"
            )


class TestCliSetNodeBalancingMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH or not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI set_value.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = dict.fromkeys(_CLI_MODULES)
        modules.update(_build_fake_amdsmi())
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = load_cli_module(
            "set_value_under_test_npm_balancing_mode", SET_VALUE_PATH, sys_path_dir=_CLI_DIR
        )

    def _make_command(self, node_handle):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.node_handle = node_handle
        cmd.helpers = _StubHelpers(self.interface)
        return cmd

    def _make_args(self, node_balancing_mode, gpu=None):
        return types.SimpleNamespace(gpu=gpu, node_balancing_mode=node_balancing_mode)

    def test_gpu_conflict_raises_invalid_parameter_exception(self):
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_balancing_mode="POWER_BALANCING", gpu="gpu0")

        with self.assertRaises(self.module.AmdSmiInvalidParameterException) as ctx:
            cmd.set_value(args)

        self.assertIn("--node-balancing-mode", str(ctx.exception))
        self.assertIn("--gpu", str(ctx.exception))

    def test_success_reports_requested_mode(self):
        calls = []
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: calls.append((h, m))
        node_handle = object()
        cmd = self._make_command(node_handle=node_handle)
        args = self._make_args(node_balancing_mode="FREQUENCY_BALANCING")

        cmd.set_value(args)

        self.assertEqual(calls, [(node_handle, "FREQUENCY_BALANCING")])
        message = cmd.logger.output["set_node_balancing_mode"]
        self.assertIn("Successfully set NPM balancing mode to FREQUENCY_BALANCING", message)

    def test_no_perm_raises_permission_error(self):
        def _raise(_handle, _mode):
            raise _FakeLibraryException(
                _STATUS_NO_PERM, "AMDSMI_STATUS_NO_PERM - Permission Denied"
            )

        self.interface.amdsmi_set_npm_balancing_mode = _raise
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_balancing_mode="POWER_BALANCING")

        with self.assertRaises(PermissionError):
            cmd.set_value(args)

    def test_not_supported_from_api_reports_error_without_raising(self):
        def _raise(_handle, _mode):
            raise _FakeLibraryException(
                _STATUS_NOT_SUPPORTED, "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported"
            )

        self.interface.amdsmi_set_npm_balancing_mode = _raise
        cmd = self._make_command(node_handle=object())
        args = self._make_args(node_balancing_mode="POWER_BALANCING")

        cmd.set_value(args)  # must not raise

        message = cmd.logger.output["set_node_balancing_mode"]
        self.assertIn(
            "NPM balancing mode is not supported on this node; cannot set balancing mode", message
        )


# ---------------------------------------------------------------------------
# Direct tests for amdsmi_helpers.AMDSMIHelpers.validate_and_set_node_balancing_mode()
# ---------------------------------------------------------------------------


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeParameterException(Exception):
    pass


def _build_fake_amdsmi_for_helpers():
    """Like ``_build_fake_amdsmi()``, plus the extra surface
    ``amdsmi_helpers.py``'s module-level ``from amdsmi_init import *`` needs
    to import cleanly.
    """
    modules = _build_fake_amdsmi()
    interface = modules["amdsmi.amdsmi_interface"]
    interface.AmdSmiInitFlags = _FakeInitFlags
    interface.amdsmi_init = lambda _flag: None
    interface.amdsmi_shut_down = lambda: None
    interface.AmdSmiLibraryException = _FakeLibraryException
    interface.AmdSmiParameterException = _FakeParameterException
    return modules


class TestValidateAndSetNodeBalancingMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = dict.fromkeys(_CLI_MODULES)
        modules.update(_build_fake_amdsmi_for_helpers())
        stub_modules(cls, modules)
        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)

        cls.interface = modules["amdsmi.amdsmi_interface"]
        import amdsmi_helpers as amdsmi_helpers_module

        cls.helpers_cls = amdsmi_helpers_module.AMDSMIHelpers
        # staticmethod() wrapping prevents `self.validate` from auto-binding
        # this TestCase instance as the method's `self` argument.
        cls.validate = staticmethod(
            amdsmi_helpers_module.AMDSMIHelpers.validate_and_set_node_balancing_mode
        )

    def setUp(self):
        self.calls = []
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: self.calls.append((h, m))
        # Reset to "NPM enabled" each test -- individual tests override this
        # to exercise the NPM-disabled pre-check branch.
        self.interface.amdsmi_get_npm_info = lambda h: {
            "status": self.interface.amdsmi_wrapper.AMDSMI_NPM_STATUS_ENABLED
        }

    def _validate(self, node_handle, requested_mode):
        # A lightweight duck-typed ``self`` -- exercises the real, unbound
        # ``validate_and_set_node_balancing_mode`` method body without paying
        # for ``AMDSMIHelpers.__init__``'s platform/hypervisor probing. Reuses
        # the real class's NPM_BALANCING_MODE_FROM_CLI translation table.
        fake_self = types.SimpleNamespace(
            error_collector=mock.Mock(),
            get_output_format=lambda: "human",
            NPM_BALANCING_MODE_FROM_CLI=self.helpers_cls.NPM_BALANCING_MODE_FROM_CLI,
            NPM_BALANCING_MODE_TO_CLI=self.helpers_cls.NPM_BALANCING_MODE_TO_CLI,
        )
        logger = _FakeLogger()
        return fake_self, self.validate(fake_self, node_handle, requested_mode, logger)

    def test_none_node_handle_rejects_without_calling_api(self):
        fake_self, result = self._validate(node_handle=None, requested_mode="POWER_BALANCING")

        self.assertIn("No node handle available", result)
        self.assertEqual(self.calls, [], "API must not be called when no node handle is available")
        fake_self.error_collector.record.assert_called_once()

    def test_success_forwards_requested_mode_verbatim(self):
        node_handle = object()
        _, result = self._validate(node_handle=node_handle, requested_mode="FREQUENCY_BALANCING")

        self.assertIn("Successfully set NPM balancing mode to FREQUENCY_BALANCING", result)
        self.assertEqual(self.calls, [(node_handle, "FB")])

    def test_no_perm_raises_permission_error(self):
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: (_ for _ in ()).throw(
            _FakeLibraryException(_STATUS_NO_PERM, "AMDSMI_STATUS_NO_PERM - Permission Denied")
        )

        with self.assertRaises(PermissionError):
            self._validate(node_handle=object(), requested_mode="POWER_BALANCING")

    def test_npm_disabled_reports_distinct_message_without_raising(self):
        # The NPM-disabled case is caught by the pre-check (amdsmi_get_npm_info()
        # status), before amdsmi_set_npm_balancing_mode() is ever called -- this
        # is what fixes the bug where a platform supporting both PB and FB
        # (supported_mode is NOT gated on NPM enablement) fell through to the
        # generic "not supported on this node" message instead of reporting
        # that NPM itself is disabled.
        self.interface.amdsmi_get_npm_info = lambda h: {
            "status": self.interface.amdsmi_wrapper.AMDSMI_NPM_STATUS_DISABLED
        }

        fake_self, result = self._validate(node_handle=object(), requested_mode="POWER_BALANCING")

        self.assertIn(
            "[AMDSMI_STATUS_NOT_SUPPORTED] NPM is disabled on this node; cannot set balancing mode",
            result,
        )
        self.assertEqual(self.calls, [], "API must not be called when NPM is disabled")
        fake_self.error_collector.record_library_error.assert_called_once_with(
            _STATUS_NOT_SUPPORTED
        )

    def test_not_supported_returns_message_without_raising(self):
        # With NPM enabled (the pre-check passes), the balancing-mode-specific
        # divergence from validate_and_set_node_power_limit() still applies:
        # AMDSMI_STATUS_NOT_SUPPORTED from the set call itself is reported as a
        # message, not raised. This covers "NPM enabled but board/mode
        # unreadable", indistinguishable from this status code alone, hence
        # the non-overclaiming message text.
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: (_ for _ in ()).throw(
            _FakeLibraryException(
                _STATUS_NOT_SUPPORTED, "AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported"
            )
        )

        fake_self, result = self._validate(node_handle=object(), requested_mode="POWER_BALANCING")

        self.assertIn(
            "NPM balancing mode is not supported on this node; cannot set balancing mode", result
        )
        fake_self.error_collector.record_library_error.assert_called_once_with(
            _STATUS_NOT_SUPPORTED
        )

    def test_setting_unavailable_mode_absent_from_platform_reports_distinct_message(self):
        # Disambiguates from the generic "NPM disabled"/"not supported" messages
        # above: when the requested mode is absent from this platform's
        # supported_mode bitmask, the library raises the distinct
        # AMDSMI_STATUS_SETTING_UNAVAILABLE status instead of NOT_SUPPORTED.
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: (_ for _ in ()).throw(
            _FakeLibraryException(
                _STATUS_SETTING_UNAVAILABLE,
                "AMDSMI_STATUS_SETTING_UNAVAILABLE - Setting is not available",
            )
        )

        fake_self, result = self._validate(node_handle=object(), requested_mode="POWER_BALANCING")

        self.assertIn(
            "BALANCING_MODE: [AMDSMI_STATUS_SETTING_UNAVAILABLE] "
            "POWER_BALANCING is not supported on this platform",
            result,
        )
        fake_self.error_collector.record_library_error.assert_called_once_with(
            _STATUS_SETTING_UNAVAILABLE
        )

    def test_other_library_error_reports_message_without_raising(self):
        self.interface.amdsmi_set_npm_balancing_mode = lambda h, m: (_ for _ in ()).throw(
            _FakeLibraryException(_STATUS_INVAL, "AMDSMI_STATUS_INVAL - Invalid parameters")
        )

        _, result = self._validate(node_handle=object(), requested_mode="POWER_BALANCING")

        self.assertIn("AMDSMI_STATUS_INVAL", result)
        self.assertIn("Unable to set NPM balancing mode to POWER_BALANCING", result)
