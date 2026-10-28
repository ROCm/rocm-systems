#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for Power Stabilization's NPM note/warning behavior.

When NPM (Node-level Power Management) coordination is active, every Power
Stabilization / Ramp Management parameter becomes node-wide. ``--gpu N``
still applies the write regardless of which GPU is targeted -- there is no
rejection and no separate node-wide flag. The CLI instead prepends one note
to the normal per-field results: ``[INFO]`` when ``--gpu`` already targets
the node's primary GPU (OAM ID 0), or ``[WARNING]`` naming the equivalent
primary-GPU command otherwise.

* ``TestPowerStabilizationNpmNote`` drives ``_power_stabilization_npm_note()``
  directly:

  * NPM disabled returns ``None`` regardless of which GPU is targeted.
  * NPM enabled and ``args.gpu`` is the primary GPU returns a one-element
    ``[INFO]`` list.
  * NPM enabled and ``args.gpu`` is not the primary GPU returns a
    ``[WARNING]`` list naming the resolved primary GPU's index and the
    ``attempted`` command text verbatim.
  * No primary GPU resolved (``self.primary_gpu_handle is None``) returns
    ``None`` -- there is no GPU to suggest.

* ``TestApplyPowerStabilizationParamsPrefixMessages`` drives
  ``_apply_power_stabilization_params()`` directly:

  * With no ``prefix_messages``, behaves exactly as before this feature
    (per-field results only).
  * With ``prefix_messages``, they lead the stored/returned results list,
    followed by the normal per-field success/failure lines.
  * A multi-token apply with a prefix still stops at the first failure.
  * ``store_output``/``print_output``/``clear_multiple_devices_output`` are
    always called (no longer conditional on anything NPM-related).

* ``TestConfigurePowerStabilizationFromFileNpmNote`` drives
  ``_configure_power_stabilization_from_file()`` against a real temp file:

  * NPM disabled: replay applies all settable fields, no prefix note.
  * NPM enabled, ``gpu`` is primary: replay applies with an ``[INFO]`` note.
  * NPM enabled, ``gpu`` is not primary: replay still applies (not
    rejected) with a ``[WARNING]`` note naming ``@<path>`` as the attempted
    command.

* ``TestParsePowerStabilizationTokenReadOnlyKeys`` drives the real
  ``amdsmi_helpers.AMDSMIHelpers.parse_power_stabilization_token()``
  directly: ``host_control=...`` is silently skipped when
  ``ignore_read_only=True`` and raises ``ValueError`` otherwise.

* ``TestIsNpmEnabled`` drives the real
  ``amdsmi_helpers.AMDSMIHelpers.is_npm_enabled()`` directly:

  * ``node_handle is None`` returns ``False`` without calling the API.
  * ``AMDSMI_NPM_STATUS_ENABLED``/``_DISABLED`` map to ``True``/``False``.
  * A library exception, a sibling ``AmdSmiParameterException``, and a
    malformed return missing the ``"status"`` key (``KeyError``) all
    propagate instead of being swallowed into a fail-open ``False``.
"""

import enum
import json
import os
import sys
import tempfile
import types
import unittest

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else None

_STATUS_NOT_SUPPORTED = 2
_STATUS_INVAL = 5
_NPM_STATUS_DISABLED = 0
_NPM_STATUS_ENABLED = 1

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


class _FakeDeviceHandle:
    """Mirrors the real device handle's identity contract: comparisons go
    through ``.value`` (the underlying ctypes pointer), never Python object
    identity -- see ``AMDSMIHelpers.get_gpu_id_from_device_handle()``."""

    def __init__(self, value):
        self.value = value


class _FakeParamId:
    def __init__(self, name):
        self.name = name


_POWER_FLOOR = _FakeParamId("POWER_FLOOR")


class _FakeAmdSmiException(Exception):
    """Mirrors the real ``AmdSmiException`` base: ``AmdSmiLibraryException``
    and ``AmdSmiParameterException`` are both direct subclasses (siblings),
    not parent/child -- see ``py-interface/amdsmi_exception.py``."""


class _FakeLibraryException(_FakeAmdSmiException):
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
    wrapper.AMDSMI_STATUS_NO_PERM = 10
    wrapper.AMDSMI_STATUS_INVAL = _STATUS_INVAL
    # amdsmi_helpers.py's CPER_DECODE_MESSAGES class-body dict (unrelated to
    # this feature) is evaluated at import time and needs these to resolve.
    wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE = 100
    wrapper.AMDSMI_STATUS_UNEXPECTED_DATA = 101
    wrapper.AMDSMI_NPM_STATUS_DISABLED = _NPM_STATUS_DISABLED
    wrapper.AMDSMI_NPM_STATUS_ENABLED = _NPM_STATUS_ENABLED
    interface.amdsmi_wrapper = wrapper
    # set_value.py imports these two constants by name at module scope.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    # Overwritten per-test.
    interface.amdsmi_set_power_stabilization_param = lambda _h, _p, _v: None
    interface.amdsmi_get_npm_info = lambda _handle: {"status": wrapper.AMDSMI_NPM_STATUS_ENABLED}

    exception.AmdSmiException = _FakeAmdSmiException
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
    def __init__(self, output_format="human"):
        self.output = {}
        self._format = output_format

    def store_output(self, _gpu, key, value):
        self.output[key] = value

    def print_output(self, *args, **kwargs):
        pass

    def clear_multiple_devices_output(self):
        pass

    def is_json_format(self):
        return self._format == "json"

    def is_csv_format(self):
        return self._format == "csv"


class _StubErrorCollector:
    def __init__(self):
        self.recorded_errors = []

    def record_library_error(self, error_code):
        self.recorded_errors.append(error_code)


class _StubHelpers:
    """Minimal ``self.helpers`` stub: ``is_npm_enabled()``'s True/False
    contract plus ``get_gpu_id_from_device_handle()``'s ``.value`` lookup,
    so these tests focus on ``set_value.py``'s own dispatch behavior -- the
    real implementations are exercised directly by ``TestIsNpmEnabled``
    below.
    """

    def __init__(self, npm_enabled=False):
        self._npm_enabled = npm_enabled
        self.error_collector = _StubErrorCollector()

    def get_output_format(self):
        return "human"

    def is_npm_enabled(self, _node_handle):
        return self._npm_enabled

    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle.value


def _setup_set_value_under_test(cls, module_name):
    """Shared ``setUpClass`` body for the suites driving ``set_value.py``
    directly."""
    if not SET_VALUE_PATH or not os.path.isfile(SET_VALUE_PATH):
        raise unittest.SkipTest(
            f"amd-smi CLI set_value.py not found (looked in {_CLI_DIR or amdsmi_path})"
        )
    modules = dict.fromkeys(_CLI_MODULES)
    modules.update(_build_fake_amdsmi())
    # set_value.py does `from amdsmi_helpers import AMDSMIHelpers`; stub that
    # module directly so the real amdsmi_helpers -> amdsmi_init import chain
    # never loads. These dispatch blocks only reach AMDSMIHelpers via
    # self.helpers (stubbed below by _StubHelpers), never the bare imported
    # class.
    modules["amdsmi_helpers"] = fake_module(
        "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
    )
    stub_modules(cls, modules)
    cls.interface = modules["amdsmi.amdsmi_interface"]
    cls.module = load_cli_module(module_name, SET_VALUE_PATH, sys_path_dir=_CLI_DIR)


class TestPowerStabilizationNpmNote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_set_value_under_test(cls, "set_value_under_test_power_stabilization_npm_note")

    def _make_command(self, node_handle, primary_gpu_handle, npm_enabled):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.node_handle = node_handle
        cmd.primary_gpu_handle = primary_gpu_handle
        cmd.helpers = _StubHelpers(npm_enabled=npm_enabled)
        return cmd

    def test_npm_disabled_returns_none(self):
        cmd = self._make_command(
            node_handle=object(), primary_gpu_handle=_FakeDeviceHandle(0), npm_enabled=False
        )
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(5))

        self.assertIsNone(cmd._power_stabilization_npm_note(args, "power_floor=25"))

    def test_npm_enabled_primary_gpu_returns_info(self):
        primary = _FakeDeviceHandle(0)
        cmd = self._make_command(node_handle=object(), primary_gpu_handle=primary, npm_enabled=True)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(0))

        note = cmd._power_stabilization_npm_note(args, "power_floor=25")

        self.assertEqual(len(note), 1)
        self.assertIn("[INFO]", note[0])
        self.assertIn("node-wide", note[0])

    def test_npm_enabled_non_primary_gpu_returns_warning_with_suggested_command(self):
        primary = _FakeDeviceHandle(0)
        cmd = self._make_command(node_handle=object(), primary_gpu_handle=primary, npm_enabled=True)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(5))

        note = cmd._power_stabilization_npm_note(args, "power_floor=25")

        self.assertEqual(len(note), 1)
        self.assertIn("[WARNING]", note[0])
        self.assertIn("amd-smi set --gpu 0 --power-stabilization-configure power_floor=25", note[0])

    def test_no_primary_gpu_resolved_returns_none(self):
        cmd = self._make_command(node_handle=object(), primary_gpu_handle=None, npm_enabled=True)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(5))

        self.assertIsNone(cmd._power_stabilization_npm_note(args, "power_floor=25"))


class TestApplyPowerStabilizationParamsPrefixMessages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_set_value_under_test(cls, "set_value_under_test_power_stabilization_apply")

    def _make_command(self):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.helpers = _StubHelpers()
        return cmd

    def test_no_prefix_messages_behaves_as_before(self):
        calls = []
        self.interface.amdsmi_set_power_stabilization_param = lambda h, p, v: calls.append(
            (h, p, v)
        )
        cmd = self._make_command()
        args = types.SimpleNamespace(gpu="gpu0")

        results = cmd._apply_power_stabilization_params(args, [(_POWER_FLOOR, 80)])

        self.assertEqual(calls, [("gpu0", _POWER_FLOOR, 80)])
        self.assertEqual(len(results), 1)
        self.assertIn("Successfully set power_floor to 80", results[0])
        self.assertEqual(cmd.logger.output["power_stabilization_configure"], results)

    def test_prefix_messages_lead_the_results(self):
        calls = []
        self.interface.amdsmi_set_power_stabilization_param = lambda h, p, v: calls.append(
            (h, p, v)
        )
        cmd = self._make_command()
        args = types.SimpleNamespace(gpu="gpu0")

        results = cmd._apply_power_stabilization_params(
            args, [(_POWER_FLOOR, 80)], prefix_messages=["[INFO] NPM is enabled; ..."]
        )

        self.assertEqual(calls, [("gpu0", _POWER_FLOOR, 80)])
        self.assertEqual(len(results), 2)
        self.assertIn("[INFO]", results[0])
        self.assertIn("Successfully set power_floor to 80", results[1])
        self.assertEqual(cmd.logger.output["power_stabilization_configure"], results)

    def test_multi_token_apply_with_prefix_stops_at_first_failure(self):
        enable_param = _FakeParamId("ENABLE")
        ramp_param = _FakeParamId("RAMP_ENABLE")
        delay_param = _FakeParamId("DELAY")
        calls = []

        def _fake_set(h, p, v):
            calls.append((h, p, v))
            if p is ramp_param:
                raise _FakeLibraryException(_STATUS_NOT_SUPPORTED, "dependency unmet")

        self.interface.amdsmi_set_power_stabilization_param = _fake_set
        cmd = self._make_command()
        args = types.SimpleNamespace(gpu="gpu5")

        results = cmd._apply_power_stabilization_params(
            args,
            [(enable_param, 1), (ramp_param, 1), (delay_param, 500)],
            prefix_messages=["[WARNING] ..."],
        )

        self.assertEqual(
            calls,
            [("gpu5", enable_param, 1), ("gpu5", ramp_param, 1)],
            "delay_param must never be attempted after ramp_param's failure",
        )
        self.assertEqual(len(results), 3)
        self.assertIn("[WARNING]", results[0])
        self.assertIn("Successfully set enable to 1", results[1])
        self.assertIn("dependency unmet", results[2])


class TestConfigurePowerStabilizationFromFileNpmNote(unittest.TestCase):
    """Drives ``_configure_power_stabilization_from_file()`` against a real
    temp file, using the *real* ``AMDSMIHelpers.parse_power_stabilization_token``
    (not a re-implementation)."""

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH or not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI set_value.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = dict.fromkeys(_CLI_MODULES)
        modules.update(_build_fake_amdsmi_for_helpers())
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]

        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        # Real import (registers the real module in sys.modules) so
        # set_value.py's `from amdsmi_helpers import AMDSMIHelpers` below
        # resolves to the real parse_power_stabilization_token.
        import amdsmi_helpers  # noqa: F401

        cls.module = load_cli_module(
            "set_value_under_test_power_stabilization_npm_note_file_replay",
            SET_VALUE_PATH,
            sys_path_dir=_CLI_DIR,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def _make_command(self, node_handle=None, primary_gpu_handle=None, npm_enabled=False):
        cmd = self.module.SetValueCommands()
        cmd.logger = _FakeLogger()
        cmd.node_handle = node_handle if node_handle is not None else object()
        cmd.primary_gpu_handle = primary_gpu_handle
        cmd.helpers = _StubHelpers(npm_enabled=npm_enabled)
        return cmd

    def _write_ps_json(self, gpu_id=0):
        entry = {
            "host_control_enabled": True,
            "enabled": True,
            "power_floor": 80,
            "delay": 0,
            "insertion_priority": "DYNAMIC",
            "ramp_enabled": True,
            "ramp_max_ramp_up_rate": 10.0,
            "ramp_max_ramp_down_rate": 10.0,
            "ramp_min_time_at_power_floor": 2,
            "ramp_stop_delay": 1,
        }
        data = {"gpu_data": [{"gpu": gpu_id, "power_stabilization": entry}]}
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        self.addCleanup(os.remove, path)
        return path

    def test_npm_disabled_replay_applies_without_note(self):
        calls = []
        self.interface.amdsmi_set_power_stabilization_param = lambda h, p, v: calls.append(
            (h, p, v)
        )
        cmd = self._make_command(npm_enabled=False)
        path = self._write_ps_json(gpu_id=0)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(0))

        cmd._configure_power_stabilization_from_file(args, path, "0000:00:00.0", 0)

        results = cmd.logger.output["power_stabilization_configure"]
        # host_control_enabled is read-only and silently skipped, leaving
        # 9 of the 10 dumped fields settable.
        self.assertTrue(any("9/9" in r for r in results), results)
        self.assertFalse(any("[INFO]" in r or "[WARNING]" in r for r in results), results)

    def test_npm_enabled_primary_gpu_replay_applies_with_info_note(self):
        primary = _FakeDeviceHandle(0)
        calls = []
        self.interface.amdsmi_set_power_stabilization_param = lambda h, p, v: calls.append(
            (h, p, v)
        )
        cmd = self._make_command(primary_gpu_handle=primary, npm_enabled=True)
        path = self._write_ps_json(gpu_id=0)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(0))

        cmd._configure_power_stabilization_from_file(args, path, "0000:00:00.0", 0)

        self.assertTrue(len(calls) > 0, "replay must actually apply, not just report a note")
        results = cmd.logger.output["power_stabilization_configure"]
        self.assertIn("[INFO]", results[0])
        self.assertTrue(any("9/9" in r for r in results), results)

    def test_npm_enabled_non_primary_gpu_replay_still_applies_with_warning_note(self):
        primary = _FakeDeviceHandle(0)
        calls = []
        self.interface.amdsmi_set_power_stabilization_param = lambda h, p, v: calls.append(
            (h, p, v)
        )
        cmd = self._make_command(primary_gpu_handle=primary, npm_enabled=True)
        path = self._write_ps_json(gpu_id=5)
        args = types.SimpleNamespace(gpu=_FakeDeviceHandle(5))

        cmd._configure_power_stabilization_from_file(args, path, "0000:05:00.0", 5)

        self.assertTrue(len(calls) > 0, "replay must not be rejected -- it must still apply")
        results = cmd.logger.output["power_stabilization_configure"]
        self.assertIn("[WARNING]", results[0])
        self.assertIn(f"--power-stabilization-configure @{path}", results[0])
        self.assertTrue(any("9/9" in r for r in results), results)


class TestParsePowerStabilizationTokenReadOnlyKeys(unittest.TestCase):
    """Drives the real ``AMDSMIHelpers.parse_power_stabilization_token()``
    directly: ``host_control`` is skipped/rejected based on
    ``ignore_read_only``."""

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = dict.fromkeys(_CLI_MODULES)
        modules.update(_build_fake_amdsmi_for_helpers())
        stub_modules(cls, modules)

        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_helpers as amdsmi_helpers_module

        cls.AMDSMIHelpers = amdsmi_helpers_module.AMDSMIHelpers

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def test_host_control_skipped_when_ignore_read_only(self):
        self.assertIsNone(
            self.AMDSMIHelpers.parse_power_stabilization_token(
                "host_control=1", ignore_read_only=True
            )
        )

    def test_host_control_rejected_when_not_ignore_read_only(self):
        with self.assertRaises(ValueError):
            self.AMDSMIHelpers.parse_power_stabilization_token(
                "host_control=1", ignore_read_only=False
            )

    def test_settable_key_still_parses(self):
        param_id, value = self.AMDSMIHelpers.parse_power_stabilization_token(
            "power_floor=80", ignore_read_only=True
        )
        self.assertEqual(param_id.name, "POWER_FLOOR")
        self.assertEqual(value, 80)


# ---------------------------------------------------------------------------
# Direct tests for amdsmi_helpers.AMDSMIHelpers.is_npm_enabled()
# ---------------------------------------------------------------------------


class _FakeInitFlags:
    INIT_ALL_PROCESSORS = 0xFFFFFFFF
    INIT_AMD_GPUS = 1
    INIT_AMD_CPUS = 2
    INIT_AMD_NICS = 4


class _FakeParameterException(_FakeAmdSmiException):
    # Sibling of _FakeLibraryException (both extend _FakeAmdSmiException
    # directly), mirroring the real AmdSmiParameterException/
    # AmdSmiLibraryException relationship -- see py-interface/amdsmi_exception.py.
    pass


class _FakeHawkInsertionPriority(enum.IntEnum):
    OPPORTUNISTIC = 0
    GUARANTEED = 1
    DYNAMIC = 2


class _FakePowerStabilizationParamId(enum.IntEnum):
    ENABLE = 0
    POWER_FLOOR = 1
    DELAY = 2
    INSERTION_PRIORITY = 3
    RAMP_ENABLE = 4
    RAMP_MAX_RAMP_UP_RATE = 5
    RAMP_MAX_RAMP_DOWN_RATE = 6
    RAMP_MIN_TIME_AT_POWER_FLOOR = 7
    RAMP_STOP_DELAY = 8


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
    interface.AmdSmiPowerStabilizationParamId = _FakePowerStabilizationParamId
    interface.AmdSmiHawkInsertionPriority = _FakeHawkInsertionPriority
    return modules


class TestIsNpmEnabled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")
        modules = dict.fromkeys(_CLI_MODULES)
        modules.update(_build_fake_amdsmi_for_helpers())
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]

        cls._path_added = _CLI_DIR not in sys.path
        if cls._path_added:
            sys.path.insert(0, _CLI_DIR)

        import amdsmi_helpers as amdsmi_helpers_module

        # staticmethod() wrapping avoids auto-binding this TestCase instance
        # as `self`; see test_cli_set_node_power_limit.py's
        # TestValidateAndSetNodePowerLimit.setUpClass.
        cls.is_npm_enabled = staticmethod(amdsmi_helpers_module.AMDSMIHelpers.is_npm_enabled)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_path_added", False) and _CLI_DIR in sys.path:
            sys.path.remove(_CLI_DIR)

    def test_none_node_handle_returns_false_without_calling_api(self):
        calls = []
        self.interface.amdsmi_get_npm_info = lambda h: calls.append(h)

        self.assertFalse(self.is_npm_enabled(types.SimpleNamespace(), None))
        self.assertEqual(calls, [])

    def test_enabled_status_returns_true(self):
        self.interface.amdsmi_get_npm_info = lambda h: {"status": _NPM_STATUS_ENABLED}

        self.assertTrue(self.is_npm_enabled(types.SimpleNamespace(), object()))

    def test_disabled_status_returns_false(self):
        self.interface.amdsmi_get_npm_info = lambda h: {"status": _NPM_STATUS_DISABLED}

        self.assertFalse(self.is_npm_enabled(types.SimpleNamespace(), object()))

    def test_library_exception_propagates(self):
        # A driver read failure must fail closed -- propagate rather than
        # being silently treated as "NPM disabled" (which would let a per-GPU
        # write through on a node where NPM might actually be enabled).
        def _raise(_h):
            raise _FakeLibraryException(_STATUS_NOT_SUPPORTED, "npm info unavailable")

        self.interface.amdsmi_get_npm_info = _raise

        with self.assertRaises(_FakeLibraryException):
            self.is_npm_enabled(types.SimpleNamespace(), object())

    def test_sibling_parameter_exception_propagates(self):
        # AmdSmiParameterException is a sibling of AmdSmiLibraryException (both
        # extend AmdSmiException directly), not a subclass of it; confirm it
        # also propagates rather than being swallowed into a fail-open False.
        def _raise(_h):
            raise _FakeParameterException("unexpected parameter error")

        self.interface.amdsmi_get_npm_info = _raise

        with self.assertRaises(_FakeParameterException):
            self.is_npm_enabled(types.SimpleNamespace(), object())

    def test_malformed_info_missing_status_key_propagates(self):
        # A malformed amdsmi_get_npm_info() return missing "status" must
        # also fail closed -- propagate the KeyError rather than returning False.
        self.interface.amdsmi_get_npm_info = lambda h: {"unexpected_key": "value"}

        with self.assertRaises(KeyError):
            self.is_npm_enabled(types.SimpleNamespace(), object())
