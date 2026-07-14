#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BDF string parsing and formatting unit tests."""

"""Driver-free unit tests for the amd-smi CLI exit-code model.

Ported from the pre-migration ``tests/python_unittest/unit_tests.py`` monolith
into the ``tests/python/unit/`` tree. Exercises the record-then-finalize error
model (``AmdSmiErrorCollector``, the library-status -> POSIX-byte fold, severity
split, and the reserved 192-255 CLI-only code band) without needing a
GPU/CPU/Core or elevated permissions.
"""

import os
import sys
import unittest

from common.common import amdsmi, amdsmi_path, find_cli_dir


# Locate the CLI dir and add it to sys.path (amdsmi_path first so an AMDSMI_PATH
# override selects the matching install; see common.find_cli_dir). None ->
# setUpModule() skips.
_CLI_DIR = find_cli_dir(amdsmi_path, os.path.dirname(os.path.abspath(__file__)))
if _CLI_DIR and _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)

# Required for when the amdgpu driver is not loaded. We are required to
# fake the initialization module so the set/reset gpu CLI commands can be ran.
if "amdsmi_init" not in sys.modules:
    import types as _types

    from amdsmi import amdsmi_exception as _amdsmi_exception
    from amdsmi import amdsmi_interface as _amdsmi_interface

    _stub_amdsmi_init = _types.ModuleType("amdsmi_init")
    _stub_amdsmi_init.AMDSMI_INIT_FLAG = 0
    _stub_amdsmi_init.AMDSMI_INITIALIZED = True
    _stub_amdsmi_init.amdsmi_interface = _amdsmi_interface
    _stub_amdsmi_init.amdsmi_exception = _amdsmi_exception
    sys.modules["amdsmi_init"] = _stub_amdsmi_init

# CLI absent (rare package-only layout) -> record the error and skip in setUpModule()
# instead of failing collection with a hard ImportError.
try:
    import amdsmi_cli_exceptions as cli_exc  # noqa: E402

    _CLI_IMPORT_ERROR = None
except ImportError as _cli_import_error:
    cli_exc = None
    _CLI_IMPORT_ERROR = _cli_import_error

from amdsmi import amdsmi_wrapper  # noqa: E402


def setUpModule():
    """Skip the suite if the amd-smi CLI couldn't be imported (CLI not installed)."""
    if _CLI_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"amd-smi CLI not found ({_CLI_DIR}): {_CLI_IMPORT_ERROR}")


class TestAmdSmiCliExitCodes(unittest.TestCase):
    """Pure-logic tests for the amd-smi CLI exit-code model.

    Exercises the record-then-finalize error model in amdsmi_cli_exceptions
    without needing a GPU/CPU/Core or elevated permissions:
    the AmdSmiErrorCollector, the library-status -> POSIX-byte fold, exit codes surfaced by
    AmdSmiLibraryErrorException, the FATAL vs DEVICE severity split, and the
    reserved 192-255 CLI-only code band.
    """

    ExitCode = cli_exc.AmdSmiExitCode if cli_exc is not None else None
    Severity = cli_exc.AmdSmiErrorSeverity if cli_exc is not None else None

    # ---- AmdSmiErrorCollector / finalize ----
    def test_no_failures_resolve_to_success(self):
        collector = cli_exc.AmdSmiErrorCollector()
        self.assertEqual(collector.resolve_exit_code(), int(self.ExitCode.SUCCESS))
        self.assertFalse(collector.has_errors)

    def test_all_same_code_resolves_to_that_code(self):
        collector = cli_exc.AmdSmiErrorCollector()
        collector.record(self.ExitCode.INVALID_PARAMETER_VALUE)
        collector.record(self.ExitCode.INVALID_PARAMETER_VALUE)
        self.assertTrue(collector.has_errors)
        self.assertEqual(collector.resolve_exit_code(), int(self.ExitCode.INVALID_PARAMETER_VALUE))

    def test_mixed_codes_resolve_to_mixed(self):
        collector = cli_exc.AmdSmiErrorCollector()
        collector.record(self.ExitCode.INVALID_PARAMETER_VALUE)
        collector.record(self.ExitCode.DEVICE_NOT_FOUND)
        self.assertEqual(collector.resolve_exit_code(), int(self.ExitCode.MIXED_DEVICE_ERRORS))

    def test_reset_clears_recorded_codes(self):
        collector = cli_exc.AmdSmiErrorCollector()
        collector.record(self.ExitCode.DEVICE_NOT_FOUND)
        collector.reset()
        self.assertFalse(collector.has_errors)
        self.assertEqual(collector.resolve_exit_code(), int(self.ExitCode.SUCCESS))

    def test_record_library_error_uses_status_value(self):
        collector = cli_exc.AmdSmiErrorCollector()
        collector.record_library_error(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)
        self.assertEqual(
            collector.resolve_exit_code(), int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)
        )

    # ---- library_code_to_exit_code ----
    def test_real_status_passes_through_as_exit_code(self):
        # Real statuses (0-56) fit in a byte, so the exit code is the status.
        for status in (
            amdsmi_wrapper.AMDSMI_STATUS_INVAL,
            amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
            amdsmi_wrapper.AMDSMI_STATUS_NO_PERM,
            amdsmi_wrapper.AMDSMI_STATUS_CORRUPTED_EEPROM,
        ):
            self.assertEqual(cli_exc.library_code_to_exit_code(status), int(status))

    def test_sentinel_statuses_fold_to_low_byte(self):
        self.assertEqual(
            cli_exc.library_code_to_exit_code(amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR), 255
        )
        self.assertEqual(
            cli_exc.library_code_to_exit_code(amdsmi_wrapper.AMDSMI_STATUS_MAP_ERROR), 254
        )

    def test_library_exception_surfaces_status(self):
        exc = cli_exc.AmdSmiLibraryErrorException("", amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)
        self.assertEqual(exc.value, int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED))

    def test_every_library_status_has_a_friendly_message(self):
        """Every AMDSMI_STATUS_* the wrapper defines must have its own entry in
        AMDSMI_ERROR_MESSAGES.

        Guard: if a new status is added to the library but the message table
        isn't updated, this test fails and names the missing codes. That forces
        the author to either map the new code here, give it a CLI exit code, or
        deliberately alias it to an existing message -- a "generic"/unrecognized
        fallback is never an acceptable resting state for a defined status.
        """
        missing = [
            f"{name} ({code})"
            for code, name in amdsmi_wrapper.amdsmi_status_t__enumvalues.items()
            if abs(code) not in cli_exc.AMDSMI_ERROR_MESSAGES
        ]
        self.assertEqual(
            missing,
            [],
            f"AMDSMI_ERROR_MESSAGES is missing friendly messages for: {', '.join(missing)}",
        )

    def test_cli_command_not_supported_is_distinct_from_library(self):
        # The CLI's parse-time "command not supported" must use its own CLI code,
        # NOT the library status AMDSMI_STATUS_NOT_SUPPORTED (2), so callers can
        # tell a CLI rejection apart from a device/driver "not supported" result.
        exc = cli_exc.AmdSmiCommandNotSupportedException("static", "")
        self.assertEqual(exc.value, int(self.ExitCode.COMMAND_NOT_SUPPORTED))
        self.assertNotEqual(exc.value, int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED))

    def test_import_failure_exits_with_import_error_code(self):
        """A failed ``import amdsmi`` must exit with AmdSmiExitCode.IMPORT_ERROR,
        pulled from the enum (no stale hardcoded number).

        Runs in a subprocess because the failure path calls ``sys.exit`` at
        import time; a meta_path finder forces ``import amdsmi`` to fail
        hermetically without touching the real install.
        """
        import subprocess

        cli_dir = os.path.dirname(os.path.abspath(cli_exc.__file__))
        child = (
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'amdsmi' or name.startswith('amdsmi.'):\n"
            "            raise ImportError('amdsmi blocked for test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            f"sys.path.insert(0, {cli_dir!r})\n"
            "import amdsmi_init\n"
        )
        result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            int(self.ExitCode.IMPORT_ERROR),
            msg=f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )

    # ---- severity classification ----
    def test_library_error_is_device_severity(self):
        # A per-device library failure must be recorded (not abort the command).
        self.assertEqual(cli_exc.AmdSmiLibraryErrorException.severity, self.Severity.DEVICE)

    def test_cli_errors_default_to_fatal_severity(self):
        # Anything not explicitly per-device stops the whole command.
        self.assertEqual(cli_exc.AmdSmiException.severity, self.Severity.FATAL)
        self.assertEqual(cli_exc.AmdSmiRequiredCommandException.severity, self.Severity.FATAL)

    # ---- CLI-only exit-code band ----
    def test_cli_only_codes_live_above_library_range(self):
        # CLI-invented codes sit above the library status range (max 56) so they
        # can never be mistaken for / collide with a real status. The band
        # bounds come from amdsmi_cli_exceptions, not hardcoded here.
        band_start = cli_exc.CLI_EXIT_CODE_BAND_START
        band_end = cli_exc.CLI_EXIT_CODE_BAND_END
        for code in self.ExitCode:
            if code is self.ExitCode.SUCCESS:
                continue
            self.assertGreaterEqual(
                int(code), band_start, f"{code.name} not in {band_start}-{band_end} band"
            )
            self.assertLessEqual(
                int(code), band_end, f"{code.name} not in {band_start}-{band_end} band"
            )

    def test_exit_codes_are_unique(self):
        values = [int(c) for c in self.ExitCode]
        self.assertEqual(len(values), len(set(values)), "duplicate exit-code values")

    def test_cli_codes_never_collide_with_library_exit_codes(self):
        """Every exit code must mean exactly one thing: either a library status
        or a CLI-invented code, never both.

        A band check alone isn't enough, because the two library sentinels also
        land in the CLI's 192-255 band (AMDSMI_STATUS_MAP_ERROR -> 254,
        AMDSMI_STATUS_UNKNOWN_ERROR -> 255). So this asserts directly that no CLI
        code equals any exit code a library status can produce -- otherwise a
        caller couldn't tell a library failure from a CLI one by exit code alone.
        SUCCESS (0) is shared with the library's success status by design and is
        exempt.

        Because it checks every CLI code, this also covers the normalized init
        codes: INIT_TIMEOUT, DRIVERS_NOT_LOADED and DEVICE_NOT_FOUND are proven
        distinct from the library statuses they map (TIMEOUT, NOT_INIT,
        DRIVER_NOT_LOADED), so no separate per-code test is needed.
        """
        library_exit_codes = {
            cli_exc.library_code_to_exit_code(code)
            for code in amdsmi_wrapper.amdsmi_status_t__enumvalues
        }
        for code in self.ExitCode:
            if code is self.ExitCode.SUCCESS:
                continue
            self.assertNotIn(
                int(code),
                library_exit_codes,
                f"CLI code {code.name} ({int(code)}) collides with a library status exit code",
            )

    # ---- store_device_error choke point (regression guard) ----
    def test_store_device_error_shows_and_records_failure(self):
        """A device failure routed through store_device_error must both display
        the message AND record the code, so the process exit stays non-zero.

        This guards the record-then-finalize fix for set/reset: forgetting the
        record (the original `set -M` bug) would leave resolve_exit_code() at 0.
        """
        from amdsmi_helpers import AMDSMIHelpers

        class _FakeLogger:
            def __init__(self):
                self.stored = []

            def store_output(self, device, key, value):
                self.stored.append((device, key, value))

        class _FakeLibErr(Exception):
            def get_error_code(self):
                return amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED

        helpers = AMDSMIHelpers()
        logger = _FakeLogger()
        helpers.store_device_error(
            logger,
            0,
            "memory_partition",
            "[AMDSMI_STATUS_NOT_SUPPORTED] ...",
            exception=_FakeLibErr(),
        )
        # Shown to the user...
        self.assertEqual(logger.stored[-1][1], "memory_partition")
        # ...and counted, so the exit code reflects the failure.
        self.assertTrue(helpers.error_collector.has_errors)
        self.assertEqual(
            helpers.error_collector.resolve_exit_code(),
            int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED),
        )

    def test_store_device_error_cli_code_path(self):
        """A CLI-level (non-library) failure records the given CLI exit code."""
        from amdsmi_helpers import AMDSMIHelpers

        class _FakeLogger:
            def store_output(self, device, key, value):
                pass

        helpers = AMDSMIHelpers()
        helpers.store_device_error(
            _FakeLogger(),
            0,
            "clk_limit",
            "Invalid clock type",
            code=int(self.ExitCode.INVALID_PARAMETER_VALUE),
        )
        self.assertEqual(
            helpers.error_collector.resolve_exit_code(), int(self.ExitCode.INVALID_PARAMETER_VALUE)
        )

    def test_store_device_error_requires_exception_or_code(self):
        """Calling store_device_error with neither exception= nor code= must
        raise TypeError, not silently show the message and record nothing.

        Guards the guard: showing a per-device error without recording it would
        leave resolve_exit_code() at 0 (the original silent exit-0 bug). This
        locks the fail-fast behavior in so it can't be quietly removed.
        """
        from amdsmi_helpers import AMDSMIHelpers

        class _FakeLogger:
            def store_output(self, device, key, value):
                pass

        helpers = AMDSMIHelpers()
        with self.assertRaises(TypeError):
            helpers.store_device_error(_FakeLogger(), 0, "key", "msg")
        # Nothing should have been recorded when the guard fires.
        self.assertFalse(helpers.error_collector.has_errors)

    # ---- multi-field CPU/core set aggregation ----
    def test_set_core_records_each_failed_field_to_mixed(self):
        """Drive the real set_core path: two fields fail with DIFFERENT library
        statuses. set_core must record BOTH (not collapse to the first) and the
        collector must finalize to MIXED_DEVICE_ERRORS. This exercises the
        actual per-field record-then-finalize code in set_core, not just the
        collector in isolation.

        handle_cores short-circuits on a non-list handle, and the id lookup and
        the two library set calls are replaced with stand-ins that raise the two
        errors under test.
        """
        import argparse
        import os
        from unittest import mock
        from amdsmi_helpers import AMDSMIHelpers

        # Import the set_value subcommand as a standalone module (it uses only
        # absolute imports), so no driver-backed package init is triggered.
        sub_dir = os.path.join(os.path.dirname(cli_exc.__file__), "subcommands")
        if sub_dir not in sys.path:
            sys.path.append(sub_dir)
        import set_value

        class _FakeLogger:
            format = "human"

            def store_core_output(self, device, key, value):
                pass

            def print_output(self, multiple_device_enabled=False):
                pass

            def store_multiple_device_output(self):
                pass

        # A library exception that needs no driver: subclass the real type (so
        # set_core's `except amdsmi_exception.AmdSmiLibraryException` catches it)
        # but skip its C-backed __init__.
        class _FakeLibErr(amdsmi.AmdSmiLibraryException):
            def __init__(self, code):
                self._code = code

            def get_error_code(self):
                return self._code

            def get_error_info(self, detailed=True):
                return f"status {self._code}"

        def _raise(code):
            def _inner(*args, **kwargs):
                raise _FakeLibErr(code)

            return _inner

        cmd = set_value.SetValueCommands()
        cmd.helpers = AMDSMIHelpers()
        cmd.logger = _FakeLogger()
        # Avoid the driver-backed core-id lookup.
        cmd.helpers.get_core_id_from_device_handle = lambda handle: 0

        args = argparse.Namespace(
            core="fake-core-handle",  # non-list, non-None -> handle_cores returns (False, handle)
            core_boost_limit=[[100]],
            core_floor_limit=[[100]],
            core_msr_floor_limit=None,
        )

        patch_boost = mock.patch.object(
            set_value.amdsmi_interface,
            "amdsmi_set_cpu_core_boostlimit",
            _raise(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED),
        )
        patch_floor = mock.patch.object(
            set_value.amdsmi_interface,
            "amdsmi_set_cpu_core_floor_freq_limit",
            _raise(amdsmi_wrapper.AMDSMI_STATUS_INVAL),
        )
        with patch_boost, patch_floor:
            cmd.set_core(args)

        self.assertTrue(cmd.helpers.error_collector.has_errors)
        self.assertEqual(
            cmd.helpers.error_collector.resolve_exit_code(), int(self.ExitCode.MIXED_DEVICE_ERRORS)
        )

    # ---- multi-DEVICE aggregation (drives handle_cores/run_device_subcommand) ----
    def test_handle_cores_aggregates_multiple_device_failures_to_mixed(self):
        """Two cores each fail with a DIFFERENT library status. handle_cores
        routes each through run_device_subcommand, which records per device and
        keeps going; the collector finalizes to MIXED_DEVICE_ERRORS. Covers the
        multi-DEVICE record-then-continue path (the set_core test covers the
        multi-FIELD path).
        """
        import argparse
        from amdsmi_helpers import AMDSMIHelpers

        class _FakeLogger:
            def print_output(self, multiple_device_enabled=False):
                pass

        class _FakeLibErr(amdsmi.AmdSmiLibraryException):
            def __init__(self, code):
                self._code = code

            def get_error_code(self):
                return self._code

        codes = {
            "core-0": amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED,
            "core-1": amdsmi_wrapper.AMDSMI_STATUS_INVAL,
        }

        def fake_subcommand(args, multiple_devices=False, core=None):
            raise _FakeLibErr(codes[core])

        helpers = AMDSMIHelpers()
        args = argparse.Namespace(core=["core-0", "core-1"])
        handled, _ = helpers.handle_cores(args, _FakeLogger(), fake_subcommand)
        self.assertTrue(handled)
        self.assertEqual(
            helpers.error_collector.resolve_exit_code(), int(self.ExitCode.MIXED_DEVICE_ERRORS)
        )

    # ---- rocm-smi compat exit-code contract ----
    def test_rocm_smi_compat_exit_codes_stay_binary(self):
        """The --rocm-smi shim intentionally follows rocm-smi's BINARY 0/1 exit
        convention, not amd-smi's 192+ band. Guards against someone 'upgrading'
        it to AmdSmiExitCode values, which would break scripts targeting
        rocm-smi's contract.
        """
        import amdsmi_rocm_smi_compat as compat

        self.assertEqual(int(compat.RocmSmiCompatExitCode.SUCCESS), 0)
        self.assertEqual(int(compat.RocmSmiCompatExitCode.ERROR), 1)
        self.assertEqual({int(c) for c in compat.RocmSmiCompatExitCode}, {0, 1})

    # ---- CLI arg guard (drives real set_value) ----
    def test_gtt_with_gpu_raises_invalid_parameter(self):
        """`set --gtt` combined with `--gpu` must raise
        AmdSmiInvalidParameterException (CLI code INVALID_PARAMETER), NOT
        sys.exit(2) which aliases library NOT_SUPPORTED. Drives the real
        set_value guard, which runs before any device dispatch.
        """
        import argparse
        import os
        from amdsmi_helpers import AMDSMIHelpers

        sub_dir = os.path.join(os.path.dirname(cli_exc.__file__), "subcommands")
        if sub_dir not in sys.path:
            sys.path.append(sub_dir)
        import set_value

        cmd = set_value.SetValueCommands()
        cmd.helpers = AMDSMIHelpers()

        args = argparse.Namespace(gtt=8.0, gpu=["fake-gpu-handle"])
        with self.assertRaises(cli_exc.AmdSmiInvalidParameterException) as ctx:
            cmd.set_value(args)
        self.assertEqual(ctx.exception.value, int(self.ExitCode.INVALID_PARAMETER))

    # ---- interactive confirmation decline (drives confirm_out_of_spec_warning) ----
    def test_confirmation_decline_exits_user_aborted(self):
        """Declining an interactive confirmation prompt must exit with
        USER_ABORTED, not a raw sys.exit(1)/sys.exit(str) (which aliases library
        INVAL). Drives the real confirm_out_of_spec_warning with a 'no' response;
        auto_respond bypasses the input() prompt so it doesn't block on a TTY.
        """
        import io
        import contextlib
        from amdsmi_helpers import AMDSMIHelpers

        helpers = AMDSMIHelpers()
        silence_out = contextlib.redirect_stdout(io.StringIO())
        silence_err = contextlib.redirect_stderr(io.StringIO())
        with self.assertRaises(SystemExit) as ctx, silence_out, silence_err:
            helpers.confirm_out_of_spec_warning(auto_respond="n")
        self.assertEqual(ctx.exception.code, int(self.ExitCode.USER_ABORTED))

    # ---- NO_PERM is fatal (drives set_core) ----
    def test_set_core_no_perm_raises_and_does_not_record(self):
        """A permission error is not a per-device failure -- it means the user
        needs sudo, which affects the whole command. So when a set call returns
        NO_PERM, set_core should stop immediately by raising PermissionError,
        instead of recording the error and continuing to the next field/device.

        This checks both halves: PermissionError is raised, and nothing was
        recorded in the collector. Uses the same fakes as the multi-field test
        above.
        """
        import argparse
        import os
        from unittest import mock
        from amdsmi_helpers import AMDSMIHelpers

        sub_dir = os.path.join(os.path.dirname(cli_exc.__file__), "subcommands")
        if sub_dir not in sys.path:
            sys.path.append(sub_dir)
        import set_value

        class _FakeLogger:
            format = "human"

            def store_core_output(self, device, key, value):
                pass

            def print_output(self, multiple_device_enabled=False):
                pass

            def store_multiple_device_output(self):
                pass

        class _FakeLibErr(amdsmi.AmdSmiLibraryException):
            def __init__(self, code):
                self._code = code

            def get_error_code(self):
                return self._code

            def get_error_info(self, detailed=True):
                return f"status {self._code}"

        cmd = set_value.SetValueCommands()
        cmd.helpers = AMDSMIHelpers()
        cmd.logger = _FakeLogger()
        cmd.helpers.get_core_id_from_device_handle = lambda handle: 0

        args = argparse.Namespace(
            core="fake-core-handle",
            core_boost_limit=[[100]],
            core_floor_limit=None,
            core_msr_floor_limit=None,
        )

        def _raise_no_perm(*a, **k):
            raise _FakeLibErr(amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)

        with mock.patch.object(
            set_value.amdsmi_interface, "amdsmi_set_cpu_core_boostlimit", _raise_no_perm
        ):
            with self.assertRaises(PermissionError):
                cmd.set_core(args)

        # NO_PERM aborts the command; it must NOT be recorded as a device error.
        self.assertFalse(cmd.helpers.error_collector.has_errors)


# ---------------------------------------------------------------------------
# Generic "-g all" set-failure guards
#
# Lock in the record-then-finalize contract for the per-device GPU `set`
# handlers: a per-device library failure must be RECORDED and the command must
# keep going -- never an unhandled crash. This is the class of bug that broke
# `amd-smi set -M/-C ... -g all` (a handler raising instead of recording). The
# completeness test forces every GPU `set` option to have a guard here.
# ---------------------------------------------------------------------------


class _DriverlessLibErr(amdsmi.AmdSmiLibraryException):
    """AmdSmiLibraryException stand-in needing no driver.

    Subclasses the real type so a handler's ``except
    amdsmi_exception.AmdSmiLibraryException`` catches it, but skips the C-backed
    __init__.
    """

    def __init__(self, code):
        self._code = code

    def get_error_code(self):
        return self._code

    def get_error_info(self, detailed=True):
        return f"status {self._code}"


class _FakeGpuLogger:
    """Minimal logger covering the calls set_gpu makes on a failure path."""

    format = "human"

    def __init__(self):
        self.stored = []

    def store_output(self, device, key, value):
        self.stored.append((device, key, value))

    def print_output(self, multiple_device_enabled=False):
        pass

    def clear_multiple_devices_output(self):
        pass

    def store_multiple_device_output(self):
        pass


def _load_set_value():
    sub_dir = os.path.join(os.path.dirname(cli_exc.__file__), "subcommands")
    if sub_dir not in sys.path:
        sys.path.append(sub_dir)
    import set_value

    return set_value


def _patch_amdsmi_interface(set_value, **overrides):
    """Context manager patching one or more amdsmi_interface functions.

    The driverless tests feed a fake (str) device handle, so every real
    amdsmi_interface call that dereferences it as a ``c_void_p`` must be
    stubbed -- including the handle-consuming reads set_gpu runs before the
    handler (e.g. ``amdsmi_get_gpu_device_bdf``).
    """
    import contextlib
    from unittest import mock

    stack = contextlib.ExitStack()
    for name, fn in overrides.items():
        stack.enter_context(mock.patch.object(set_value.amdsmi_interface, name, fn))
    return stack


def _raise_not_supported(*args, **kwargs):
    raise _DriverlessLibErr(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED)


def _fake_bdf(handle):
    return "0000:00:00.0"


def _noop(*args, **kwargs):
    """Success-path stand-in: a patched amdsmi call that returns cleanly."""
    return None


def _gpu_set_options(set_value):
    """Authoritative GPU ``set`` option dests, sourced from set_gpu's own
    required-arg check (the ``getattr(args, "X")`` list). Auto-updates when a
    new option is added -- that is what makes the completeness test a real
    forcing function.
    """
    import inspect
    import re

    src = inspect.getsource(set_value.SetValueCommands.set_gpu)
    return set(re.findall(r'getattr\(args,\s*"(\w+)"', src))


def _make_set_gpu_cmd(set_value):
    from amdsmi_helpers import AMDSMIHelpers

    cmd = set_value.SetValueCommands()
    cmd.helpers = AMDSMIHelpers()
    cmd.logger = _FakeGpuLogger()
    cmd.device_handles = []
    cmd.group_check_printed = True
    cmd.helpers.is_baremetal = lambda: True
    cmd.helpers.get_gpu_id_from_device_handle = lambda handle: 0
    return cmd


def _gpu_set_args(set_value, **option):
    """Namespace with every arg set_gpu references defaulted to None, plus a
    single-device gpu handle and the one option under test."""
    import argparse
    import inspect
    import re

    src = inspect.getsource(set_value.SetValueCommands.set_gpu)
    names = set(re.findall(r"args\.(\w+)", src))
    ns = argparse.Namespace(**{n: None for n in names})
    ns.gpu = "fake-gpu-handle"  # non-list, non-None -> single-device path
    for key, value in option.items():
        setattr(ns, key, value)
    return ns


def _build_set_specs(set_value):
    """Per-option drivers for the generic GPU ``set`` exercisers.

    Each entry is ``name -> configure(cmd, mode)`` where ``mode`` is ``"fail"``
    or ``"success"``. ``configure`` installs any helper stubs the handler needs
    on a driverless box and returns ``(arg_value, amdsmi_patches)``:

      * ``arg_value``      -- value placed on ``args.<name>`` to enter the handler.
      * ``amdsmi_patches`` -- ``{amdsmi_interface_fn: replacement}``:
          - mode="fail":    the earliest handle-consuming call raises
            NOT_SUPPORTED (must be *recorded*, not raised past the handler).
          - mode="success": every call returns a valid value, so the handler
            finishes clean -- nothing recorded, exit 0.

    ``amdsmi_get_gpu_device_bdf`` is patched by the caller for every option
    (set_gpu reads it up front to build the error string).

    The registry keys are the single source of truth for the completeness test,
    so adding a new GPU ``set`` option without a spec here fails the suite --
    that is the forcing function that blocks a future ``-g all`` regression.
    """
    import collections

    ai = set_value.amdsmi_interface

    def first_member(enum):
        return next(n for n in enum.__members__ if n != "INVALID")

    ClkLevel = collections.namedtuple("ClkLevel", "clk_type perf_levels")
    ClkLimit = collections.namedtuple("ClkLimit", "clk_type lim_type val")
    PowerCap = collections.namedtuple("PowerCap", "pwr_type watts")

    class _Fmt:  # PTL format element: only .name is read on the failure path.
        def __init__(self, name):
            self.name = name

    def fan(cmd, mode):
        cmd.helpers.detect_gpu_od = lambda bdf: (False, None)  # legacy hwmon path
        setfn = _raise_not_supported if mode == "fail" else _noop
        return (50, True), {"amdsmi_set_gpu_fan_speed": setfn}

    def perf_level(cmd, mode):
        if mode == "fail":
            cmd.helpers.get_perf_levels = lambda: (["AUTO", "LOW"],)
        setfn = _raise_not_supported if mode == "fail" else _noop
        return first_member(ai.AmdSmiDevPerfLevel), {"amdsmi_set_gpu_perf_level": setfn}

    def profile(cmd, mode):
        cmd.helpers.get_power_profile_name_mapping = lambda: {"CUSTOM": 1}
        if mode == "fail":
            return "CUSTOM", {
                "amdsmi_set_gpu_power_profile": _raise_not_supported,
                "amdsmi_get_gpu_power_profile_presets": _raise_not_supported,
            }
        return "CUSTOM", {"amdsmi_set_gpu_power_profile": _noop}

    def perf_determinism(cmd, mode):
        setfn = _raise_not_supported if mode == "fail" else _noop
        return 800, {"amdsmi_set_gpu_perf_determinism_mode": setfn}

    def compute_partition(cmd, mode):
        name = first_member(ai.AmdSmiComputePartitionType)
        # Force the "profiles not enumerable" fallback so a valid TYPE name is
        # attempted and the driver's status is what surfaces.
        cmd.helpers.get_accelerator_choices_types_indices = lambda: (
            [name],
            {"profile_types": [], "profile_indices": []},
        )
        setfn = _raise_not_supported if mode == "fail" else _noop
        return name, {"amdsmi_set_gpu_compute_partition": setfn}

    def memory_partition(cmd, mode):
        cmd.helpers.confirm_changing_memory_partition_gpu_reload_warning = lambda *a, **k: None
        arg = first_member(ai.AmdSmiMemoryPartitionType)
        if mode == "fail":
            return arg, {
                "amdsmi_get_gpu_memory_partition_config": _raise_not_supported,
                "amdsmi_set_gpu_memory_partition": _raise_not_supported,
            }
        return arg, {
            "amdsmi_get_gpu_memory_partition_config": lambda *a, **k: {
                "partition_caps": [arg],
                "mp_mode": arg,
            },
            "amdsmi_set_gpu_memory_partition": _noop,
        }

    def soc_pstate(cmd, mode):
        setfn = _raise_not_supported if mode == "fail" else _noop
        return 0, {"amdsmi_set_soc_pstate": setfn}

    def xgmi_plpd(cmd, mode):
        setfn = _raise_not_supported if mode == "fail" else _noop
        return 0, {"amdsmi_set_xgmi_plpd": setfn}

    def clk_level(cmd, mode):
        if mode == "fail":
            # Fails at the first step (set perf level -> MANUAL), which is recorded.
            return ClkLevel("sclk", [0]), {"amdsmi_set_gpu_perf_level": _raise_not_supported}
        return ClkLevel("sclk", [0]), {
            "amdsmi_set_gpu_perf_level": _noop,
            "amdsmi_get_clk_freq": lambda *a, **k: {"num_supported": 8},
            "amdsmi_set_clk_freq": _noop,
        }

    def ptl_status(cmd, mode):
        setfn = _raise_not_supported if mode == "fail" else _noop
        return 1, {"amdsmi_set_gpu_ptl_state": setfn}

    def ptl_format(cmd, mode):
        arg = (_Fmt("NC"), _Fmt("NC"))
        if mode == "fail":
            return arg, {"amdsmi_get_gpu_ptl_formats": _raise_not_supported}
        ptl_val = list(ai.AmdSmiPtlData)[0].value
        return arg, {
            "amdsmi_get_gpu_ptl_formats": lambda *a, **k: (ptl_val, ptl_val),
            "amdsmi_set_gpu_ptl_formats": _noop,
        }

    def power_cap(cmd, mode):
        # Delegates to a helper that owns its own error recording; emulate that
        # helper's per-mode behavior so the handler wiring is exercised without a
        # real power-cap validation flow.
        if mode == "fail":

            def _delegate(*a, **k):
                cmd.helpers.error_collector.record_library_error(
                    amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                )
                return "[AMDSMI_STATUS_NOT_SUPPORTED] Unable to set power cap"

        else:

            def _delegate(*a, **k):
                return "Successfully set power cap"

        cmd.helpers.validate_and_set_power_cap = _delegate
        return PowerCap("ppt0", 100), {}

    def clk_limit(cmd, mode):
        if mode == "fail":
            return ClkLimit("sclk", "min", 100), {"amdsmi_get_clock_info": _raise_not_supported}
        return ClkLimit("sclk", "min", 1000), {
            "amdsmi_get_clock_info": lambda *a, **k: {"max_clk": 2000, "min_clk": 500},
            "amdsmi_set_gpu_clk_limit": _noop,
        }

    def process_isolation(cmd, mode):
        if mode == "fail":
            return 1, {"amdsmi_get_gpu_process_isolation": _raise_not_supported}
        return 1, {
            "amdsmi_get_gpu_process_isolation": lambda *a, **k: 0,
            "amdsmi_set_gpu_process_isolation": _noop,
        }

    def mem_carveout(cmd, mode):
        cmd.helpers.prompt_reboot = lambda *a, **k: None
        if mode == "fail":
            return 0, {"amdsmi_get_gpu_uma_carveout_info": _raise_not_supported}
        return 0, {
            "amdsmi_get_gpu_uma_carveout_info": lambda *a, **k: {
                "options": [{"description": "default"}],
                "current_index": -1,
            },
            "amdsmi_set_gpu_uma_carveout": _noop,
        }

    def compute_partition_mem_alloc_mode(cmd, mode):
        setfn = _raise_not_supported if mode == "fail" else _noop
        return first_member(ai.AmdSmiComputePartitionMemAllocModeType), {
            "amdsmi_set_gpu_compute_partition_mem_alloc_mode": setfn
        }

    return {
        "fan": fan,
        "perf_level": perf_level,
        "profile": profile,
        "perf_determinism": perf_determinism,
        "compute_partition": compute_partition,
        "memory_partition": memory_partition,
        "soc_pstate": soc_pstate,
        "xgmi_plpd": xgmi_plpd,
        "clk_level": clk_level,
        "ptl_status": ptl_status,
        "ptl_format": ptl_format,
        "power_cap": power_cap,
        "clk_limit": clk_limit,
        "process_isolation": process_isolation,
        "mem_carveout": mem_carveout,
        "compute_partition_mem_alloc_mode": compute_partition_mem_alloc_mode,
    }


class TestSetGpuGAllFailureGuards(unittest.TestCase):
    """A per-device GPU `set` failure must be recorded and must NOT crash."""

    def test_every_gpu_set_option_is_driven(self):
        """Forcing function: every GPU `set` option must have a driver in
        _build_set_specs. Adding a new option without a spec fails here, which
        is what blocks a future `-g all` regression from shipping untested."""
        set_value = _load_set_value()
        options = _gpu_set_options(set_value)
        specced = set(_build_set_specs(set_value))
        self.assertEqual(
            options,
            specced,
            "GPU set options and their -g all failure drivers are out of sync.\n"
            f"  unspecced (add a _build_set_specs entry): {sorted(options - specced)}\n"
            f"  stale (spec exists but no longer an option): {sorted(specced - options)}",
        )

    def test_every_gpu_set_option_records_on_failure_not_crashes(self):
        """Generic exerciser: drive each GPU `set` option through set_gpu with a
        per-device library failure injected, and assert the record-then-finalize
        contract holds -- the failure is recorded (exit != 0) and the handler
        returns instead of raising. This is the class of bug that broke
        `amd-smi set ... -g all` for -M/-C in #862."""
        set_value = _load_set_value()
        specs = _build_set_specs(set_value)
        for name in sorted(specs):
            with self.subTest(option=name):
                cmd = _make_set_gpu_cmd(set_value)
                arg_value, patches = specs[name](cmd, "fail")
                args = _gpu_set_args(set_value, **{name: arg_value})
                with _patch_amdsmi_interface(
                    set_value, amdsmi_get_gpu_device_bdf=_fake_bdf, **patches
                ):
                    cmd.set_gpu(args)  # must NOT raise
                self.assertTrue(
                    cmd.helpers.error_collector.has_errors,
                    f"{name}: per-device failure was not recorded (would exit 0)",
                )
                self.assertNotEqual(
                    cmd.helpers.error_collector.resolve_exit_code(),
                    0,
                    f"{name}: exit code resolved to 0 despite a recorded failure",
                )

    def test_every_gpu_set_option_succeeds_cleanly(self):
        """Mirror contract: a successful per-device `set` records NOTHING and
        finalizes to exit 0. Guards the opposite regression from #862 -- a
        success path that wrongly records an error, crashes, or resolves to a
        non-zero exit code."""
        set_value = _load_set_value()
        specs = _build_set_specs(set_value)
        for name in sorted(specs):
            with self.subTest(option=name):
                cmd = _make_set_gpu_cmd(set_value)
                arg_value, patches = specs[name](cmd, "success")
                args = _gpu_set_args(set_value, **{name: arg_value})
                with _patch_amdsmi_interface(
                    set_value, amdsmi_get_gpu_device_bdf=_fake_bdf, **patches
                ):
                    cmd.set_gpu(args)  # must NOT raise
                self.assertFalse(
                    cmd.helpers.error_collector.has_errors,
                    f"{name}: success path wrongly recorded an error",
                )
                self.assertEqual(
                    cmd.helpers.error_collector.resolve_exit_code(),
                    0,
                    f"{name}: success path did not resolve to exit 0",
                )

    def test_g_all_memory_partition_failure_records_not_crashes(self):
        """`set -M ... -g all`: a per-device NOT_SUPPORTED is recorded and the
        handler returns (must not raise/crash)."""
        set_value = _load_set_value()
        cmd = _make_set_gpu_cmd(set_value)
        # memory_partition prompts a reload confirmation on the first set.
        cmd.helpers.confirm_changing_memory_partition_gpu_reload_warning = lambda *a, **k: None
        mp_name = next(
            n
            for n in set_value.amdsmi_interface.AmdSmiMemoryPartitionType.__members__
            if n != "INVALID"
        )
        args = _gpu_set_args(set_value, memory_partition=mp_name)

        with _patch_amdsmi_interface(
            set_value,
            amdsmi_get_gpu_device_bdf=_fake_bdf,
            amdsmi_get_gpu_memory_partition_config=_raise_not_supported,
            amdsmi_set_gpu_memory_partition=_raise_not_supported,
        ):
            cmd.set_gpu(args)  # must NOT raise

        self.assertTrue(
            cmd.helpers.error_collector.has_errors,
            "memory_partition per-device failure was not recorded (would exit 0)",
        )
        self.assertEqual(
            cmd.helpers.error_collector.resolve_exit_code(),
            int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED),
            "memory_partition failure did not finalize to the library exit code",
        )
        messages = [v for (_d, key, v) in cmd.logger.stored if key == "memory_partition"]
        self.assertTrue(
            messages and "memory partition" in messages[-1].lower(),
            f"memory_partition error message not surfaced to the user: {messages}",
        )

    def test_g_all_compute_partition_failure_records_not_crashes(self):
        """`set -C ... -g all`: a per-device NOT_SUPPORTED is recorded and the
        handler returns (must not raise/crash)."""
        set_value = _load_set_value()
        cmd = _make_set_gpu_cmd(set_value)
        cp_name = next(
            n
            for n in set_value.amdsmi_interface.AmdSmiComputePartitionType.__members__
            if n != "INVALID"
        )
        # Force the "profiles not enumerable" fallback so a valid TYPE name is
        # attempted and the driver's status is what surfaces.
        cmd.helpers.get_accelerator_choices_types_indices = lambda: (
            [cp_name],
            {"profile_types": [], "profile_indices": []},
        )
        args = _gpu_set_args(set_value, compute_partition=cp_name)

        with _patch_amdsmi_interface(
            set_value,
            amdsmi_get_gpu_device_bdf=_fake_bdf,
            amdsmi_set_gpu_compute_partition=_raise_not_supported,
        ):
            cmd.set_gpu(args)  # must NOT raise

        self.assertTrue(
            cmd.helpers.error_collector.has_errors,
            "compute_partition per-device failure was not recorded (would exit 0)",
        )
        self.assertEqual(
            cmd.helpers.error_collector.resolve_exit_code(),
            int(amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED),
            "compute_partition failure did not finalize to the library exit code",
        )
        messages = [v for (_d, key, v) in cmd.logger.stored if key == "accelerator_partition"]
        self.assertTrue(
            messages and "accelerator partition" in messages[-1].lower(),
            f"compute_partition error message not surfaced to the user: {messages}",
        )


# ---------------------------------------------------------------------------
# Generic "-g all" reset-failure guards
#
# Same record-then-finalize contract as the set guards above, applied to the
# per-device GPU `reset` handlers: a per-device library failure must be
# RECORDED and the command must keep going -- never an unhandled crash. `gtt`
# is intentionally excluded (it is a system-wide reset, rejected with --gpu and
# dispatched before the per-device loop).
# ---------------------------------------------------------------------------


def _load_reset():
    sub_dir = os.path.join(os.path.dirname(cli_exc.__file__), "subcommands")
    if sub_dir not in sys.path:
        sys.path.append(sub_dir)
    import reset

    return reset


def _reset_gpu_options(reset):
    """Authoritative per-device GPU ``reset`` option dests, sourced from
    reset_gpu's own required-arg ``any([...])`` gate. Auto-updates when a new
    option is added -- that is what makes the completeness test a forcing
    function. ``gtt``/``mem_carveout`` are not in the gate (system-wide / no
    per-device handler) and are correctly excluded.
    """
    import inspect
    import re

    src = inspect.getsource(reset.ResetCommands.reset)
    names = set()
    for group in re.findall(r"if not any\(\s*\[(.*?)\]\s*\)", src, re.DOTALL):
        names |= set(re.findall(r"args\.(\w+)", group))
    return names


def _make_reset_cmd(reset):
    from amdsmi_helpers import AMDSMIHelpers

    cmd = reset.ResetCommands()
    cmd.helpers = AMDSMIHelpers()
    cmd.logger = _FakeGpuLogger()
    cmd.device_handles = []
    cmd.group_check_printed = True
    cmd.helpers.is_baremetal = lambda: True
    cmd.helpers.get_gpu_id_from_device_handle = lambda handle: 0
    return cmd


def _reset_args(reset, **option):
    """Namespace with every arg reset references defaulted to None, plus a
    single-device gpu handle and the one option under test set truthy."""
    import argparse
    import inspect
    import re

    src = inspect.getsource(reset.ResetCommands.reset)
    names = set(re.findall(r"args\.(\w+)", src))
    ns = argparse.Namespace(**{n: None for n in names})
    ns.gpu = "fake-gpu-handle"  # non-list, non-None -> single-device path
    for key, value in option.items():
        setattr(ns, key, value)
    return ns


def _build_reset_specs(reset):
    """Per-option drivers for the generic GPU ``reset`` exercisers.

    Each entry is ``name -> configure(cmd, mode)`` where ``mode`` is ``"fail"``
    or ``"success"``. ``configure`` installs any helper stubs the handler needs
    and returns the ``{amdsmi_interface_fn: replacement}`` map:

      * mode="fail":    the failure path raises NOT_SUPPORTED (must be recorded).
      * mode="success": every call returns cleanly (nothing recorded, exit 0).

    Every reset option is a boolean flag, so the exerciser sets
    ``args.<name> = True``; there is no per-option value to build.

    The registry keys are the single source of truth for the completeness test,
    so adding a new per-device ``reset`` option without a spec here fails the
    suite -- the forcing function that blocks a future ``-g all`` regression.
    """

    def gpureset(cmd, mode):
        cmd.helpers.is_amd_device = lambda gpu: True
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_reset_gpu": fn}

    def clocks(cmd, mode):
        # Overdrive is reset first, then perf level (twice) with the same
        # handle; both must be stubbed so a fake handle never reaches hardware.
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_set_gpu_overdrive_level": fn, "amdsmi_set_gpu_perf_level": fn}

    def fans(cmd, mode):
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_reset_gpu_fan": fn}

    def profile(cmd, mode):
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_set_gpu_power_profile": fn}

    def xgmierr(cmd, mode):
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_reset_gpu_xgmi_error": fn}

    def perf_determinism(cmd, mode):
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_set_gpu_perf_level": fn}

    def power_cap(cmd, mode):
        if mode == "fail":
            return {"amdsmi_get_supported_power_cap": _raise_not_supported}
        # An empty sensor list makes the per-sensor loop a no-op: nothing is set
        # and nothing is recorded, so the command finishes clean.
        return {
            "amdsmi_get_supported_power_cap": lambda *a, **k: {
                "sensor_inds": [],
                "sensor_types": {},
            }
        }

    def clean_local_data(cmd, mode):
        fn = _raise_not_supported if mode == "fail" else _noop
        return {"amdsmi_clean_gpu_local_data": fn}

    return {
        "gpureset": gpureset,
        "clocks": clocks,
        "fans": fans,
        "profile": profile,
        "xgmierr": xgmierr,
        "perf_determinism": perf_determinism,
        "power_cap": power_cap,
        "clean_local_data": clean_local_data,
    }


class TestResetGpuGAllFailureGuards(unittest.TestCase):
    """A per-device GPU `reset` failure must be recorded and must NOT crash."""

    def test_every_reset_option_is_driven(self):
        """Forcing function: every per-device GPU `reset` option must have a
        driver in _build_reset_specs. Adding a new option without a spec fails
        here, blocking a future `-g all` reset regression from shipping."""
        reset = _load_reset()
        options = _reset_gpu_options(reset)
        specced = set(_build_reset_specs(reset))
        self.assertEqual(
            options,
            specced,
            "GPU reset options and their -g all failure drivers are out of sync.\n"
            f"  unspecced (add a _build_reset_specs entry): {sorted(options - specced)}\n"
            f"  stale (spec exists but no longer an option): {sorted(specced - options)}",
        )

    def test_every_reset_option_records_on_failure_not_crashes(self):
        """Generic exerciser: drive each GPU `reset` option through reset with a
        per-device library failure injected, and assert the record-then-finalize
        contract holds -- the failure is recorded (exit != 0) and the handler
        returns instead of raising."""
        reset = _load_reset()
        specs = _build_reset_specs(reset)
        for name in sorted(specs):
            with self.subTest(option=name):
                cmd = _make_reset_cmd(reset)
                patches = specs[name](cmd, "fail")
                args = _reset_args(reset, **{name: True})
                with _patch_amdsmi_interface(reset, **patches):
                    cmd.reset(args)  # must NOT raise
                self.assertTrue(
                    cmd.helpers.error_collector.has_errors,
                    f"{name}: per-device failure was not recorded (would exit 0)",
                )
                self.assertNotEqual(
                    cmd.helpers.error_collector.resolve_exit_code(),
                    0,
                    f"{name}: exit code resolved to 0 despite a recorded failure",
                )

    def test_every_reset_option_succeeds_cleanly(self):
        """Mirror contract: a successful per-device `reset` records NOTHING and
        finalizes to exit 0. Guards against a success path that wrongly records
        an error, crashes, or resolves to a non-zero exit code."""
        reset = _load_reset()
        specs = _build_reset_specs(reset)
        for name in sorted(specs):
            with self.subTest(option=name):
                cmd = _make_reset_cmd(reset)
                patches = specs[name](cmd, "success")
                args = _reset_args(reset, **{name: True})
                with _patch_amdsmi_interface(reset, **patches):
                    cmd.reset(args)  # must NOT raise
                self.assertFalse(
                    cmd.helpers.error_collector.has_errors,
                    f"{name}: success path wrongly recorded an error",
                )
                self.assertEqual(
                    cmd.helpers.error_collector.resolve_exit_code(),
                    0,
                    f"{name}: success path did not resolve to exit 0",
                )
