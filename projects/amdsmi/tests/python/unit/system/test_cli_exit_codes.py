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

from common.common import amdsmi, amdsmi_path

# The amd-smi CLI ships beside the amdsmi package: ``amdsmi_path`` resolves to
# ``<rocm>/share/amd_smi`` and the CLI installs to the sibling
# ``<rocm>/libexec/amdsmi_cli``. Add it to sys.path so the CLI modules under
# test import without the compiled driver package.
_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
_CLI_DIR = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli")
if _CLI_DIR not in sys.path:
    sys.path.append(_CLI_DIR)

import amdsmi_cli_exceptions as cli_exc  # noqa: E402
from amdsmi import amdsmi_wrapper  # noqa: E402


class TestAmdSmiCliExitCodes(unittest.TestCase):
    """Pure-logic tests for the amd-smi CLI exit-code model.

    Exercises the record-then-finalize error model in amdsmi_cli_exceptions
    without needing a GPU/CPU/Core or elevated permissions:
    the AmdSmiErrorCollector, the library-status -> POSIX-byte fold, exit codes surfaced by
    AmdSmiLibraryErrorException, the FATAL vs DEVICE severity split, and the
    reserved 192-255 CLI-only code band.
    """

    ExitCode = cli_exc.AmdSmiExitCode
    Severity = cli_exc.AmdSmiErrorSeverity

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
