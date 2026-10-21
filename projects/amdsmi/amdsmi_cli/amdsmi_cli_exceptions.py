#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import csv
import enum
import io
import json

try:
    from amdsmi_i18n import lazy as _L, tr as _tr
except ImportError:  # loaded by file path (e.g. unit tests): make the CLI dir importable
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    try:
        from amdsmi_i18n import lazy as _L, tr as _tr
    except ImportError:  # copied without amdsmi_i18n.py: English only

        def _L(template, **kwargs):
            return template.format(**kwargs) if kwargs else template

        def _tr(value):
            return value


# NOTE: intentionally no module-level ``from amdsmi import amdsmi_wrapper`` here.
# This module (and its AmdSmiExitCode enum) must stay importable even when the
# amdsmi package failed to load, so the import-error paths can exit with
# AmdSmiExitCode.IMPORT_ERROR instead of a hardcoded number. The wrapper is
# imported lazily where it is actually needed (see AmdSmiPermissionDeniedException).


@enum.unique
class AmdSmiExitCode(enum.IntEnum):
    """CLI-invented process exit codes, in the reserved 193-253 band.

    The single byte a process can return is split four ways::

        0         success
        1-128     library AMDSMI_STATUS_* values (currently 0-56)
        129-192   POSIX signal codes (128 + signal number, SIGRTMAX = 64).
                  Owned by the OS, which reports them for any process
        193-253   the CLI-invented codes below
        254/255   library sentinels: AMDSMI_STATUS_MAP_ERROR (0xFFFFFFFE) -> 254,
                  AMDSMI_STATUS_UNKNOWN_ERROR (0xFFFFFFFF) -> 255

    Library statuses are NOT redefined here: a library failure surfaces as its
    real AMDSMI_STATUS_* value via library_code_to_exit_code. 254/255 are
    *library* errors even though they sit above the CLI band (enforced by
    test_cli_codes_never_collide_with_library_exit_codes).
    Each member carries a human-readable ``.note``.

    Note on "not supported": a *library* AMDSMI_STATUS_NOT_SUPPORTED (2) is a
    runtime result from a device call and is honored as-is (exit code 2). The
    CLI's own COMMAND_NOT_SUPPORTED (200) is a parse-time decision (the command
    isn't available on this system, so no library call is made), so callers can
    tell the two apart from the exit code alone. DEVICE_INTERFACE_UNAVAILABLE
    (201) covers the remaining case: the command is valid and the device is
    present, but an interface the CLI reads directly yielded no usable data, so
    there is no library status to fold.

    Next free code is 208. Assigning one is free only until it ships: a released
    code keeps its meaning forever, because scripts branch on the number.
    """

    def __new__(cls, value, note=""):
        # Per-code doc string, queryable for a future --help/exit-code table (no runtime consumer today).
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.note = note
        return obj

    SUCCESS = (0, "all recorded operations succeeded")
    IMPORT_ERROR = (193, "Python import failure")
    INVALID_COMMAND = (194, "unrecognized command")
    INVALID_PARAMETER = (195, "invalid parameter")
    # No devices present = a live driver enumerating zero handles. A missing
    # driver never reaches here, it exits DRIVERS_NOT_LOADED.
    DEVICE_NOT_FOUND = (196, "requested device not found, or no devices present")
    INVALID_FILE_PATH = (197, "invalid file path")
    INVALID_PARAMETER_VALUE = (198, "invalid parameter value")
    MISSING_PARAMETER_VALUE = (199, "missing parameter value")
    COMMAND_NOT_SUPPORTED = (200, "command not available on this system (parse-time)")
    # TEMPORARY. Sole user is the gpu_od fan OD_RANGE read, which parses sysfs
    # directly because no API reports the range minimum. Retire this code once
    # amdsmi_get_gpu_fan_speed_range() lands and that branch can record the
    # library status instead (see the TODO in tests/python/common/common.py).
    DEVICE_INTERFACE_UNAVAILABLE = (201, "required device interface unavailable to the CLI")
    REQUIRED_COMMAND = (202, "required command/target missing")
    INVALID_SUBCOMMAND = (203, "invalid subcommand")
    MIXED_DEVICE_ERRORS = (204, "aggregated: >1 recorded failure with DIFFERING codes")
    INIT_TIMEOUT = (205, "amdsmi_init() watchdog fired (library call hung)")
    DRIVERS_NOT_LOADED = (206, "no usable AMD drivers / modules not loaded")
    USER_ABORTED = (207, "user declined an interactive confirmation prompt")
    # Sits at the top of the band as a sentinel, mirroring the library's 254/255.
    UNREPRESENTABLE_LIBRARY_STATUS = (253, "library status outside 0-128 and not a known sentinel")


# Reserved band for the CLI-invented codes above. Starts one past the highest
# POSIX signal code (128 + SIGRTMAX = 192) and stops below the library's 254/255
# sentinels, so a CLI code can collide with neither.
CLI_EXIT_CODE_BAND_START = 193
CLI_EXIT_CODE_BAND_END = 253

# The library's half of the byte. It may not grow past this into signal space.
LIBRARY_STATUS_MAX = 128


class AmdSmiErrorSeverity(enum.Enum):
    """Whether an error should stop the whole command or just this device.

    FATAL  - abort immediately (import / init / driver-not-loaded / bad command
             syntax / missing required target / permission). Continuing makes no
             sense because nothing device-specific succeeded.
    DEVICE - record the failure and keep iterating the remaining devices; the
             process exit code is decided once at the end by
             ``AmdSmiErrorCollector.resolve_exit_code()``.
    """

    FATAL = "fatal"
    DEVICE = "device"


class AmdSmiDeviceKind(enum.Enum):
    """Device kinds the CLI can report as missing. The value is the label used
    in the message; NIC covers both AI-NIC and Broadcom, which ``-N`` selects
    together.
    """

    GPU = "GPU"
    CPU = "CPU"
    CPU_CORE = "CPU CORE"
    NIC = "NIC"
    SWITCH = "SWITCH"


# The two 32-bit library sentinels (see amdsmi_wrapper.amdsmi_status_t), named here
# so this module stays importable without the amdsmi package and avoids bare hex.
AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE
AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF


AMDSMI_ERROR_MESSAGES = {
    0: _L("Success"),
    1: _L("Invalid parameters"),
    2: _L("Command not supported"),
    3: _L("Command not yet implemented"),
    4: _L("Failed load module"),
    5: _L("Failed load symbol"),
    6: _L("Drm error"),
    7: _L("API call failed"),
    8: _L("Timeout in API call"),
    9: _L("Retry operation"),
    10: _L("Permission Denied"),
    11: _L("Interrupt occurred during execution"),
    12: _L("I/O Error"),
    13: _L("Address fault"),
    14: _L("Error opening file"),
    15: _L("Not enough memory"),
    16: _L("Internal error"),
    17: _L("Out of bounds"),
    18: _L("Initialization error"),
    19: _L("Internal reference counter exceeded"),
    20: _L("Directory not found"),
    21: _L("IPC error"),
    # Reserved for future error messages
    30: _L("Device busy"),
    31: _L("Device Not found"),
    32: _L("Device not initialized"),
    33: _L("No more free slot"),
    34: _L("Driver not loaded"),
    # Reserved for future error messages
    39: _L("More data available than the provided buffer"),
    40: _L("No data was found for given input"),
    41: _L("Insufficient size for operation"),
    42: _L("Unexpected size of data was read"),
    43: _L("The data read or provided was unexpected"),
    44: _L("System has different cpu than AMD"),
    45: _L("Energy driver not found"),
    46: _L("MSR driver not found"),
    47: _L("HSMP driver not found"),
    48: _L("HSMP not supported"),
    49: _L("HSMP message/feature not supported"),
    50: _L("HSMP message timed out"),
    51: _L("No Energy and HSMP driver present"),
    52: _L("File or directory not found"),
    53: _L("Parsed argument is invalid"),
    54: _L("AMDGPU restart error"),
    55: _L("Setting is not available"),
    56: _L("EEPROM is corrupted"),
    AMDSMI_STATUS_MAP_ERROR: _L("AMD-SMI Library error did not map to a status code"),
    AMDSMI_STATUS_UNKNOWN_ERROR: _L("Unknown error"),
}


def _get_error_message(error_code):
    # test_every_library_status_has_a_friendly_message guarantees a mapping for
    # every known status, so this fallback only fires for an unrecognized value
    # (e.g. a mismatched library version). Include the raw code so it stays
    # distinct from AMDSMI_STATUS_UNKNOWN_ERROR's own "Unknown error".
    code = abs(error_code)
    if code in AMDSMI_ERROR_MESSAGES:
        return AMDSMI_ERROR_MESSAGES[code]
    return _L("Unrecognized error code {code}", code=code)


def _get_error_name(error_code):
    """Return the AMDSMI_STATUS_* name for a status code.

    e.g. 43 -> 'AMDSMI_STATUS_UNEXPECTED_DATA'. Imports the wrapper lazily so
    this module stays importable without the amdsmi package, and falls back to
    a synthetic 'AMDSMI_STATUS_<code>' if the code is not recognized (e.g. a
    mismatched library version).
    """
    code = abs(int(error_code))
    try:
        from amdsmi import amdsmi_wrapper

        name = amdsmi_wrapper.amdsmi_status_t__enumvalues.get(code)
        if name:
            return name
    except Exception:
        pass
    return f"AMDSMI_STATUS_{code}"


def _get_error_type_name(value):
    """Return the name for an exit code, whichever space it's from.

    Tries the CLI-invented ``AmdSmiExitCode`` band first (e.g. 198 ->
    'INVALID_PARAMETER_VALUE'), since those values fall outside the library's
    status range and ``_get_error_name`` would otherwise synthesize a bogus
    'AMDSMI_STATUS_<code>' for them. Falls back to ``_get_error_name`` for a
    real library AMDSMI_STATUS_* value.
    """
    try:
        return AmdSmiExitCode(value).name
    except ValueError:
        return _get_error_name(value)


def _csv_error(message, code, error_type):
    # Messages carry commas and newlines (option lists, multi-line hints).
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(
        [["error", "code", "error_type"], [message, code, error_type]]
    )
    return buf.getvalue().rstrip("\n")


class AmdSmiException(Exception):
    # Default: an unclassified error stops everything. Per-device failures
    # (see AmdSmiLibraryErrorException) override this to DEVICE.
    severity = AmdSmiErrorSeverity.FATAL

    def __init__(self):
        self.json_message = {}
        self.csv_message = ""
        self.stdout_message = ""
        self.message = ""
        self.output_format = ""
        self.device_type = ""
        self.value = 0

    def _build_output_messages(self, common_message, error_type_name=None):
        """Render *common_message* into each output format __str__ selects from.

        *error_type_name* lets a subclass supply an already-resolved name (see
        ``AmdSmiLibraryErrorException``, whose ``status_name`` is derived from
        the raw library code, not the already-mapped ``self.value``).
        """
        # json/csv are read a line at a time, so they carry the message without
        # the layout whitespace the human-readable hints use.
        flat = " ".join(common_message.split())
        if error_type_name is None:
            error_type_name = _get_error_type_name(self.value)
        self.json_message["error"] = flat
        self.json_message["code"] = self.value
        self.json_message["error_type"] = error_type_name
        self.csv_message = _csv_error(flat, self.value, error_type_name)
        self.stdout_message = _L(
            "{common_message} Error code: {value} [{error_type_name}]",
            common_message=common_message,
            value=self.value,
            error_type_name=error_type_name,
        )

    def __str__(self):
        # Return message according to the current output format
        if self.output_format == "json":
            self.message = json.dumps(self.json_message)
        elif self.output_format == "csv":
            self.message = self.csv_message
        else:
            # Human-readable only: localized text (identical English str when no catalog)
            self.message = _tr(self.stdout_message)

        return self.message


class AmdSmiInvalidCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_COMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "Command '{command}' is invalid. Run 'amd-smi -h' for more info.", command=self.command
        )

        if message:
            common_message = message

        self._build_output_messages(common_message)


class AmdSmiInvalidParameterException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str, message=None, hint=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_PARAMETER)
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = _L(
            "Parameter '{arg}' is invalid. Run 'amd-smi {command} -h' for more info.",
            arg=self.arg,
            command=self.command,
        )
        if message:
            common_message = message
        if hint:
            common_message += f" {hint}"

        self._build_output_messages(common_message)


class AmdSmiDeviceNotFoundException(AmdSmiException):
    def __init__(self, command, outputformat: str, device_kind: AmdSmiDeviceKind):
        super().__init__()
        self.value = int(AmdSmiExitCode.DEVICE_NOT_FOUND)
        self.command = command
        self.output_format = outputformat
        self.device_type = device_kind.value

        common_message = _L(
            "Can not find a device: {device_type} '{command}'",
            device_type=self.device_type,
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiInvalidFilePathException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_FILE_PATH)
        self.command = command
        self.output_format = outputformat

        common_message = _L("Path '{command}' cannot be found.", command=self.command)

        if message:
            common_message = message

        self._build_output_messages(common_message)


class AmdSmiInvalidParameterValueException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str, hint: str = None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_PARAMETER_VALUE)
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = _L(
            "Value '{arg}' is not of valid type or format. Run 'amd-smi {command} -h' for more info.",
            arg=self.arg,
            command=self.command,
        )
        if hint:
            common_message += f" {hint}"

        self._build_output_messages(common_message)


class AmdSmiMissingParameterValueException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.MISSING_PARAMETER_VALUE)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "Parameter '{command}' requires a value. Run '--help' for more info.",
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiCommandNotSupportedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        # CLI-level (parse-time) decision: the command name is not available on
        # this system, so no library call was made. Distinct from a library
        # AMDSMI_STATUS_NOT_SUPPORTED (2), which is honored as-is elsewhere.
        self.value = int(AmdSmiExitCode.COMMAND_NOT_SUPPORTED)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "Command '{command}' is not supported on the system. Run '--help' for more info.",
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiRequiredCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.REQUIRED_COMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "Command '{command}' requires a target argument. Run 'amd-smi {command} -h' for more info.",
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiInvalidSubcommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_SUBCOMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "AMD-SMI Command '{command}' is invalid. Must receive valid AMD-SMI Command first. Run 'amd-smi -h' for more info.",
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiPermissionDeniedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        # Only raised after a live library call returned NO_PERM, so amdsmi is
        # loaded here. Import lazily so this module stays importable without it.
        from amdsmi import amdsmi_wrapper

        self.value = int(amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)
        self.command = command
        self.output_format = outputformat

        common_message = _L(
            "AMD-SMI Command '{command}' requires elevation (sudo privileges required)",
            command=self.command,
        )

        self._build_output_messages(common_message)


class AmdSmiLibraryErrorException(AmdSmiException):
    # A library error is (almost always) specific to one device, so it is
    # recorded and the command keeps going to the next device.
    severity = AmdSmiErrorSeverity.DEVICE

    def __init__(self, outputformat: str, error_code, detail=None):
        super().__init__()
        # Surface the underlying AMDSMI_STATUS_* value as the exit code.
        self.value = library_code_to_exit_code(error_code)
        self.amdsmi_lib_code = error_code
        self.output_format = outputformat

        # Format: "[AMDSMI_STATUS_<name>] <message>". ``detail`` overrides the
        # generic library message with caller-specific context when provided.
        # status_name / status_message are exposed so callers can render the
        # pieces separately (e.g. structured JSON/CSV rows).
        self.status_name = _get_error_name(self.amdsmi_lib_code)
        self.status_message = detail if detail else _get_error_message(self.amdsmi_lib_code)
        # LazyStr.__radd__ keeps the localized companion of status_message (English str unchanged)
        common_message = f"[{self.status_name}] " + self.status_message

        self._build_output_messages(common_message, error_type_name=self.status_name)


def library_code_to_exit_code(error_code):
    """Map an AMDSMI_STATUS_* value to a POSIX process exit code (0-255).

    The library owns 0-128 plus its two 32-bit sentinels, which report as their
    low byte (``AMDSMI_STATUS_MAP_ERROR`` = 0xFFFFFFFE -> 254,
    ``AMDSMI_STATUS_UNKNOWN_ERROR`` = 0xFFFFFFFF -> 255). Statuses currently run
    0-56, so real values pass through and the exit code *is* the status.

    Anything else cannot be reported faithfully. 129-192 is signal territory,
    and a value above 255 loses its identity in a byte -- status 300 would
    arrive as 44, a real but unrelated status, and status 256 as 0, reporting
    success. Those report UNREPRESENTABLE_LIBRARY_STATUS instead of a plausible
    wrong answer.
    """
    status = abs(int(error_code))
    if status in (AMDSMI_STATUS_MAP_ERROR, AMDSMI_STATUS_UNKNOWN_ERROR):
        return status & 0xFF
    if status > LIBRARY_STATUS_MAX:
        return int(AmdSmiExitCode.UNREPRESENTABLE_LIBRARY_STATUS)
    return status


class AmdSmiErrorCollector:
    """Collects per-device failures across a command and decides one exit code.

    This is the heart of the "record-then-finalize" model: instead of a device
    error raising and unwinding past the device loop (which aborts the remaining
    devices and drops their buffered output), each device's failure is recorded
    here and the loop keeps going. ``resolve_exit_code()`` is consulted once, at
    the very end, to pick the single process exit code.
    """

    def __init__(self):
        self._codes = []

    def reset(self):
        """Clear recorded codes (e.g. between watch-mode iterations).

        Note: this is a future-proofing measure. Set/reset
        are not watch enabled commands, but we may want to add for
        future watch mode commands. Today, this is not a requirement.
        """
        self._codes = []

    def record(self, exit_code):
        """Record one device failure by its exit code."""
        self._codes.append(int(abs(exit_code)))

    def record_library_error(self, error_code):
        """Record a device failure from a raw AMDSMI_STATUS_* value."""
        self._codes.append(library_code_to_exit_code(error_code))

    @property
    def has_errors(self):
        return bool(self._codes)

    def resolve_exit_code(self):
        """Pick the process exit code from everything recorded.

        none recorded            -> SUCCESS (0)
        all recorded are equal   -> that code
        mixed different codes    -> MIXED_DEVICE_ERRORS

        "Mixed" is any set of recorded failures with differing codes in a single
        run: across multiple devices, OR across multiple sub-steps/fields of one
        command (e.g. reset --clocks). So a single GPU can resolve to
        MIXED_DEVICE_ERRORS if its sub-steps fail with different codes -- the
        name keeps DEVICE for registry compatibility, but the semantics are
        per-run, not strictly per-device.
        """
        if not self._codes:
            return int(AmdSmiExitCode.SUCCESS)
        if len(set(self._codes)) == 1:
            return self._codes[0]
        return int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)
