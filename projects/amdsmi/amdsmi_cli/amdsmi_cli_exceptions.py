#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import enum
import json

# NOTE: intentionally no module-level ``from amdsmi import amdsmi_wrapper`` here.
# This module (and its AmdSmiExitCode enum) must stay importable even when the
# amdsmi package failed to load, so the import-error paths can exit with
# AmdSmiExitCode.IMPORT_ERROR instead of a hardcoded number. The wrapper is
# imported lazily where it is actually needed (see AmdSmiPermissionDeniedException).


class AmdSmiExitCode(enum.IntEnum):
    """CLI-invented process exit codes, in the reserved 192-255 band.

    Library statuses are NOT redefined here: a library failure surfaces as its
    real AMDSMI_STATUS_* value (currently 0-56) folded to a byte by
    library_code_to_exit_code. The two 32-bit library sentinels also fold into
    the byte range -- AMDSMI_STATUS_MAP_ERROR (0xFFFFFFFE) -> 254 and
    AMDSMI_STATUS_UNKNOWN_ERROR (0xFFFFFFFF) -> 255. Those two values are
    *library* errors, NOT CLI codes, even though 254/255 fall inside the 192-255
    band; the CLI codes below live in 192-208 and deliberately avoid 254/255
    (enforced by test_cli_codes_never_collide_with_library_exit_codes).
    Each member carries a human-readable ``.note``.

    Note on "not supported": a *library* AMDSMI_STATUS_NOT_SUPPORTED (2) is a
    runtime result from a device call and is honored as-is (exit code 2). The
    CLI's own COMMAND_NOT_SUPPORTED (199) is a parse-time decision (the command
    isn't available on this system, so no library call is made), so callers can
    tell the two apart from the exit code alone.

    Reserved, not emitted by this CLI (via another PR; consolidating with original PR author.
    Free to re-assign if needed and remove this comment (and the below items).):
      200 - retired "parameter not supported" slot (AmdSmiParameterNotSupportedException, removed)
      203 - retired; permission-denied surfaces the library NO_PERM (10) directly
      204 - reserved for a CLI platform/build mismatch
    """

    def __new__(cls, value, note=""):
        # Per-code doc string, queryable for a future --help/exit-code table (no runtime consumer today).
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.note = note
        return obj

    SUCCESS = (0, "all recorded operations succeeded")
    IMPORT_ERROR = (192, "Python import failure")
    INVALID_COMMAND = (193, "unrecognized command")
    INVALID_PARAMETER = (194, "invalid parameter")
    DEVICE_NOT_FOUND = (195, "target device not found")
    INVALID_FILE_PATH = (196, "invalid file path")
    INVALID_PARAMETER_VALUE = (197, "invalid parameter value")
    MISSING_PARAMETER_VALUE = (198, "missing parameter value")
    COMMAND_NOT_SUPPORTED = (199, "command not available on this system (parse-time)")
    REQUIRED_COMMAND = (201, "required command/target missing")
    INVALID_SUBCOMMAND = (202, "invalid subcommand")
    MIXED_DEVICE_ERRORS = (205, "aggregated: >1 recorded failure with DIFFERING codes")
    INIT_TIMEOUT = (206, "amdsmi_init() watchdog fired (library call hung)")
    DRIVERS_NOT_LOADED = (207, "no usable AMD drivers / modules not loaded")
    USER_ABORTED = (208, "user declined an interactive confirmation prompt")


# Reserved band for the CLI-invented codes above. Sits above the library status
# range (statuses fold to ~0-56, sentinels to 254/255) so the two never collide;
# 0xFF is the POSIX single-byte exit-code ceiling.
CLI_EXIT_CODE_BAND_START = 192
CLI_EXIT_CODE_BAND_END = 0xFF


class AmdSmiErrorSeverity(enum.Enum):
    """Whether an error should stop the whole command or just this device.

    FATAL  - abort immediately (import / init / driver-not-loaded / bad command
             syntax / missing required target / permission). Continuing makes no
             sense because nothing device-specific succeeded.
    DEVICE - record the failure and keep iterating the remaining devices; the
             process exit code is decided once at the end by ``finalize()``.
    """

    FATAL = "fatal"
    DEVICE = "device"


# The two 32-bit library sentinels (see amdsmi_wrapper.amdsmi_status_t), named here
# so this module stays importable without the amdsmi package and avoids bare hex.
AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE
AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF


AMDSMI_ERROR_MESSAGES = {
    0: "Success",
    1: "Invalid parameters",
    2: "Command not supported",
    3: "Command not yet implemented",
    4: "Failed load module",
    5: "Failed load symbol",
    6: "Drm error",
    7: "API call failed",
    8: "Timeout in API call",
    9: "Retry operation",
    10: "Permission Denied",
    11: "Interrupt occurred during execution",
    12: "I/O Error",
    13: "Address fault",
    14: "Error opening file",
    15: "Not enough memory",
    16: "Internal error",
    17: "Out of bounds",
    18: "Initialization error",
    19: "Internal reference counter exceeded",
    20: "Directory not found",
    21: "IPC error",
    # Reserved for future error messages
    30: "Device busy",
    31: "Device Not found",
    32: "Device not initialized",
    33: "No more free slot",
    34: "Driver not loaded",
    # Reserved for future error messages
    39: "More data available than the provided buffer",
    40: "No data was found for given input",
    41: "Insufficient size for operation",
    42: "Unexpected size of data was read",
    43: "The data read or provided was unexpected",
    44: "System has different cpu than AMD",
    45: "Energy driver not found",
    46: "MSR driver not found",
    47: "HSMP driver not found",
    48: "HSMP not supported",
    49: "HSMP message/feature not supported",
    50: "HSMP message timed out",
    51: "No Energy and HSMP driver present",
    52: "File or directory not found",
    53: "Parsed argument is invalid",
    54: "AMDGPU restart error",
    55: "Setting is not available",
    56: "EEPROM is corrupted",
    AMDSMI_STATUS_MAP_ERROR: "AMD-SMI Library error did not map to a status code",
    AMDSMI_STATUS_UNKNOWN_ERROR: "Unknown error",
}


def _get_error_message(error_code):
    # test_every_library_status_has_a_friendly_message guarantees a mapping for
    # every known status, so this fallback only fires for an unrecognized value
    # (e.g. a mismatched library version). Include the raw code so it stays
    # distinct from AMDSMI_STATUS_UNKNOWN_ERROR's own "Unknown error".
    code = abs(error_code)
    if code in AMDSMI_ERROR_MESSAGES:
        return AMDSMI_ERROR_MESSAGES[code]
    return f"Unrecognized error code {code}"


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

    def __str__(self):
        # Return message according to the current output format
        if self.output_format == "json":
            self.message = json.dumps(self.json_message)
        elif self.output_format == "csv":
            self.message = self.csv_message
        else:
            self.message = self.stdout_message

        return self.message


class AmdSmiInvalidCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_COMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = f"Command '{self.command}' is invalid. Run 'amd-smi -h' for more info."

        if message:
            common_message = message

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidParameterException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str, message=None, hint=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_PARAMETER)
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = (
            f"Parameter '{self.arg}' is invalid. Run 'amd-smi {self.command} -h' for more info."
        )
        if message:
            common_message = message
        if hint:
            common_message += f" {hint}"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiDeviceNotFoundException(AmdSmiException):
    def __init__(self, command, outputformat: str, gpu: bool, cpu: bool, core: bool):
        super().__init__()
        self.value = int(AmdSmiExitCode.DEVICE_NOT_FOUND)
        self.command = command
        self.output_format = outputformat

        # Handle different devices
        self.device_type = ""
        if gpu:
            self.device_type = "GPU"
        elif cpu:
            self.device_type = "CPU"
        elif core:
            self.device_type = "CPU CORE"

        common_message = f"Can not find a device: {self.device_type} '{self.command}'"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidFilePathException(AmdSmiException):
    def __init__(self, command, outputformat: str, message=None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_FILE_PATH)
        self.command = command
        self.output_format = outputformat

        common_message = f"Path '{self.command}' cannot be found."

        if message:
            common_message = message

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidParameterValueException(AmdSmiException):
    def __init__(self, command, arg, outputformat: str, hint: str = None):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_PARAMETER_VALUE)
        self.command = command
        self.arg = arg
        self.output_format = outputformat

        common_message = f"Value '{self.arg}' is not of valid type or format. Run 'amd-smi {self.command} -h' for more info."
        if hint:
            common_message += f" {hint}"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiMissingParameterValueException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.MISSING_PARAMETER_VALUE)
        self.command = command
        self.output_format = outputformat

        common_message = f"Parameter '{self.command}' requires a value. Run '--help' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiCommandNotSupportedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        # CLI-level (parse-time) decision: the command name is not available on
        # this system, so no library call was made. Distinct from a library
        # AMDSMI_STATUS_NOT_SUPPORTED (2), which is honored as-is elsewhere.
        self.value = int(AmdSmiExitCode.COMMAND_NOT_SUPPORTED)
        self.command = command
        self.output_format = outputformat

        common_message = (
            f"Command '{self.command}' is not supported on the system. Run '--help' for more info."
        )

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiRequiredCommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.REQUIRED_COMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = f"Command '{self.command}' requires a target argument. Run 'amd-smi {self.command} -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiInvalidSubcommandException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        self.value = int(AmdSmiExitCode.INVALID_SUBCOMMAND)
        self.command = command
        self.output_format = outputformat

        common_message = f"AMD-SMI Command '{self.command}' is invalid. Must receive valid AMD-SMI Command first. Run 'amd-smi -h' for more info."

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


class AmdSmiPermissionDeniedException(AmdSmiException):
    def __init__(self, command, outputformat: str):
        super().__init__()
        # Only raised after a live library call returned NO_PERM, so amdsmi is
        # loaded here. Import lazily so this module stays importable without it.
        from amdsmi import amdsmi_wrapper

        self.value = int(amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)
        self.command = command
        self.output_format = outputformat

        common_message = (
            f"AMD-SMI Command '{self.command}' requires elevation (sudo privileges required)"
        )

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


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
        common_message = f"[{self.status_name}] {self.status_message}"

        self.json_message["error"] = common_message
        self.json_message["code"] = self.value
        self.csv_message = f"error,code\n{common_message}, {self.value}"
        self.stdout_message = f"{common_message} Error code: {self.value}"


def library_code_to_exit_code(error_code):
    """Fold an AMDSMI_STATUS_* value into a POSIX process exit code (0-255).

    Real status values (currently 0-56) fit in a byte and pass through unchanged,
    so the exit code *is* the underlying status. The two 32-bit sentinels
    (``AMDSMI_STATUS_MAP_ERROR`` = 0xFFFFFFFE, ``AMDSMI_STATUS_UNKNOWN_ERROR`` =
    0xFFFFFFFF) fold to their low byte (254 / 255). No status numbers are
    hardcoded here.
    """
    status = abs(int(error_code))
    folded = status & 0xFF
    # A nonzero status that is an exact multiple of 256 folds to 0, reporting success.
    if status and not folded:
        return CLI_EXIT_CODE_BAND_END
    return folded


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
        mixed different codes    -> MIXED_DEVICE_ERRORS (205)

        "Mixed" is any set of recorded failures with differing codes in a single
        run: across multiple devices, OR across multiple sub-steps/fields of one
        command (e.g. reset --clocks). So a single GPU can resolve to 205 if its
        sub-steps fail with different codes -- the name keeps DEVICE for registry
        compatibility, but the semantics are per-run, not strictly per-device.
        """
        if not self._codes:
            return int(AmdSmiExitCode.SUCCESS)
        if len(set(self._codes)) == 1:
            return self._codes[0]
        return int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)
