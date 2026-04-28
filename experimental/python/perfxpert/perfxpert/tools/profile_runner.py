"""profile_runner — profile.run EXECUTION tool.

Wraps rocprofv3 in a sanitized subprocess:
- argv is validated against rocprofv3 flag allowlist (§5.8)
- env is filtered through build_safe_env() (API keys NEVER forwarded)
- shell=False, argv is list (no shell injection possible)

Extends the existing `_filter_rec_commands` pattern from analyze.py.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Set

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    build_safe_env,
    confine_to_project_root,
    reject_shell_metachars,
)
from perfxpert.tools._tooldep import require_tool


_ROCPROFV3_TIMEOUT_SEC = 600

# Authoritative rocprofv3 flag allowlist. Extend via knowledge yaml in a
# future task — hard-coded here so a hostile YAML can't widen it.
_ROCPROFV3_NO_VALUE_FLAGS: Set[str] = {
    "--sys-trace",
    "--hip-trace",
    "--kernel-trace",
    "--memory-copy-trace",
    "--hsa-trace",
    "--stats",
    "--att",
    "--pc-sampling",
    "--process-sync",
    "--list-avail",
    "--list-counters",
    "--help",
}

_ROCPROFV3_VALUE_FLAGS: Set[str] = {
    "--att-library-path",
    "--att-target-cu",
    "--att-simd-select",
    "--att-buffer-size",
    "--att-activity",
    "--pc-sampling-method",
    "--pc-sampling-unit",
    "--pc-sampling-interval",
    "-d",
    "--output-dir",
    "-o",
    "--output",
    "--pid",
}

_ROCPROFV3_MULTI_VALUE_FLAGS: Set[str] = {
    "--pmc",
}

_ROCPROFV3_OPTIONAL_BOOL_VALUE_FLAGS: Set[str] = {
    "--pc-sampling-beta-enabled",
}

_ROCPROFV3_OUTPUT_PATH_VALUE_FLAGS: Set[str] = {
    "-d",
    "--output-dir",
    "-o",
    "--output",
}

_ROCPROFV3_LIBRARY_PATH_VALUE_FLAGS: Set[str] = {
    "--att-library-path",
}

_ROCPROFV3_FLAGS: Set[str] = (
    _ROCPROFV3_NO_VALUE_FLAGS
    | _ROCPROFV3_VALUE_FLAGS
    | _ROCPROFV3_MULTI_VALUE_FLAGS
    | _ROCPROFV3_OPTIONAL_BOOL_VALUE_FLAGS
    | {"--"}
)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_BOOL_VALUES = {"1", "0", "true", "false", "on", "off", "yes", "no"}
_PC_SAMPLING_METHODS = {"stochastic", "host_trap"}
_PC_SAMPLING_UNITS = {"instructions", "cycles", "time"}


class RocprofFlagError(Exception):
    """Raised when argv contains a rocprofv3 flag not in the allowlist."""


def _validate_output_path_value(flag: str, value: str, cwd: Path | None) -> None:
    """Reject rocprofv3 output paths that are not confined to cwd."""
    normalized = value.replace("\\", "/")
    if (
        not value
        or value.startswith(("~", "/", "\\"))
        or _WINDOWS_DRIVE_RE.match(value)
    ):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")
    if cwd is None and ".." in normalized.split("/"):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")
    if cwd is not None:
        try:
            confine_to_project_root(Path(cwd), value)
        except Exception as e:
            raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}") from e


def _validate_library_path_value(flag: str, value: str) -> None:
    """Reject malformed rocprofv3 library search paths without project confinement."""
    normalized = value.replace("\\", "/")
    if not value or value.startswith(("~", "\\")) or _WINDOWS_DRIVE_RE.match(value):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")
    if ".." in normalized.split("/"):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")


def _validate_value(flag: str, value: str, cwd: Path | None) -> None:
    if not value:
        raise RocprofFlagError(f"rocprofv3 flag requires a value: {flag!r}")
    if flag in _ROCPROFV3_OUTPUT_PATH_VALUE_FLAGS:
        _validate_output_path_value(flag, value, cwd)
    elif flag in _ROCPROFV3_LIBRARY_PATH_VALUE_FLAGS:
        _validate_library_path_value(flag, value)
    elif flag == "--pc-sampling-method" and value.lower() not in _PC_SAMPLING_METHODS:
        raise RocprofFlagError(f"invalid value for {flag}: {value!r}")
    elif flag == "--pc-sampling-unit" and value.lower() not in _PC_SAMPLING_UNITS:
        raise RocprofFlagError(f"invalid value for {flag}: {value!r}")
    elif flag == "--pc-sampling-interval":
        try:
            if int(value) <= 0:
                raise ValueError
        except ValueError as e:
            raise RocprofFlagError(f"invalid value for {flag}: {value!r}") from e


def _validate_bool_value(flag: str, value: str) -> None:
    if value.lower() not in _BOOL_VALUES:
        raise RocprofFlagError(f"invalid boolean value for {flag}: {value!r}")


def _validate_argv(argv: List[str], *, cwd: Path | None = None) -> None:
    """Sanitize + allowlist-check every token."""
    if not argv:
        raise ValueError("argv must not be empty")
    if argv[0] != "rocprofv3":
        raise ValueError(f"argv[0] must be 'rocprofv3', got {argv[0]!r}")
    for tok in argv:
        reject_shell_metachars(tok)

    # Split at `--`; before it are rocprofv3 flags, after it is the target app.
    try:
        sep = argv.index("--")
    except ValueError:
        sep = len(argv)

    rocprof_tokens = argv[1:sep]
    target_tokens = argv[sep + 1:] if sep < len(argv) else []

    # Flag allowlist for rocprofv3 flags
    i = 0
    while i < len(rocprof_tokens):
        tok = rocprof_tokens[i]
        if tok.startswith("--") or tok.startswith("-"):
            key = tok.split("=", 1)[0]
            if key not in _ROCPROFV3_FLAGS:
                raise RocprofFlagError(
                    f"rocprofv3 flag not in allowlist: {tok!r}"
                )
            has_inline_value = "=" in tok
            if key in _ROCPROFV3_NO_VALUE_FLAGS:
                if has_inline_value:
                    raise RocprofFlagError(f"rocprofv3 flag does not take a value: {tok!r}")
            elif key in _ROCPROFV3_MULTI_VALUE_FLAGS:
                if has_inline_value:
                    values = [tok.split("=", 1)[1]]
                else:
                    values = []
                    while (
                        i + 1 < len(rocprof_tokens)
                        and not rocprof_tokens[i + 1].startswith("-")
                    ):
                        i += 1
                        values.append(rocprof_tokens[i])
                if not values:
                    raise RocprofFlagError(f"rocprofv3 flag requires a value: {tok!r}")
                for value in values:
                    _validate_value(key, value, cwd)
            elif key in _ROCPROFV3_OPTIONAL_BOOL_VALUE_FLAGS:
                if has_inline_value:
                    _validate_bool_value(key, tok.split("=", 1)[1])
                elif (
                    i + 1 < len(rocprof_tokens)
                    and not rocprof_tokens[i + 1].startswith("-")
                ):
                    i += 1
                    _validate_bool_value(key, rocprof_tokens[i])
            else:
                if has_inline_value:
                    value = tok.split("=", 1)[1]
                else:
                    if i + 1 >= len(rocprof_tokens):
                        raise RocprofFlagError(f"rocprofv3 flag requires a value: {tok!r}")
                    value = rocprof_tokens[i + 1]
                    if value.startswith("-"):
                        raise RocprofFlagError(f"rocprofv3 flag requires a value: {tok!r}")
                    i += 1
                _validate_value(key, value, cwd)
        else:
            raise RocprofFlagError(f"unexpected rocprofv3 value without flag: {tok!r}")
        i += 1

    # target tokens consumed for sanitization only; no allowlist on them


@tool_class(ToolClass.EXECUTION)
def run(
    argv: List[str],
    *,
    cwd: Path,
    timeout: int = _ROCPROFV3_TIMEOUT_SEC,
    extra_env: dict | None = None,
) -> Dict[str, Any]:
    """Invoke rocprofv3 with sanitized inputs.

    Returns:
        {"returncode": int, "stdout": str, "stderr": str}

    Raises:
        RocprofFlagError on unknown flag.
        ShellMetacharError on metachar-bearing token.
        subprocess.TimeoutExpired on timeout.
    """
    _validate_argv(argv, cwd=cwd)
    require_tool("rocprofv3")

    proc = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        cwd=str(cwd),
        env=build_safe_env(extra=extra_env),
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
    }
