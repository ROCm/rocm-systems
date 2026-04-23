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
from perfxpert.tools._safety import build_safe_env, reject_shell_metachars
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
    "--pc-sampling",
    "--pmc",
    "-d",
    "--output-dir",
    "-o",
    "--output",
    "--pid",
}

_ROCPROFV3_PATH_VALUE_FLAGS: Set[str] = {
    "--att-library-path",
    "-d",
    "--output-dir",
    "-o",
    "--output",
}

_ROCPROFV3_FLAGS: Set[str] = _ROCPROFV3_NO_VALUE_FLAGS | _ROCPROFV3_VALUE_FLAGS | {"--"}
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class RocprofFlagError(Exception):
    """Raised when argv contains a rocprofv3 flag not in the allowlist."""


def _validate_path_value(flag: str, value: str) -> None:
    """Reject rocprofv3 path values that cannot be project-confined later."""
    normalized = value.replace("\\", "/")
    if not value or value.startswith(("~", "/", "\\")) or _WINDOWS_DRIVE_RE.match(value):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")
    if ".." in normalized.split("/"):
        raise RocprofFlagError(f"unsafe path value for {flag}: {value!r}")


def _validate_argv(argv: List[str]) -> None:
    """Sanitize + allowlist-check every token."""
    if not argv:
        raise ValueError("argv must not be empty")
    if argv[0] != "rocprofv3":
        raise ValueError(f"argv[0] must be 'rocprofv3', got {argv[0]!r}")

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
                if key in _ROCPROFV3_PATH_VALUE_FLAGS:
                    _validate_path_value(key, value)
        else:
            raise RocprofFlagError(f"unexpected rocprofv3 value without flag: {tok!r}")
        i += 1

    # Sanitize every token (including target app path + args)
    for tok in argv:
        reject_shell_metachars(tok)

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
    _validate_argv(argv)
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
