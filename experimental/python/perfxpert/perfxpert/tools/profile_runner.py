"""profile_runner — profile.run EXECUTION tool.

Wraps rocprofv3 in a sanitized subprocess:
- argv is validated against rocprofv3 flag allowlist (§5.8)
- env is filtered through build_safe_env() (API keys NEVER forwarded)
- shell=False, argv is list (no shell injection possible)

Extends the existing `_filter_rec_commands` pattern from analyze.py.
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    build_safe_env,
    confine_to_project_root,
    reject_shell_metachars,
)


_ROCPROFV3_TIMEOUT_SEC = 600
_TRUSTED_ATT_LIBRARY_ROOT = Path("/opt/rocm/lib")

# Authoritative rocprofv3 flag allowlist. Extend via knowledge yaml in a
# future task — hard-coded here so a hostile YAML can't widen it.
_ROCPROFV3_FLAGS: Set[str] = {
    # trace modes
    "--sys-trace", "--hip-trace", "--kernel-trace", "--memory-copy-trace",
    "--hsa-trace", "--stats",
    # ATT
    "--att", "--att-library-path", "--att-target-cu", "--att-simd-select",
    "--att-buffer-size", "--att-activity",
    # PC sampling
    "--pc-sampling",
    # counters
    "--pmc",
    # output
    "-d", "--output-dir", "-o", "--output",
    # process controls
    "--process-sync", "--pid",
    # listing / info
    "--list-avail", "--list-counters", "--help",
    # discovery separator
    "--",
}


class RocprofFlagError(Exception):
    """Raised when argv contains a rocprofv3 flag not in the allowlist."""


_COUNTER_VALUE_RE = re.compile(r"^[A-Za-z0-9_:+,\-]+$")

_VALUE_TAKING_FLAGS: dict[str, Callable[[str, Path], None]] = {}


def _validate_path_flag_value(value: str, cwd: Path) -> None:
    if not value or value.startswith("-"):
        raise RocprofFlagError(f"path flag value must not be another flag: {value!r}")
    confine_to_project_root(cwd, value)


def _validate_external_path_value(value: str, cwd: Path) -> None:
    if not value or value.startswith("-"):
        raise RocprofFlagError(f"path flag value must not be another flag: {value!r}")
    if value.startswith("~"):
        raise RocprofFlagError(
            f"path flag value must be explicit and must not use '~': {value!r}"
        )
    if not Path(value).is_absolute():
        raise RocprofFlagError(
            f"--att-library-path must be an absolute path under /opt/rocm/lib: {value!r}"
        )

    trusted_root = _TRUSTED_ATT_LIBRARY_ROOT.resolve(strict=False)
    candidate = Path(value).resolve(strict=False)
    try:
        candidate.relative_to(trusted_root)
    except ValueError as e:
        raise RocprofFlagError(
            f"--att-library-path must stay within {trusted_root}: {value!r}"
        ) from e


def _validate_non_negative_int(value: str, cwd: Path) -> None:
    del cwd
    if not value.isdigit():
        raise RocprofFlagError(f"flag value must be an unsigned integer: {value!r}")


def _validate_positive_int(value: str, cwd: Path) -> None:
    _validate_non_negative_int(value, cwd)
    if int(value) <= 0:
        raise RocprofFlagError(f"flag value must be > 0: {value!r}")


def _validate_att_simd_select(value: str, cwd: Path) -> None:
    del cwd
    if value.startswith("-"):
        raise RocprofFlagError(f"flag value must not be another flag: {value!r}")
    if value.startswith("0x"):
        try:
            int(value, 16)
        except ValueError as e:
            raise RocprofFlagError(f"invalid hex value for --att-simd-select: {value!r}") from e
        return
    if not value.isdigit():
        raise RocprofFlagError(
            f"invalid value for --att-simd-select: {value!r}"
        )


def _validate_pmc_value(value: str, cwd: Path) -> None:
    del cwd
    if not value or value.startswith("-"):
        raise RocprofFlagError(f"invalid --pmc value: {value!r}")
    counters = [tok for tok in re.split(r"[\s,]+", value) if tok]
    if not counters:
        raise RocprofFlagError(f"invalid --pmc value: {value!r}")
    for counter in counters:
        if not _COUNTER_VALUE_RE.fullmatch(counter):
            raise RocprofFlagError(f"invalid counter token in --pmc value: {counter!r}")


def _validate_att_activity(value: str, cwd: Path) -> None:
    del cwd
    if not value or value.startswith("-"):
        raise RocprofFlagError(f"invalid --att-activity value: {value!r}")
    if not _COUNTER_VALUE_RE.fullmatch(value):
        raise RocprofFlagError(f"invalid --att-activity value: {value!r}")


_VALUE_TAKING_FLAGS = {
    "--att-library-path": _validate_external_path_value,
    "--att-target-cu": _validate_non_negative_int,
    "--att-simd-select": _validate_att_simd_select,
    "--att-buffer-size": _validate_positive_int,
    "--att-activity": _validate_att_activity,
    "--pmc": _validate_pmc_value,
    "-d": _validate_path_flag_value,
    "--output-dir": _validate_path_flag_value,
    "-o": _validate_path_flag_value,
    "--output": _validate_path_flag_value,
    "--pid": _validate_positive_int,
}


def _validate_argv(argv: List[str], cwd: Path) -> None:
    """Sanitize + allowlist-check every token and every value-taking flag."""
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
        if not (tok.startswith("--") or tok.startswith("-")):
            raise RocprofFlagError(
                f"unexpected non-flag token before '--': {tok!r}"
            )

        key, has_inline_value, inline_value = tok.partition("=")
        if key not in _ROCPROFV3_FLAGS:
            raise RocprofFlagError(f"rocprofv3 flag not in allowlist: {tok!r}")

        validator = _VALUE_TAKING_FLAGS.get(key)
        if validator is None:
            if has_inline_value:
                raise RocprofFlagError(f"flag does not take a value: {tok!r}")
            i += 1
            continue

        if has_inline_value:
            value = inline_value
        else:
            if key == "--pmc":
                value_tokens: list[str] = []
                j = i + 1
                while j < len(rocprof_tokens):
                    next_tok = rocprof_tokens[j]
                    if next_tok == "--" or next_tok.startswith("-"):
                        break
                    value_tokens.append(next_tok)
                    j += 1
                if not value_tokens:
                    raise RocprofFlagError(f"flag requires a value: {tok!r}")
                value = " ".join(value_tokens)
                i = j
                validator(value, cwd)
                continue

            if i + 1 >= len(rocprof_tokens):
                raise RocprofFlagError(f"flag requires a value: {tok!r}")
            value = rocprof_tokens[i + 1]
            i += 1
        validator(value, cwd)
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
    _validate_argv(argv, cwd)
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
